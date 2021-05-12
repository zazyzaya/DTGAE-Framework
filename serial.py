from copy import deepcopy

import torch 
from torch import nn 
from torch.optim import Adam 
from torch_geometric.nn import GCNConv

from models.recurrent import GRU 
import loaders.load_lanl as ld 
from loaders.tdata import TData
from utils import get_optimal_cutoff, get_score

class TGCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(TGCN, self).__init__()

        # Field for anom detection
        self.cutoff = None

        # Topological encoder
        self.c1 = GCNConv(x_dim, h_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        self.c2 = GCNConv(h_dim, h_dim)
        self.tanh = nn.Tanh() 

        # Temporal encoder
        self.rnn = GRU(h_dim, h_dim, z_dim)

    
    def forward(self, data, mask, h0=None):
        zs = []
        for i in range(data.T):
            zs.append(
                self.forward_once(
                    data.ei_masked(mask, i),
                    data.ew_masked(mask, i),
                    data.xs
                )
            )

        return self.rnn(torch.stack(zs), h0, include_h=True)
        
    def forward_once(self, ei, ew, x):
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)
        return self.tanh(x)


    def decode(self, e, zs):
        src,dst = e 
        return torch.sigmoid(
            (zs[src] * zs[dst]).sum(dim=1)
        )

    def bce(self, pos, neg):
        EPS = 1e-8
        ps = -torch.log(pos+EPS).mean()
        ns = -torch.log(1-neg+EPS).mean()

        return (ps + ns) * 0.5

    def calc_loss(self, p,n,zs):
        tot_loss = torch.zeros(1)

        for i in range(len(zs)):
            tot_loss += self.bce(
                self.decode(p[i], zs[i]),
                self.decode(n[i], zs[i])
            )

        return tot_loss.true_divide(len(zs))

    def calc_scores(self, p,n,zs):
        ps, ns = [], []
        for i in range(len(zs)):
            ps.append(self.decode(p[i], zs[i]))
            ns.append(self.decode(n[i], zs[i]))
        
        return torch.cat(ps, dim=0), torch.cat(ns, dim=0)


EPOCHS = 1500
PATIENCE = 5
MIN = 25

def train(model: TGCN, data: TData):
    opt = Adam(model.parameters(), lr=0.001)

    best = (None, 0)
    no_progress = 0

    for e in range(EPOCHS):
        model.train()
        opt.zero_grad()
        
        zs, _ = model.forward(data, TData.TRAIN)
        
        p = [data.ei_masked(TData.TRAIN, i) for i in range(data.T)]
        n = data.get_negative_edges(TData.TRAIN, nratio=10)
        loss = model.calc_loss(p,n,zs)

        loss.backward()
        opt.step() 

        print("[%d] Loss: %0.4f" % (e, loss.item()))

        model.eval()
        with torch.no_grad():
            zs, _ = model.forward(data, TData.TRAIN)
            
            p = [data.ei_masked(TData.VAL, i) for i in range(data.T)]
            n = data.get_negative_edges(TData.VAL, nratio=10)
            p,n = model.calc_scores(p,n,zs)

            auc,ap = get_score(p,n)
            print("\tVal  AUC: %0.4f  AP: %0.4f" % (auc,ap), end='')

            tot = auc+ap
            if tot > best[1]:
                best = (deepcopy(model), tot)
                print("*")
            else:
                print()
                if e >= MIN:
                    no_progress += 1 

            if no_progress == PATIENCE:
                print("Early stopping!")
                break 

    model = best[0]
    _, h0 = model.forward(data, TData.ALL)

    return model, h0


def find_cutoff(model: TGCN, data: TData, h0: torch.Tensor):
    p = data.eis
    n = data.get_negative_edges(TData.ALL, nratio=10)
    
    model.eval()
    with torch.no_grad():
        zs, h0 = model.forward(data, TData.ALL, h0=h0)
        p,n = model.calc_scores(p,n,zs)

    model.cutoff = get_optimal_cutoff(p,n,fw=0.6)
    return h0 


def test(model: TGCN, data: TData, h0: torch.Tensor):
    model.eval()
    with torch.no_grad():
        zs, _ = model.forward(data, TData.ALL, h0=h0)

        scores = torch.cat(
            [model.decode(data.eis[i], zs[i])
            for i in range(data.T)],
            dim=0
        )

    y = torch.cat(data.ys)
    y_hat = torch.zeros(scores.size(0))
    y_hat[scores <= model.cutoff] = 1

    tpr = y_hat[y==1].mean() * 100
    fpr = y_hat[y==0].mean() * 100
    
    tp = y_hat[y==1].sum()
    fp = y_hat[y==0].sum()
    
    fn = (y==1).sum() - tp
    f1 = tp / (tp + 0.5*(fp + fn))

    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)

DELTA=(60**2) * 2
TR_START=0
TR_END=ld.DATE_OF_EVIL_LANL-DELTA*2

VAL_START=TR_END
VAL_END=VAL_START+DELTA*2

TE_START=ld.DATE_OF_EVIL_LANL
TE_END = 740104 # First 100 anoms

def run_all():
    data = ld.load_lanl_dist(8, start=TR_START, end=TR_END, delta=DELTA)
    model = TGCN(data.xs.size(1), 32, 16)

    model, h0 = train(model, data)

    data = ld.load_lanl_dist(2, start=VAL_START, end=VAL_END, delta=DELTA)
    h0 = find_cutoff(model, data, h0)

    data = ld.load_lanl_dist(8, start=TE_START, end=TE_END, delta=DELTA, is_test=True)
    test(model, data, h0)


if __name__ == '__main__':
    torch.set_num_threads(8)
    run_all()