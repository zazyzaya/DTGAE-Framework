import os
import pickle
import time

import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

import loaders.load_lanl as ld
from loaders.tdata import TData
from models.dist_static import StaticEncoder, StaticRecurrent 
from models.dist_utils import _remote_method_async, _remote_method
from models.dist_static_modules import static_gcn_rref
from models.recurrent import GRU 
from utils import get_score, get_optimal_cutoff

DEFAULTS = {
    'h_size': 32, 
    'z_size': 16,
    'lr': 0.01,
    'epochs': 1500,
    'min': 5,
    'patience': 5,
    'n_gru': 1,
    'nratio': 10,
    'val_nratio': 1,
    'delta': 2
}

WORKERS=4
W_THREADS=2
M_THREADS=1

DELTA=int((60**2) * DEFAULTS['delta'])
TR_START=0
TR_END=ld.DATE_OF_EVIL_LANL-DELTA*2

val = (TR_END - TR_START) // 20
VAL_START = TR_END-val
VAL_END = TR_END
TR_END = VAL_START

TE_START=ld.DATE_OF_EVIL_LANL
#TE_END = 228642 # First 20 anoms
TE_END = 740104 # First 100 anoms
#TE_END = 1089597 # First 500 anoms
#TE_END = 5011200 # Full

TE_DELTA=DELTA

torch.set_num_threads(1)

'''
Constructs params for data loaders
'''
def get_work_units(num_workers, start, end, delta, isTe):
    slices_needed = (end-start) // delta
    slices_needed += 1

    # Puts minimum tasks on each worker with some remainder
    per_worker = [slices_needed // num_workers] * num_workers 

    remainder = slices_needed % num_workers 
    if remainder:
        # Put remaining tasks on last workers since it's likely the 
        # final timeslice is stopped halfway (ie it's less than a delta
        # so giving it extra timesteps is more likely okay)
        for i in range(num_workers, num_workers-remainder, -1):
            per_worker[i-1]+=1 

    print("Tasks: %s" % str(per_worker))
    kwargs = []
    prev = start
    for i in range(num_workers):
            end_t = min(prev + delta*per_worker[i], end)
            kwargs.append({
                'start': prev,
                'end': end_t,
                'delta': delta, 
                'is_test': isTe,
                'jobs': min(W_THREADS, 8)
            })
            prev = end_t

    return kwargs
    

def init_workers(num_workers, h_dim, start, end, delta, isTe):
    kwargs = get_work_units(num_workers, start, end, delta, isTe)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                static_gcn_rref,
                args=(ld.load_lanl_dist, kwargs[i], h_dim, h_dim)
            )
        )

    return rrefs

def init_procs(rank, world_size, tr_args=DEFAULTS):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:42068'

    # Master (RNN module)
    if rank == world_size-1:
        torch.set_num_threads(M_THREADS)

        # Master gets 16 threads and 4x4 threaded workers
        # In theory, only 16 threads should run at a time while
        # master sleeps, waiting on worker procs
        #torch.set_num_threads(16)

        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

        rrefs = init_workers(
            world_size-1, tr_args['h_size'], 
            TR_START, TR_END, DELTA, False
        )

        model, zs, h0 = train(rrefs, tr_args)
        get_cutoff(model, h0, tr_args)
        test(model, zs, h0, rrefs, tr_args)

    # Slaves
    else:
        # If there are 4 workers, give them each 4 threads 
        # (Total 16 is equal to serial model)
        torch.set_num_threads(W_THREADS)
        
        # Slaves are their own process group. This allows
        # DDP to work between these processes
        dist.init_process_group(
            'gloo', rank=rank, 
            world_size=world_size-1
        )

        rpc.init_rpc(
            'worker'+str(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

    # Block until all procs complete
    rpc.shutdown()


def train(rrefs, kwargs):
    rnn = GRU(kwargs['h_size'], kwargs['h_size'], kwargs['z_size'], kwargs['n_gru'])
    model = StaticRecurrent(rnn, rrefs)

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    times = []
    best = (None, 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        # Get loss and send backward
        model.train()
        with dist_autograd.context() as context_id:
            st = time.time()
            zs = model.forward(TData.TRAIN)
            loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'])

            print("backward")
            dist_autograd.backward(context_id, loss)
            
            print("step")
            opt.step(context_id)

            elapsed = time.time()-st 
            times.append(elapsed)
            l = torch.stack(loss).sum()
            print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            auc,ap = get_score(p,n)

            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')
            tot = ap+auc

            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1 

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break 

    model.load_states(best[0][0], best[0][1])
    zs, h0 = model(TData.TEST, include_h=True)

    states = {'gcn': best[0][0], 'rnn': best[0][1]}
    f = open('model_save.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % (sum(times)/len(times)) )
    
    return model, zs[-1], h0


'''
Given a trained model, generate the optimal cutoff point using
the validation data
'''
def get_cutoff(model, h0, kwargs):
    # First load validation data onto one of the GCNs
    _remote_method(
        StaticEncoder.load_new_data,
        model.gcns[0],
        ld.load_lanl_dist,
        {
            'start': VAL_START,
            'end': VAL_END,
            'delta': DELTA,
            'jobs': 2,
            'is_test': False
        }
    )

    # Then generate GCN embeds
    model.eval()
    zs = _remote_method(
        StaticEncoder.forward,
        model.gcns[0], 
        TData.ALL,
        True
    )

    # Finally, generate actual embeds
    with torch.no_grad():
        zs = model.rnn(zs, h0)

    # Then score them
    p,n = _remote_method(
        StaticEncoder.score_edges, 
        model.gcns[0],
        zs, TData.ALL,
        kwargs['val_nratio']
    )

    # Finally, figure out the optimal cutoff score
    model.cutoff = get_optimal_cutoff(p,n,fw=0.6)
    print()


def test(model, zs, h0, rrefs, kwargs):
    # Load train data into workers
    ld_args = get_work_units(len(rrefs), TE_START, TE_END, DELTA, True)
    
    print("Loading test data")
    futs = [
        _remote_method_async(
            StaticEncoder.load_new_data,
            rrefs[i], 
            ld.load_lanl_dist, 
            ld_args[i]
        ) for i in range(len(rrefs))
    ]

    # Wait until all workers have finished
    [f.wait() for f in futs]

    with torch.no_grad():
        model.eval()
        zs = model(TData.TEST, h0=h0, no_grad=True)

    # Scores all edges and matches them with name/timestamp
    print("Scoring")
    scores, labels = model.score_all(zs)

    anoms = scores[labels==1].sort()[0]

    # Classify using cutoff from earlier
    classified = torch.zeros(labels.size())
    classified[scores <= model.cutoff] = 1

    default = torch.zeros(labels.size())
    default[scores <= 0.5] = 1

    tpr = classified[labels==1].mean() * 100
    fpr = classified[labels==0].mean() * 100
    
    tp = classified[labels==1].sum()
    fp = classified[labels==0].sum()
    
    fn = (labels==1).sum() - tp
    f1 = tp / (tp + 0.5*(fp + fn))

    dtpr = default[labels==1].mean() * 100
    dfpr = default[labels==0].mean() * 100
    
    dtp = default[labels==1].sum()
    dfp = default[labels==0].sum()
    
    dfn = (labels==1).sum() - dtp
    df1 = dtp / (dtp + 0.5*(dfp + dfn))

    print("Learned Cutoff %0.4f" % model.cutoff)
    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f\n" % f1)

    print("Default Cutoff %0.4f" % 0.5)
    print("TPR: %0.2f, FPR: %0.2f" % (dtpr, dfpr))
    print("TP: %d  FP: %d" % (dtp, dfp))
    print("F1: %0.8f\n" % df1)

    print("Top anom scored %0.04f" % anoms[0].item())
    print("Lowest anom scored %0.4f" % anoms[-1].item())
    print("Mean anomaly score: %0.4f" % anoms.mean().item())

    with open('out.txt', 'a') as f:
        f.write("Delta: %shrs, pred\n" % kwargs['delta'])
        f.write("Learned Cutoff %0.4f" % model.cutoff)
        f.write("TPR: %0.2f, FPR: %0.2f\n" % (tpr, fpr))
        f.write("TP: %d  FP: %d\n" % (tp, fp))
        f.write("F1: %0.8f\n\n" % f1)

        f.write("Default Cutoff %0.4f\n" % 0.5)
        f.write("TPR: %0.2f, FPR: %0.2f\n" % (dtpr, dfpr))
        f.write("TP: %d  FP: %d\n" % (dtp, dfp))
        f.write("F1: %0.8f\n" % df1)


if __name__ == '__main__':
    max_workers = (TR_END-TR_START) // DELTA 
    workers = min(max_workers, WORKERS)

    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )