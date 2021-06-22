import math
import torch 
from torch import nn 
from torch.nn import functional as F
from torch.distributed import rpc
from torch.distributed.rpc.api import get_worker_info 
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn.conv.message_passing import MessagePassing

from .dist_framework import DTGAE_Embed_Unit
from .dist_static import StaticEncoder
from .dist_dynamic import DynamicEncoder

class GCN(DTGAE_Embed_Unit):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super(GCN, self).__init__()

        # Load in the data before initing params
        # Note: passing None as the start or end data_kw skips the 
        # actual loading part, and just pulls the x-dim 
        print("%s loading %s-%s" % (
            rpc.get_worker_info().name, 
            str(data_kws['start']), 
            str(data_kws['end']))
        )

        self.data = data_load(data_kws.pop("jobs"), **data_kws)
        

        # Params 
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
    
    
    def inner_forward(self, mask_enum):
        '''
        Override parent's abstract inner_forward method
        '''
        zs = []
        for i in range(self.data.T):
            # Small optimization. Running each loop step as its own thread
            # is a tiny bit faster. Plus we have 28 threads/proc to work
            # with. May as well use some inter-op threads if we have em
            zs.append(
                torch.jit._fork(self.forward_once, mask_enum, i)
                #self.forward_once(mask_enum, i)
            )

        return torch.stack([torch.jit._wait(z) for z in zs])
        #return torch.stack(zs)

    
    def forward_once(self, mask_enum, i):
        '''
        Helper function to make inner_forward a little more readable 
        Just passes each time step through a 2-layer GCN with final tanh activation
        '''
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs 

        ei = self.data.ei_masked(mask_enum, i)
        ew = self.data.ew_masked(mask_enum, i)

        # Simple 2-layer GCN. Tweak if desired
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


# Added dummy **kws param so we can use the same constructor for dynamic
def static_gcn_rref(loader, kwargs, h_dim, z_dim, **kws):
    return StaticEncoder(
        GCN(loader, kwargs, h_dim, z_dim)
    )

def dynamic_gcn_rref(loader, kwargs, h_dim, z_dim, head=False):
    return DynamicEncoder(
        GCN(loader, kwargs, h_dim, z_dim), head
    )



class GAT(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, heads=3):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        self.c1 = GATConv(self.data.x_dim, h_dim, heads=heads)
        self.c2 = GATConv(h_dim*heads, z_dim, concat=False)

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs 

        ei = self.data.ei_masked(mask_enum, i)

        # Only difference is GATs can't handle edge weights
        x = self.c1(x, ei)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


def static_gat_rref(loader, kwargs, h_dim, z_dim, **kws):
    return StaticEncoder(
        GAT(loader, kwargs, h_dim, z_dim)
    )

def dynamic_gat_rref(loader, kwargs, h_dim, z_dim, head=False):
    return DynamicEncoder(
        GAT(loader, kwargs, h_dim, z_dim), head
    )

'''
The official PyTorch Geometric package does not actually follow the paper
This is problematic from both a performance standpoint, and an accuracy one
'''
class PoolSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        
        self.aggr_n = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        self.e_lin = nn.Linear(out_channels, out_channels)
        self.r_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, ei):
        x_e = self.aggr_n(x)
        x_e = self.propagate(ei, x=x_e, size=None)
        x_e = self.e_lin(x_e)

        x_r = self.r_lin(x)
        
        x = x_r + x_e
        x = F.normalize(x, p=2., dim=-1)
        return x


class SAGE(GAT):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        self.c1 = PoolSAGEConv(self.data.x_dim, h_dim)
        self.c2 = PoolSAGEConv(h_dim, z_dim)


def static_sage_rref(loader, kwargs, h_dim, z_dim, **kws):
    return StaticEncoder(
        SAGE(loader, kwargs, h_dim, z_dim)
    )

def dynamic_sage_rref(loader, kwargs, h_dim, z_dim, head=False):
    return DynamicEncoder(
        SAGE(loader, kwargs, h_dim, z_dim), head
    )

class GIN(GAT):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super().__init__(data_load, data_kws, h_dim, z_dim)
        
        self.mp = MessagePassing(aggr='add')
        self.mlp1 = nn.Sequential(
            nn.Linear(self.data.x_dim, h_dim),
            nn.RReLU(),
            nn.Linear(h_dim, h_dim),
            nn.RReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.RReLU(),
            nn.Linear(h_dim, z_dim),
            nn.RReLU()
        )
        
        # Implimenting c1 in a more efficient way 
        # to speed up message passing
        del self.c1 
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.c2 = GINConv(self.mlp2, train_eps=True)
        self.bn2 = nn.BatchNorm1d(z_dim)

    def forward_once(self, mask_enum, i):
        '''
        Assumes feats are one-hot mat catted to smaller feat
        vector
        '''
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs 

        num_nodes = x.size(0)
        ei = self.data.ei_masked(mask_enum, i)
        
        # Building an adj mat is equivilant to message passing
        # the node ids to each other (and a lot faster)
        A = torch.zeros(num_nodes, num_nodes)
        A[ei[0], ei[1]] = 1
        
        # Use message passing on remaining feats
        if x.size(1) > num_nodes:
            feats = x[:, num_nodes:]
            x = self.mp.propagate(ei, x=feats, size=None)
            x = torch.cat([A,x], dim=1)
        else:
            x = A
    
        # Finally, run through MLP and we have accomplished
        # GIN-0 manually, but a little faster than the module does
        x = self.mlp1(x)
        x = self.bn1(x)

        x = self.drop(x)

        # Now that vectors are a managable size, use the builtin GIN conv
        x = self.c2(x, ei)
        x = self.bn2(x)

        return torch.tanh(x)

def static_gin_rref(loader, kwargs, h_dim, z_dim, **kws):
    return StaticEncoder(
        GIN(loader, kwargs, h_dim, z_dim)
    )

def dynamic_gin_rref(loader, kwargs, h_dim, z_dim, head=False):
    return DynamicEncoder(
        GIN(loader, kwargs, h_dim, z_dim), head
    )