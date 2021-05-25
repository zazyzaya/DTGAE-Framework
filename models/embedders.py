import torch 
from torch import nn 
from torch.nn import functional as F
from torch.distributed import rpc
from torch.distributed.rpc.api import get_worker_info 
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
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


class SAGE(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        # For layer 1 we can mult the features with W before 
        # we message pass, because we know all inputs are 1-hot
        # vectors and we're doing max pooling. 
        del self.c1 
        self.mp = MessagePassing(aggr='max')
        self.e_lin = nn.Linear(self.data.x_dim, h_dim)
        self.r_lin = nn.Linear(self.data.x_dim, h_dim)

        self.c2 = SAGEConv(h_dim, h_dim, aggr='max')

    def forward_once(self, mask_enum, i):
        ei = self.data.ei_masked(mask_enum, i)
        x = self.data.xs if not self.data.dynamic_feats \
            else self.data.xs[i]

        # Conv 1 we do the GCN way, multiplying feats to the weights
        # before they are propogated. This saves time, and is equiv 
        # to doing it the other way around so long as we max pool
        x_e = self.e_lin(x)
        x_e = self.mp.propagate(ei, x=x_e, size=None)
        x_r = self.r_lin(x)
        x = x_r + x_e
        x = F.normalize(x, p=2., dim=-1)

        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei)

        return self.tanh(x)


def static_sage_rref(loader, kwargs, h_dim, z_dim, **kws):
    return StaticEncoder(
        SAGE(loader, kwargs, h_dim, z_dim)
    )

def dynamic_sage_rref(loader, kwargs, h_dim, z_dim, head=False):
    return DynamicEncoder(
        SAGE(loader, kwargs, h_dim, z_dim), head
    )