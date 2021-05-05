import torch 
from torch import nn 
from torch.distributed import rpc 
from torch_geometric.nn import GCNConv, GATConv

from .dist_framework import DTGAE_Embed_Unit
from .dist_static import StaticEncoder

class StaticGCN(DTGAE_Embed_Unit):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super(StaticGCN, self).__init__()

        # Load in the data before initing params
        print("%s loading %d-%d" % (
            rpc.get_worker_info().name, 
            data_kws['start'], 
            data_kws['end'])
        )

        self.data = data_load(data_kws.pop("jobs"), **data_kws)

        # Params 
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
    
    '''
    Override parent's abstract inner_forward method
    '''
    def inner_forward(self, mask_enum):
        zs = []
        for i in range(self.data.T):
            zs.append(self.forward_once(mask_enum, i))

        return torch.stack(zs)

    '''
    Helper function to make __forward a little more readable 
    Just passes each time step through a 2-layer GCN with final tanh activation
    '''
    def forward_once(self, mask_enum, i):
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


def static_gcn_rref(loader, kwargs, h_dim, z_dim):
    return StaticEncoder(
        StaticGCN(loader, kwargs, h_dim, z_dim)
    )


class StaticGAT(StaticGCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, heads=3):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        self.c1 = GATConv(self.data.x_dim, h_dim, heads=heads)
        self.c2 = GATConv(h_dim*heads, z_dim, concat=False)