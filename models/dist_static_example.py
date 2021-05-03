import torch 
from torch import nn 
from torch_geometric.nn import GCNConv

from .dist_framework import DTGAE_Embed_Unit
from .dist_static_framework import StaticEncoder, StaticRecurrent

class StaticGCN(DTGAE_Embed_Unit):
    def __init__(self, x_dim, h_dim, z_dim):
        super(StaticGCN, self).__init__()

        # Params 
        self.c1 = GCNConv(x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()

        # Must be loaded before running model
        self.data = None 
    
    '''
    Override parent's abstract __forward method
    '''
    def __forward(self, mask_enum):
        assert not isinstance(self.data, None),\
            "Must load data onto workers before calling forward"

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