import numpy as np
import torch 
from torch_geometric.data import Data 

'''
Special data object that the dist_framework uses
'''
class TData(Data):
    # Enum like for masked function used by worker processes
    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 2

    def __init__(self, eis, xs, ys, masks, ews=None, **kwargs):
        super(TData, self).__init__(**kwargs)
        
        # Required fields for models to use this 
        self.eis = eis 
        self.T = len(eis)
        self.xs = xs 
        self.masks = masks 
        self.dynamic_feats = isinstance(xs, list)
        self.ews = ews 
        self.ys = ys 
        self.is_test = not isinstance(ys, None.__class__)

        # Makes finding sizes of positive samples a little easier
        self.ei_sizes = [
            (
                self.ei_masked(self.TRAIN, t).size(1), 
                self.ei_masked(self.VAL, t).size(1),
                self.eis[t].size(1)
            ) 
            for t in range(self.T)
        ]

        if self.dynamic_feats:
            self.num_nodes = max([x.size(0) for x in xs])
            self.x_dim = xs[0].size(1)
        else:
            self.num_nodes = xs.size(0)
            self.x_dim = xs.size(1)

    '''
    Returns masked ei/ew at timestep t
    Assumes it will only be called on tr or val data 
    (i.e. test data is the entirity of certain time steps)
    '''
    def ei_masked(self, enum, t):
        if enum == self.TEST or self.is_test:
            return self.eis[t]
        
        return self.eis[t][:, self.masks[t]] if enum == self.TRAIN \
            else self.eis[t][:, ~self.masks[t]]

    def ew_masked(self, enum, t):
        if isinstance(self.ews, None.__class__):
            return None

        if enum == self.TEST or self.is_test:
            return self.ews[t]

        return self.ews[t][self.masks[t]] if enum == self.TRAIN \
            else self.ews[t][~self.masks[t]]

    
    def get_negative_edges(self, enum, nratio=1):
        negs = []
        
        for t in range(self.T):
            if enum == self.TRAIN:
                pos = self.ei_masked(enum, t)
            else:
                pos = self.eis[t]
            
            num_pos = self.ei_sizes[t][enum]
            negs.append(
                fast_negative_sampling(
                    pos, 
                    int(num_pos*nratio),
                    self.num_nodes
                )
            )

        return negs


'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin 
torch geo method)
'''
def fast_negative_sampling(edge_list, batch_size, num_nodes, oversample=1.25):    
    # For faster membership checking
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

    el1d = el_hash(edge_list).numpy()
    neg = np.array([[],[]])

    while(neg.shape[1] < batch_size):
        maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample)))
        neg_hash = el_hash(maybe_neg)
        
        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
            axis=1
        )

    # May have gotten some extras
    neg = neg[:, :batch_size]
    return torch.tensor(neg).long()