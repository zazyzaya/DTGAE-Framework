import torch 
from torch_geometric import Data 

'''
Special data object that the dist_framework uses
'''
class TData(Data):
    # Enum like for masked function used by worker processes
    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 2

    def __init__(self, eis, xs, masks, dynamic_feats=False, **kwargs):
        super(TData, self).__init__(**kwargs)
        
        # Required fields
        self.eis = eis 
        self.T = len(eis)
        self.xs = xs 
        self.masks = masks 
        self.dynamic_feats=dynamic_feats

    '''
    Returns masked ei/ew at timestep t
    Assumes it will only be called on tr or val data 
    (i.e. test data only applies to certain time steps)
    '''
    def ei_masked(self, enum, t):
        return self.eis[t][:, self.masks[t]] if enum == self.TRAIN \
            else self.eis[t][:, ~self.masks[t]]

    def ew_masked(self, enum, t):
        return self.ews[t][self.masks[t]] if enum == self.TRAIN \
            else self.ews[t][~self.masks[t]]

    
    def get_negative_edges(self, enum, nratio=1):
        for t in range(self.T):
            # TODO
            pass 