from copy import deepcopy

import torch 
from torch import nn
from torch.distributed import rpc 
from torch.nn.parallel import DistributedDataParallel as DDP

from .dist_utils import _remote_method, _remote_method_async, _param_rrefs

'''
Wrapper class for the DDP class that holds the data this module will operate on
as well as a clone of the module itself
    
    module: a torch.module subclass
    **kwargs: any args for the DDP constructor

Requirements: module must have a field called data containing all time slices it will
operate on
'''
class DTGAE_Encoder(DDP):
    def __init__(self, module: torch.nn.Module, **kwargs):
        super().__init__(module, **kwargs)

    '''
    This method is inacceessable in the wrapped model by default
    '''
    def train(self, mode=True):
        self.module.train(mode=mode)

    '''
    Put different data on worker. Must be called before work can be done
    '''
    def load_new_data(self, loader, kwargs):
        print(rpc.get_worker_info().name + ": Reloading %d - %d" % (kwargs['start'], kwargs['end']))
        
        jobs = kwargs.pop('jobs')
        self.module.data = loader(jobs, **kwargs)
        return True

    '''
    Return some field from this worker's data object
    '''
    def get_data_field(self, field):
        return self.module.data.__getattribute__(field)

    '''
    Run an arbitrary function using this machine
    '''
    def run_arbitrary_fn(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    '''
    Given a single edge list and embeddings, return the dot product
    likelihood of each edge
    '''
    def decode(self, e,z):
        src,dst = e 
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    '''
    Computes binary cross entropy given a tensor of true edge encodings
    and false edge encodings
    '''
    def bce(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (pos_loss + neg_loss) * 0.5


    '''
    Rather than sending edge index to master, calculate loss 
    on workers all at once 
    '''
    def calc_loss(self, z, partition, nratio):
        raise NotImplementedError

    '''
    Given node embeddings, return edge likelihoods for 
    all subgraphs held by this model

    Implimented differently for predictive and static models
    '''
    def decode_all(self, zs):
        raise NotImplementedError

    
    '''
    Scores all known edges and randomly sampled non-edges
    '''
    def score_edges(self, z, partition, nratio):
        raise NotImplementedError


'''
Abstract class for master module that holds all workers
and calculates loss

    rnn: An RNN-like module that accepts 4D tensors
    remote_rrefs: a list of RRefs to DTGAE_Workers

'''
class DTGAE_Recurrent(nn.Module):
    def __init__(self, rnn: nn.Module, remote_rrefs: list):
        super(DTGAE_Recurrent, self).__init__()

        self.gcns = remote_rrefs
        self.rnn = rnn 

        self.num_workers = len(self.gcns)
        self.len_from_each = []

        # Used for LR when classifying anomalies
        self.cutoff = 0.5


    '''
    First have each worker encode their data, then run the embeddings through the RNN 
    '''
    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        futs = self.encode(mask_enum, no_grad)

        # Run through RNN as embeddings come in 
        # Also prevents sequences that are super long from being encoded
        # all at once. (This is another reason to put extra tasks on the
        # workers with higher pids)
        zs = []
        for f in futs:
            z, h0 = self.rnn(
                f.wait(),
                h0, include_h=True
            )
            zs.append(z)
        
        #zs = [f.wait() for f in futs]

        # May as well do this every time, not super expensive
        self.len_from_each = [
            embed.size(0) for embed in zs
        ]
        zs = torch.cat(zs, dim=0)

        #zs, h0 = self.rnn(torch.cat(zs, dim=0), h0, include_h=True)

        if include_h:
            return zs, h0 
        else:
            return zs

    '''
    Tell each remote GCN to encode their data. Data lives there to minimise 
    net traffic 
    '''
    def encode(self, mask_enum, no_grad):
        embed_futs = []
        
        for i in range(self.num_workers):    
            embed_futs.append(
                _remote_method_async(
                    DDP.forward, 
                    self.gcns[i],
                    mask_enum, no_grad
                )
            )

        return embed_futs


    '''
    Distributed optimizer needs RRefs to params rather than the literal
    locations of them that you'd get with self.parameters() 
    '''
    def parameter_rrefs(self):
        params = []
        for rref in self.gcns: 
            params.extend(
                _remote_method(
                    _param_rrefs, rref
                )
            )
        
        params.extend(_param_rrefs(self.rnn))
        return params

    '''
    Makes a copy of the current state dict as well as 
    the distributed GCN state dict (just worker 0)
    '''
    def save_states(self):
        gcn = _remote_method(
            DDP.state_dict, self.gcns[0]
        )

        return gcn, deepcopy(self.state_dict())

    '''
    Given the state dict for one GCN and the RNN load them
    into the dist and local models
    '''
    def load_states(self, gcn_state_dict, rnn_state_dict):
        self.load_state_dict(rnn_state_dict)
        
        jobs = []
        for rref in self.gcns:
            jobs.append(
                _remote_method_async(
                    DDP.load_state_dict, rref, 
                    gcn_state_dict
                )
            )

        [j.wait() for j in jobs]

    
    '''
    Propogate mode to all workers
    '''
    def train(self, mode=True):
        super(DTGAE_Recurrent, self).train() 
        [_remote_method(
            DTGAE_Encoder.train,
            self.gcns[i],
            mode=mode
        ) for i in range(self.num_workers)]

    def eval(self):
        super(DTGAE_Recurrent, self).train(False)
        [_remote_method(
            DTGAE_Encoder.train,
            self.gcns[i],
            mode=False
        ) for i in range(self.num_workers)]

    '''
    Has the distributed models score and label all of their edges
    Need to change which zs are given to workers depending on if 
    predictive or static
    '''
    def score_all(self, zs):
        raise NotImplementedError

    '''
    Runs NLL on each worker machine given the generated embeds
    Need to change which zs are given to workers depending on if 
    predictive or static 
    '''
    def loss_fn(self, zs, partition, nratio=1):
        raise NotImplementedError

    '''
    Gets edge scores from dist modules, and negative edges
    '''
    def score_edges(self, zs, partition, nratio=1):
        raise NotImplementedError


'''
Demonstrates how forward must be called on embedding units 
Data must live in the models, and must be masked via enums passed to them
'''
class DTGAE_Embed_Unit(nn.Module):
    def inner_forward(self, mask_enum):
        raise NotImplementedError

    def forward(self, mask_enum, no_grad):
        if no_grad:
            with torch.no_grad():
                return self.inner_forward(mask_enum)
        
        return self.inner_forward(mask_enum)