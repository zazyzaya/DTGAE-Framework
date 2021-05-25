import torch 
from torch.distributed import rpc 

from .dist_framework import DTGAE_Encoder, DTGAE_Recurrent
from .dist_utils import _remote_method, _remote_method_async

class DynamicEncoder(DTGAE_Encoder):
    def __init__(self, module: torch.nn.Module, head: bool, **kwargs):
        super().__init__(module, **kwargs)
        self.is_head = 1 if head else 0
        if self.is_head:
            print("%s is head" % rpc.get_worker_info().name)


    def decode_all(self, zs):
        '''
        Given node embeddings, return edge likelihoods for 
        all subgraphs held by this model

        For dynamic model, assume we are given embeddings 
        for timesteps -1 to N-1 (where Z_{-1} is a dummy value
        to be ignored if this is worker 0) with which to predict
        E_1 to E_N
        '''
        preds = []
        for i in range(self.module.data.T-self.is_head):
            preds.append(
                self.decode(self.module.data.eis[i+self.is_head], zs[i+self.is_head])
            )

        return preds

    
    def score_edges(self, z, partition, nratio):
        '''
        Given a set of Z embeddings, returns likelihood scores for all known
        edges, and randomly sampled negative edges
        '''
        
        # Skip neg edges for E_0 if this is the head node
        n = self.module.data.get_negative_edges(
            partition, nratio=nratio, start=self.is_head
        )

        p_scores = []
        n_scores = []

        # Head worker is given a dummy Z_{-1} to be skipped so
        # Z_0 is aligned with E_1 to be used for prediction
        for i in range(self.is_head, len(z)):
            p = self.module.data.ei_masked(partition, i)

            p_scores.append(self.decode(p, z[i]))
            n_scores.append(self.decode(n[i-self.is_head], z[i]))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores


    def calc_loss(self, z, partition, nratio):
        '''
        Sum up all of the loss per time step, then average it. For some reason
        this works better than running score edges on everything at once. It's better
        to run BCE per time step rather than all at once
        '''
        tot_loss = torch.zeros(1)
        ns = self.module.data.get_negative_edges(
            partition, nratio=nratio, start=self.is_head
        )

        for i in range(self.is_head, len(z)):
            tot_loss += self.bce(
                self.decode(self.module.data.ei_masked(partition, i), z[i]),
                self.decode(ns[i-self.is_head], z[i])
            )

        return tot_loss.true_divide(len(z))


class DynamicRecurrent(DTGAE_Recurrent):
    '''
    With very minimal changes to the static version, we can make the 
    module work for predictive tasks as well. 
    
    The main changes involve which embeddings we send to 
    which workers, as they are now trying to predict future time steps.
    As such, if a worker holds edge lists at times t-t*n we send it embeddings
    generated for steps t-1 to t*n-1 

    In the edge case of the worker holding timestep 0 (which would recieve an 
    embedding for -1 which doesn't exist), we provide a dummy -1 embedding that
    that worker knows to ignore
    '''
    def score_all(self, zs):
        '''
        Has the distributed models score and label all of their edges
        Need to change which zs are given to workers depending on if 
        predictive or static

        For dynamic, we append a dummy Z_{-1} value that's ignored in 
        the workers, to align embeds with future edge lists 
        '''
        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )
        
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    DynamicEncoder.decode_all,
                    self.gcns[i],
                    zs[start : end]
                )
            )
            start = end 

        scores = sum([f.wait() for f in futs], [])
        ys = [
            _remote_method(
                DynamicEncoder.get_data_field,
                self.gcns[i],
                'ys'
            ) for i in range(self.num_workers)
        ]

        # Remove labels for edgelist 0 as it has no embeddings 
        # it can be compared to 
        ys[0] = ys[0][1:]
        scores = torch.cat(scores, dim=0)
        ys = torch.cat(sum(ys, []), dim=0)

        print(scores.size())
        print(ys.size())
        print(
            torch.stack([ys[:10], scores[:10]], dim=1)
        )

        return scores, ys



    def loss_fn(self, zs, partition, nratio=1):
        '''
        Runs NLL on each worker machine given the generated embeds
        Need to change which zs are given to workers depending on if 
        predictive or static 

        For dynamic, only difference is the dummy Z_{-1} value
        '''
        
        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )
        
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    DynamicEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end 

        tot_loss = torch.zeros(1)
        for f in futs:
            tot_loss += f.wait()

        return [tot_loss.true_divide(sum(self.len_from_each))]

    
    def score_edges(self, zs, partition, nratio=1):
        '''
        Gets edge scores from dist modules, and negative edges
        Same deal, just add the padding to the zs
        '''

        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )
        
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    DynamicEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end], 
                    partition, nratio
                )
            )
            start = end 

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)