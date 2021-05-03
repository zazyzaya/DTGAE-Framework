import torch 
from torch.distributed import rpc 

from .dist_framework import DTGAE_Encoder, DTGAE_Recurrent
from .utils import _remote_method, _remote_method_async

class StaticEncoder(DTGAE_Encoder):
    '''
    Given node embeddings, return edge likelihoods for 
    all subgraphs held by this model

    For static model, it's very simple. Just return the embeddings
    for ei[n] given zs[n]
    '''
    def decode_all(self, zs):
        assert not zs.size(0) < self.module.data.T, \
            "%s was given fewer embeddings than it has time slices"\
            % rpc.get_worker_info().name

        assert not zs.size(0) > self.module.data.T, \
            "%s was given more embeddings than it has time slices"\
            % rpc.get_worker_info().name

        preds = []
        for i in range(self.module.data.T):
            preds.append(
                self.decode(self.module.eis[i], zs[i])
            )

        return preds


class StaticRecurrent(DTGAE_Recurrent):
    '''
    Has the distributed models score and label all of their edges
    Need to change which zs are given to workers depending on if 
    predictive or static
    '''
    def score_all(self, zs):
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            _remote_method_async(
                StaticEncoder.decode_all,
                self.gcns[i],
                zs[start : end]
            )
            start = end 

        scores = sum([f.wait() for f in futs], [])
        ys = [
            _remote_method(
                StaticEncoder.get_data_field,
                self.gcns[i],
                'ys'
            )
        ]

        return scores, sum(ys, [])

    '''
    Runs NLL on each worker machine given the generated embeds
    Need to change which zs are given to workers depending on if 
    predictive or static 
    '''
    def loss_fn(self, zs, partition, nratio=1):
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            _remote_method_async(
                StaticEncoder.calc_loss,
                self.gcns[i],
                zs[start : end]
            )
            start = end 

        return torch.stack([f.wait() for f in futs]).mean()

    '''
    Gets edge scores from dist modules, and negative edges
    '''
    def score_edges(self, zs, partition, nratio=1):
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            _remote_method_async(
                StaticEncoder.score_edges,
                self.gcns[i],
                zs[start : end], partition, nratio
            )
            start = end 

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)