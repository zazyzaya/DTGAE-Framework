import torch 
from torch.distributed import rpc 

from .dist_framework import DTGAE_Encoder, DTGAE_Recurrent
from .dist_utils import _remote_method, _remote_method_async

class StaticEncoder(DTGAE_Encoder):
    
    def decode_all(self, zs):
        '''
        Given node embeddings, return edge likelihoods for 
        all subgraphs held by this model

        For static model, it's very simple. Just return the embeddings
        for ei[n] given zs[n]
        '''
        assert not zs.size(0) < self.module.data.T, \
            "%s was given fewer embeddings than it has time slices"\
            % rpc.get_worker_info().name

        assert not zs.size(0) > self.module.data.T, \
            "%s was given more embeddings than it has time slices"\
            % rpc.get_worker_info().name

        preds = []
        for i in range(self.module.data.T):
            preds.append(
                self.decode(self.module.data.eis[i], zs[i])
            )

        return preds

    def score_edges(self, z, partition, nratio):
        '''
        Given a set of Z embeddings, returns likelihood scores for all known
        edges, and randomly sampled negative edges
        '''
        n = self.module.data.get_negative_edges(partition, nratio)

        p_scores = []
        n_scores = []

        for i in range(len(z)):
            p = self.module.data.ei_masked(partition, i)

            p_scores.append(self.decode(p, z[i]))
            n_scores.append(self.decode(n[i], z[i]))

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
        ns = self.module.data.get_negative_edges(partition, nratio)

        for i in range(len(z)):
            tot_loss += self.bce(
                self.decode(self.module.data.ei_masked(partition, i), z[i]),
                self.decode(ns[i], z[i])
            )

        return tot_loss.true_divide(len(z))


class StaticRecurrent(DTGAE_Recurrent):
    def score_all(self, zs):
        '''
        Has the distributed models score and label all of their edges
        Need to change which zs are given to workers depending on if 
        predictive or static
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    StaticEncoder.decode_all,
                    self.gcns[i],
                    zs[start : end]
                )
            )
            start = end 

        scores = [f.wait() for f in futs]
        ys = [
            _remote_method(
                StaticEncoder.get_data_field,
                self.gcns[i],
                'ys'
            ) for i in range(self.num_workers)
        ]

        return scores, ys


    def loss_fn(self, zs, partition, nratio=1):
        '''
        Runs NLL on each worker machine given the generated embeds
        Need to change which zs are given to workers depending on if 
        predictive or static 
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    StaticEncoder.calc_loss,
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

        #return [f.wait() for f in futs]

    def score_edges(self, zs, partition, nratio=1):
        '''
        Gets edge scores from dist modules, and negative edges
        '''
        futs = []
        start = 0
    
        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    StaticEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end], 
                    partition, nratio
                )
            )
            start = end 

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)