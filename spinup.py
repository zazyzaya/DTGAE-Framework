import os
import pickle
import time

import torch 
import torch.distributed as dist 
import torch.distributed.rpc as rpc 
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam

import loaders.load_lanl as ld
from loaders.tdata import TData
from models.dist_framework import DTGAE_Encoder
from models.dist_static import StaticEncoder, StaticRecurrent 
from models.dist_dynamic import DynamicEncoder, DynamicRecurrent
from models.dist_utils import _remote_method_async, _remote_method
from models.embedders import static_gcn_rref
from models.recurrent import GRU 
from utils import get_score, get_optimal_cutoff, get_f1

DEFAULTS = {
    'lr': 0.01,
    'epochs': 1500,
    'min': 1,
    'patience': 5,
    'nratio': 1,
    'val_nratio': 1,
}

# Defaults
WORKER_ARGS = [32,32]
RNN_ARGS = [32,32,16,1]

WORKERS=4
W_THREADS=1
M_THREADS=2

TMP_FILE = 'tmp.dat'
SCORE_FILE = 'scores.txt'

# Callable that returns TData object
# method signature must match
# workers: int, start=int, end=int, delta=int, is_test=bool 
LOAD_FN = None

torch.set_num_threads(1)

'''
Constructs params for data loaders
'''
def get_work_units(num_workers, start, end, delta, isTe):
    slices_needed = (end-start) // delta
    slices_needed += 1

    # Puts minimum tasks on each worker with some remainder
    per_worker = [slices_needed // num_workers] * num_workers 

    remainder = slices_needed % num_workers 
    if remainder:
        # Put remaining tasks on last workers since it's likely the 
        # final timeslice is stopped halfway (ie it's less than a delta
        # so giving it extra timesteps is more likely okay)
        for i in range(num_workers, num_workers-remainder, -1):
            per_worker[i-1]+=1 

    # Only uncomment when running late at night
    # Don't be a thread-hog, fucko
    load_threads = W_THREADS*2 if isTe else W_THREADS
    #load_threads = W_THREADS

    # Make sure workers are collectively using at least 8 threads
    # since loading the data takes forever otherwise
    min_threads = min(8, load_threads*num_workers)
    t_per_worker = max(1, min_threads//num_workers)

    print("Tasks: %s" % str(per_worker))
    kwargs = []
    prev = start
    end_t = min(prev + delta*per_worker[0], end)
    for i in range(num_workers):
            kwargs.append({
                'start': prev,
                'end': end_t,
                'delta': delta, 
                'is_test': isTe,
                'jobs': t_per_worker
            })
            prev = end_t + 1
            end_t = min(prev-1 + delta*per_worker[i], end)

    return kwargs
    

def init_workers(num_workers, start, end, delta, isTe, worker_constructor, worker_args):
    kwargs = get_work_units(num_workers, start, end, delta, isTe)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                worker_constructor,
                args=(LOAD_FN, kwargs[i], *worker_args),
                kwargs={'head': i==0}
            )
        )

    return rrefs

def init_empty_workers(num_workers, worker_constructor, worker_args):
    empty = {'jobs': 0, 'start': None, 'end': None}
    
    rrefs = [
        rpc.remote(
            'worker'+str(i),
            worker_constructor,
            args=(LOAD_FN, empty, *worker_args),
            kwargs={'head': i==0}
        )
        for i in range(num_workers)
    ]

    return rrefs

def init_procs(rank, world_size, rnn_constructor, rnn_args, worker_constructor, worker_args, 
                times, just_test, fw, static, single_emb, run_speed_test, load_fn, manual, 
                tr_args=DEFAULTS):
    # DDP info
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42069'

    # RPC info
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:42068'

    # This is a lot easier than actually changing it in all the methods
    # at this point
    global LOAD_FN
    LOAD_FN = load_fn

    # Master (RNN module)
    if rank == world_size-1:
        torch.set_num_threads(M_THREADS)
        rpc.init_rpc(
            'master', rank=rank, 
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

        # Speed test doesn't need to train the model, 
        # just use random state it starts with and check 
        # how long it takes to run, then return that data
        if run_speed_test:
            rrefs = init_empty_workers(
                world_size-1, 
                worker_constructor, worker_args
            )

            rnn = rnn_constructor(*rnn_args)
            model = StaticRecurrent(rnn, rrefs) if static\
                else DynamicRecurrent(rnn, rrefs)

            stats = speed_test(model, rrefs, times)
            stats['delta'] = times['delta']


        # Evaluating a pre-trained model, so no need to train 
        elif just_test:
            rrefs = init_empty_workers(
                world_size-1, 
                worker_constructor, worker_args
            )

            rnn = rnn_constructor(*rnn_args)
            model = StaticRecurrent(rnn, rrefs) if static\
                else DynamicRecurrent(rnn, rrefs)

            states = pickle.load(open('model_save.pkl', 'rb'))
            model.load_states(states['gcn'], states['rnn'])
            h0 = states['h0']
            tpe = 0


        # Building and training a fresh model
        else:
            rrefs = init_workers(
                world_size-1, 
                times['tr_start'], times['tr_end'], times['delta'], False,
                worker_constructor, worker_args
            )

            model, h0, tpe = train(rrefs, tr_args, rnn_constructor, rnn_args, static)

        if not run_speed_test:
            h0, zs = get_cutoff(model, h0, times, tr_args, fw)
            
            if single_emb:
                stats = test_single_embed(zs, rrefs, times, model.cutoff)
            else:
                stats = test(model, h0, times, rrefs, manual=manual)

            stats['TPE'] = tpe

    # Slaves
    else:
        torch.set_num_threads(W_THREADS)
        
        # Slaves are their own process group. This allows
        # DDP to work between these processes
        dist.init_process_group(
            'gloo', rank=rank, 
            world_size=world_size-1
        )

        rpc.init_rpc(
            'worker'+str(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

    # Block until all procs complete
    rpc.shutdown()

    # Write output to a tmp file to get it back to the parent process
    if rank == world_size-1:
        pickle.dump(stats, open(TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)


def train(rrefs, kwargs, rnn_constructor, rnn_args, static):
    rnn = rnn_constructor(*rnn_args)
    model = StaticRecurrent(rnn, rrefs) if static \
        else DynamicRecurrent(rnn, rrefs)

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    times = []
    best = (None, 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        # Get loss and send backward
        model.train()
        with dist_autograd.context() as context_id:
            st = time.time()
            zs = model.forward(TData.TRAIN)
            loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'])

            print("backward")
            dist_autograd.backward(context_id, loss)
            
            print("step")
            opt.step(context_id)

            elapsed = time.time()-st 
            times.append(elapsed)
            l = torch.stack(loss).sum()
            print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        # Get validation info to prevent overfitting
        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            auc,ap = get_score(p,n)

            print("\tValidation: AP: %0.4f  AUC: %0.4f" 
                % (ap, auc), end='')
            tot = ap+auc

            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1 

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break 

    model.load_states(best[0][0], best[0][1])
    zs, h0 = model(TData.TEST, include_h=True)

    states = {'gcn': best[0][0], 'rnn': best[0][1], 'h0': h0}
    f = open('model_save.pkl', 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)
    
    return model, h0, tpe


'''
Given a trained model, generate the optimal cutoff point using
the validation data
'''
def get_cutoff(model, h0, times, kwargs, fw):
    # Weirdly, calling the parent class' method doesn't work
    # whatever. This is a hacky solution, but it works
    Encoder = StaticEncoder if isinstance(model, StaticRecurrent) \
        else DynamicEncoder

    # First load validation data onto one of the GCNs
    _remote_method(
        Encoder.load_new_data,
        model.gcns[0],
        LOAD_FN,
        {
            'start': times['tr_end'],
            'end': times['val_end'],
            'delta': times['delta'],
            'jobs': 2,
            'is_test': False
        }
    )

    # Then generate GCN embeds
    model.eval()
    zs = _remote_method(
        Encoder.forward,
        model.gcns[0], 
        TData.ALL,
        True
    )

    # Finally, generate actual embeds
    with torch.no_grad():
        zs, h0 = model.rnn(zs, h0, include_h=True)

    # Then score them
    p,n = _remote_method(
        Encoder.score_edges, 
        model.gcns[0],
        zs, TData.ALL,
        kwargs['val_nratio']
    )

    # Finally, figure out the optimal cutoff score
    model.cutoff = get_optimal_cutoff(p,n,fw=fw)
    print()
    return h0, zs[-1]


def test(model, h0, times, rrefs, manual=False):
    # For whatever reason, it doesn't know what to do if you call
    # the parent object's methods. Kind of defeats the purpose of 
    # using OOP at all IMO, but whatever
    Encoder = StaticEncoder if isinstance(model, StaticRecurrent) \
        else DynamicEncoder

    # Load train data into workers
    ld_args = get_work_units(
        len(rrefs), 
        times['te_start'], 
        times['te_end'],
        times['delta'], 
        True
    )

    print("Loading test data")
    futs = [
        _remote_method_async(
            Encoder.load_new_data,
            rrefs[i], 
            LOAD_FN, 
            ld_args[i]
        ) for i in range(len(rrefs))
    ]

    # Wait until all workers have finished
    [f.wait() for f in futs]

    with torch.no_grad():
        model.eval()
        s = time.time()
        zs = model.forward(TData.TEST, h0=h0, no_grad=True)
        ctime = time.time()-s

    # Scores all edges and matches them with name/timestamp
    print("Scoring")
    scores, labels = model.score_all(zs)

    if manual:
        labels = [
            _remote_method_async(
                Encoder.get_repr,
                rrefs[i],
                scores[i], 
                delta=times['delta']
            )
            for i in range(len(rrefs))
        ]
        labels = sum([l.wait() for l in labels], [])
        labels.sort(key=lambda x : x[0])

        with open(SCORE_FILE, 'w+') as f:
            cutoff_hit = False
            for l in labels:
                f.write(str(l[0].item()))
                f.write('\t')
                f.write(l[1])
                f.write('\n')

                if l[0] >= model.cutoff and not cutoff_hit:
                    f.write('-'*100 + '\n')
                    cutoff_hit = True

        return {}

    scores = torch.cat(sum(scores, []), dim=0)
    labels = torch.cat(sum(labels, []), dim=0)

    anoms = scores[labels==1].sort()[0]

    # Classify using cutoff from earlier
    classified = torch.zeros(labels.size())
    classified[scores <= model.cutoff] = 1

    #default = torch.zeros(labels.size())
    #default[scores <= 0.5] = 1

    tpr = classified[labels==1].mean() * 100
    fpr = classified[labels==0].mean() * 100
    
    tp = classified[labels==1].sum()
    fp = classified[labels==0].sum()
    
    f1 = get_f1(classified, labels)

    auc,ap = get_score(scores[labels==0], scores[labels==1])

    print("Learned Cutoff %0.4f" % model.cutoff)
    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))

    print("Top anom scored %0.04f" % anoms[0].item())
    print("Lowest anom scored %0.4f" % anoms[-1].item())
    print("Mean anomaly score: %0.4f" % anoms.mean().item())

    return {
        'TPR':tpr.item(), 
        'FPR':fpr.item(), 
        'TP':tp.item(), 
        'FP':fp.item(), 
        'F1':f1.item(), 
        'AUC':auc, 
        'AP': ap,
        'FwdTime':ctime
    }

'''
Requires using LANL. This was just for testing, so leaving it as is, 
but will error if trained on non-LANL data
'''
def test_single_embed(zs, rrefs, times, cutoff):
    # Load train data into workers
    ld_args = get_work_units(
        len(rrefs), 
        times['te_start'], 
        times['te_end'],
        times['delta'], 
        True
    )

    jobs = [
        _remote_method_async(
            DTGAE_Encoder.run_arbitrary_fn,
            rrefs[i],
            ld.load_edge_list_dist, W_THREADS*2, ld_args[i]['start'], ld_args[i]['end']
        )
        for i in range(len(rrefs))
    ]

    probs = []
    guesses = []
    labels = []
    for i in range(len(jobs)):
        ei, y = jobs[i].wait()

        # Decode the list of edges using just the last zs 
        # generated by the model
        prob = _remote_method(
            DTGAE_Encoder.decode,
            rrefs[i],
            ei, zs
        )

        guess = torch.zeros(prob.size())
        guess[prob <= cutoff] = 1

        probs.append(prob)
        guesses.append(guess)
        labels.append(y)

    # Using the same var names as in the test() method
    # so I can do a little ctrl+c, ctrl+v action here
    scores = torch.cat(probs)
    classified = torch.cat(guesses)
    labels = torch.cat(labels)
    anoms = scores[labels==1].sort()[0]

    # Copied from test()
    tpr = classified[labels==1].mean() * 100
    fpr = classified[labels==0].mean() * 100
    
    tp = classified[labels==1].sum()
    fp = classified[labels==0].sum()
    
    f1 = get_f1(classified, labels)

    auc,ap = get_score(scores[labels==0], scores[labels==1])

    print("Learned Cutoff %0.4f" % cutoff)
    print("TPR: %0.2f, FPR: %0.2f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))

    print("Top anom scored %0.04f" % anoms[0].item())
    print("Lowest anom scored %0.4f" % anoms[-1].item())
    print("Mean anomaly score: %0.4f" % anoms.mean().item())

    return {
        'TPR':tpr.item(), 
        'FPR':fpr.item(), 
        'TP':tp.item(), 
        'FP':fp.item(), 
        'F1':f1.item(), 
        'AUC':auc, 
        'AP': ap,
        'FwdTime': 0
    }


def speed_test(model, rrefs, times):
    Encoder = StaticEncoder if isinstance(model, StaticRecurrent) \
        else DynamicEncoder

    # Load train data into workers
    ld_args = get_work_units(
        len(rrefs), 
        times['tr_start'], 
        times['te_end'],
        times['delta'], 
        False
    )

    print("Loading test data")
    futs = [
        _remote_method_async(
            Encoder.load_new_data,
            rrefs[i], 
            LOAD_FN, 
            ld_args[i]
        ) for i in range(len(rrefs))
    ]

    # Wait until all workers have finished
    [f.wait() for f in futs]

    print("Running model")
    with torch.no_grad():
        model.eval()
        s = time.time()
        zs = model.forward(TData.ALL, no_grad=True)
        elapsed = time.time() - s    
    
    timesteps = zs.size(0)
    return {
        'workers': len(rrefs),
        'time': elapsed,
        'timesteps': timesteps
    }


def run_all(workers, rnn_constructor, rnn_args, worker_constructor, 
            worker_args, delta, just_test, fw, static, single_emb,
            run_speed_test, load_fn, tr_start, tr_end, manual, te_end=None):
    '''
    Starts up proceses, trains validates and tests the model given 
    the inputs 

        workers: how many worker processes to use
        rnn_constructor: constructor for StaticRNN model
        rnn_args: list of arguments for static rnn model
        worker_constructor: constructs StaticDDP wrapped RRef to worker
        worker_args: non-file loading related worker arguments
        delta=None : size of time window in hours to partition graphs
        te_end=None: how much test data to load. By default stops after first 
                    100 anomalous edges are loaded
    '''

    # Convert to hours
    delta = int(60**2 * delta)

    #tr_start=0
    #tr_end=ld.DATE_OF_EVIL_LANL
    
    # Need at least 2 deltas; default to 5% of tr data if that's enough
    val = max((tr_end - tr_start) // 20, delta*2)
    val_start = tr_end-val
    val_end = tr_end
    
    tr_end = val_start

    if not te_end:
        #te_end = 228642 # First 20 anoms
        #te_end = 740104 # First 100 anoms
        #te_end = 1089597 # First 500 anoms
        te_end = 5011199  # Full
    
    else:
        te_end = min(te_end * delta, 5011199) if run_speed_test \
            else te_end

    if not run_speed_test:
        max_workers = int((tr_end-tr_start) // delta)
        workers = max(min(max_workers, workers), 1)

    # Let timesteps overlap for dynamic as it never
    # runs loss on the last time step during training
    te_start = val_end if static \
        else val_end - delta

    times = {
        'tr_start': tr_start,
        'tr_end': tr_end,
        'val_end': val_end,
        'te_start': te_start,
        'te_end': te_end,
        'delta': delta
    }

    print(times)

    # Start workers
    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(
            world_size, 
            rnn_constructor, 
            rnn_args, 
            worker_constructor, 
            worker_args,
            times,
            just_test,
            fw,
            static,
            single_emb,
            run_speed_test,
            load_fn,
            manual
        ),
        nprocs=world_size,
        join=True
    )

    # Retrieve stats, and cleanup temp file
    stats = pickle.load(open(TMP_FILE, 'rb'))
    os.remove(TMP_FILE)

    print(stats)
    return stats

if __name__ == '__main__':
    run_all(WORKERS, GRU, RNN_ARGS, static_gcn_rref, WORKER_ARGS, 2.0, False, 0.6, True, False)