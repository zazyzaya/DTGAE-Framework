import os 
import pickle 
from joblib import Parallel, delayed

import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

from .load_utils import edge_tv_split

DATE_OF_EVIL_LANL = 150885
FILE_DELTA = 10000
LANL_FOLDER = '/mnt/raid0_24TB/isaiah/code/TGCN/src/data/split_LANL/'
RED_LOG = '/mnt/raid0_24TB/datasets/LANL_2015/data_files/redteam.txt'

COMP = 0
USR = 1
SPEC = 2

'''
 Build balanced edge weights between [0,1]
 Track how much of an outlier an edge is 
 AKA how many std's it is away from the mean (which is
 likely something like 10) then sigmoid squish it

 The effect is normal edges have weight of abt. 0.5, and 
 outlier edges (those which happen a lot) are closer to 1 
'''
def std_edge_w(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = torch.tensor(ew_t, dtype=torch.float)
        ew_t = (ew_t.long() / ew_t.std()).long()
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews

def normalized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = torch.tensor(ew_t, dtype=torch.float)
        ew_t = ew_t.true_divide(ew_t.mean())
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews

def load_lanl_dist(workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w):
    if workers <= 1:
        return load_partial_lanl(start, end, delta, is_test, ew_fn)

    num_slices = ((end - start) // delta) + 1 
    per_worker = [num_slices // workers] * workers 
    remainder = num_slices % workers 

    # Give everyone a balanced number of tasks 
    # put remainders on last machines as last task 
    # is probably smaller than a full delta
    if remainder:
        for i in range(workers, workers-remainder, -1):
            per_worker[i-1] += 1

    kwargs = []
    prev = start 
    for i in range(workers):
        end_t = prev + delta*per_worker[i]
        kwargs.append({
            'start': prev, 
            'end': min(end_t-1, end),
            'delta': delta,
            'is_test': is_test,
            'ew_fn': ew_fn
        })
        prev = end_t
    
    # Now start the jobs in parallel 
    datas = Parallel(n_jobs=workers, prefer='processes')(
        delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers)
    )

    # Helper method to concatonate one field from all of the datas
    data_reduce = lambda x : sum([datas[i].__getattribute__(x) for i in range(workers)], [])

    # Just join all the lists from all the data objects
    print("Joining Data objects")
    x = datas[0].x
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    te_starts = max([datas[i].te_starts for i in range(workers)])
    num_nodes = datas[0].num_nodes
    T = len(eis)
    slices = data_reduce('slices')
    ews = data_reduce('ews')
    node_map = datas[0].node_map

    if is_test:
        ys = data_reduce('ys')
    else:
        ys = None

    assert len(eis) == sum(per_worker), \
        "Something went wrong. Too many, or too few edge lists in reduced object"

    # After everything is combined, wrap it in a fancy new object, and you're
    # on your way to coolsville flats
    print("Done")
    return LANL_Data(
        x=x, eis=eis, masks=masks, te_starts=te_starts, 
        num_nodes=num_nodes, T=T, slices=slices, ys=ys,
        ews=ews, node_map=node_map
    )

    

# wrapper bc its annoying to send kwargs with Parallel
def load_partial_lanl_job(pid, args):
    #print("%d: Building %d - %d" % (pid, args['start'], args['end']))
    data = load_partial_lanl(**args)
    data.serialize()
    return data


def mapper(**kwargs):
    kwargs['ew_fn'] = lambda x : x 
    kwargs['delta'] = 1e9 # Arbitrarilly large number 
    return load_partial_lanl(**kwargs)

'''
Takes several data objects that have one timeslice each 
and combine them into a bigger graph that is also one timeslice
'''
def reducer(datas, ew_fn=std_edge_w):
    e_dict = {}

    def add_edge(et, y, weight):
        if et in e_dict:
            val = e_dict[et]
            e_dict[et] = (max(y, val[0]), val[1]+weight)
        else:
            e_dict[et] = (y, weight)

    '''
    In theory, could add another nested loop to account for 
    data obj.s with multiple time slices, but the map fn should
    make such an occurence difficult
    '''
    for d in datas:
        ei = d.eis[0]
        ys = d.ys[0]
        ew = d.ews[0]

        for i in range(ei.size(1)):
            src, dst = ei[:, i]
            add_edge(
                (src.item(), dst.item()),
                ys[i], ew[i]
            )

    ei = list(zip(*e_dict.keys()))
    y,ew = list(zip(*e_dict.values()))
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))
                
    return make_data_obj(
        [ei], -1, ew_fn, 
        slices=None,
        ys=[y],
        ews=[ew],
        node_map=node_map
    )


def make_data_obj(eis, tr_set_partition_end, ew_fn, **kwargs):
    # Known value for LANL
    cl_cnt = 17684

    # Use computer/user/special as features on top of nid
    feats = torch.zeros(cl_cnt+1, 3)

    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    for i in range(len(nm)):
        if nm[i][0] == 'C':
            feats[i][COMP] = 1
        elif nm[i][0] == 'U':
            feats[i][USR] = 1
        else:
            feats[i][SPEC] = 1

    x = feats
    
    # Build time-partitioned edge lists
    eis_t = []
    splits = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)
        
        # Add val mask for all time slices in training set
        if i < tr_set_partition_end:
            splits.append(edge_tv_split(ei)[0])

    # Balance the edge weights if they exist
    if 'ews' in kwargs:
        kwargs['ews'] = ew_fn(kwargs['ews'])

    # Finally, return Data object
    data = LANL_Data(
        x=x, 
        eis=eis_t,
        masks=splits,
        te_starts=tr_set_partition_end,
        num_nodes=cl_cnt,
        T=len(eis),
        **kwargs
    )
    return data

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files 
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, ew_fn=std_edge_w):
    cur_slice = start - (start % FILE_DELTA)
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    times = []

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    # Helper functions (trims the trailing \n)
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2][:-1]))
    def get_next_anom(rf):
        line = rf.readline().split(',')
        
        # Check we haven't reached EOF
        if len(line) > 1:
            return (int(line[0]), line[2], line[3])
        else:
            return float('inf'), float('inf'), float('inf')

    # For now, just keeps one copy of each edge. Could be
    # modified in the future to add edge weight or something
    # but for now, edges map to their anomaly value (1 == anom, else 0)
    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1)
        else:
            edges_t[et] = (is_anom, 1)

    def is_anomalous(src, dst, anom):
        src = node_map[src]
        dst = node_map[dst]
        return src==anom[1] and dst==anom[2][:-1]

    # If we're testing for anomalous edges, get the first anom that
    # will appear in this range (usually just the first one, but allows
    # for checking late time steps as well)
    if is_test:
        rf = open(RED_LOG, 'r')
        rf.readline() # Skip header
        
        next_anom = get_next_anom(rf)
        while next_anom[0] < start:
            next_anom = get_next_anom(rf)
    else:
        next_anom = (-1, 0,0)


    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_starts = 0
    anom_marked = False
    keep_reading = True

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime 
    while keep_reading:
        while line:
            l = line.split(',')
            
            # Scan to the correct part of the file
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts 
                curtime = ts 
                continue
            
            ts, src, dst = fmt_line(l)
            et = (src,dst)

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit 
            # (assumes no missing timesteps in the log files)
            if (curtime != ts and (curtime-ts) % delta == 0) or ts >=end:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    y,ew = list(zip(*edges_t.values()))
                    ews.append(ew)

                    if is_test:
                        ys.append(y)

                    edges_t = {}
                    times.append(str(curtime) + '-' + str(ts-1))

                # If the list was empty, just keep going if you can
                curtime = ts 

                # Break out of loop after saving if hit final timestep
                if ts >= end:
                    keep_reading = False 
                    break 

            # Skip self-loops
            if et[0] == et[1]:
                line = in_f.readline()
                continue

            # Mark edge as anomalous if it is 
            if ts == next_anom[0] and is_anomalous(src, dst, next_anom):
                add_edge(et, is_anom=1)
                next_anom = get_next_anom(rf)

                # Mark the first timestep with anomalies as test set start
                if not anom_marked:
                    anom_marked = True
                    anom_starts = len(edges)

            else:
                add_edge(et)

            line = in_f.readline()

        in_f.close() 
        cur_slice += FILE_DELTA 

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            break
    
    in_f.close() 
    
    if is_test:
        rf.close() 

    if not is_test:
        anom_starts = len(edges)

    ys = ys if is_test else None
    
    tot_anoms = 0
    if is_test:
        for y in ys:
            tot_anoms += sum(y)

    scan_prog.close()
    prog.close()

    return make_data_obj(
        edges, 
        anom_starts, 
        ew_fn=ew_fn, 
        slices=times,
        ys=ys,
        tot_anoms=tot_anoms,
        ews=ews,
        node_map=node_map
    )

if __name__ == '__main__':
    data = load_partial_lanl(end=DATE_OF_EVIL_LANL-1)
    print(data)