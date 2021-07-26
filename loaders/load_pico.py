import json
import os 
from tqdm import tqdm

import torch 
from dateutil import parser as ps

from .load_utils import edge_tv_split
from .tdata import TData

# Last timestamp before malware in system
DATE_OF_EVIL_PICO = ps.parse("2019-07-19T20:41:36.000000Z").timestamp()

PICO_START = 1563494703.935567
PICO_END = ps.parse('2019-07-21T16:10:00.000000Z').timestamp()

def pico_file_loader(fname, keep=['client', 'ts', 'service']):
    # Not really in the right schema to just use json.loads on the 
    # whole thing. Each line is its own json object
    with open(fname, 'r') as f:
        lines = f.read().split('\n')
        logs = [json.loads(l) for l in lines if len(l) > 1]

    # Filter out noisey logs. Only care about TGS kerb logs (for now)
    unflogs = logs #[l for l in logs if 'request_type' in l.keys()]
    
    # Get rid of extranious data, and make sure required data exists
    logs = []
    for l in unflogs:
        try:
            logs.append({k:l[k] for k in keep})
        except KeyError as e:
            continue 

    return logs 


'''
Given several JSON objects representing single hour of auth data,
generate temporal graphs with delta-hour long partitions
'''
def pico_logs_to_graph(logs, delta, start, end, whitelist=[]):
    '''
    (Useful) Kerb logs have the following structure:
                                   (where they access from)
    client:   USR (or) COMPUTER$ / INSTANCE . subdomain . subdomain (etc)

             (optional)    (the computer)                     (optional)
    service:  service   /  TOP LEVEL .    sub domain . (etc)  @  realm  

    Worth noting, service names for computers are in all caps, client names are.. varied.
    To be safe, capitalize everything
    '''
    cl_to_id = {}  # Map of string names to numeric 0-|N|
    cl_cnt = 0     # Cur id

    eis = []
    ews = []
    ei = {}

    ts_human_readable = []

    # Kind of cheating because DNS changes these, but if possible
    # when an admin account does something, it's helpful to know
    # what computer it was from for comparison to the Red Log 
    ip_map = {
        '27': 'RND-WIN10-2',
        '29': 'RND-WIN10-1',
        '30': 'HR-WIN7-2',
        '152': 'HR-WIN7-1',
        '160': 'SUPERSECRETXP',
        '5': 'CORP-DC',
        '100': 'PFSENSE'
    }

    
    halt_recording = False 
    window_start = ps.parse(logs[0]['ts']).timestamp()
    hr = logs[0]['ts']

    for i in tqdm(range(len(logs))):
        l = logs[i]
        
        ts = ps.parse(l['ts']).timestamp()
        if ts < start:
            window_start = ts
            continue
        
        # First parse out client 
        client = l['client'].split('/')[0] # Don't really care about instance 
        client = client.split('$')[0]
        client = client.upper()
        
        skip = False
        for wl in whitelist:
            if wl in l['service'].upper():
                skip = True
        if skip:
            continue
    
        if 'ADMIN' in client:
            client = client.replace('ADMINISTRATOR', 'ADMIN') # Shorten a bit

        if client in cl_to_id:
            client = cl_to_id[client]
        else:
            cl_to_id[client] = cl_cnt
            client = cl_cnt
            cl_cnt += 1
        
        # Then parse out server & service 
        srv = l['service'].split('/')
        if len(srv) > 1:
            srv = srv[1]
        else:
            srv = srv[0]

        server = srv.split('.')[0].upper()
        server = server.split('$')[0]

        # Add in id of server 
        if server in cl_to_id:
            server = cl_to_id[server]
        else:
            cl_to_id[server] = cl_cnt
            server = cl_cnt 
            cl_cnt += 1

        # Ignore self loops
        if client != server:
            # Add to edge list 
            if (client, server) in ei:
                ei[(client, server)] += 1
            else:
                ei[(client, server)] = 1

        # Create new partition for next set of edges when DELTA time
        # has been parsed, or when end of file is reached
        if not halt_recording and (ts >= window_start+delta or ts >= end):
            while window_start < ts:
                window_start += delta 

            if len(ei.keys()) > 0:
                eis.append(
                    list(zip(*list(ei.keys())))
                )
                ews.append(list(ei.values()))
                ei = {}

                ts_human_readable.append(hr)

            hr = l['ts']
        
        # Dataset so small, its worth it to just read in the whole thing
        # to get the number of nodes
        if ts >= end:
            halt_recording = True

    print([len(ei[0]) for ei in eis])
    return make_data_obj(eis, ews, cl_to_id, hr=ts_human_readable)


def make_data_obj(eis, ews, cl_to_id, **kwargs):
    cl_cnt = max(cl_to_id.values())
    
    node_map = [None] * (max(cl_to_id.values()) + 1)
    for k,v in cl_to_id.items():
        node_map[v] = k

    # No node feats really
    x = torch.eye(cl_cnt+1).float()
    
    # Build time-partitioned edge lists
    eis_t = []
    ews_t = []
    splits = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)    
        
        # Standardize edge weights
        ew = torch.tensor(ews[i]).float()
        ew = torch.sigmoid((ew - ew.mean()) / ew.std())
        ews_t.append(ew)

        splits.append(edge_tv_split(ei, v_size=0.25)[0])

    # Finally, return Data object
    data = TData(eis_t, x, None, splits, ews=ews_t, nmap=node_map, **kwargs)
    return data

def load_pico(workers, start=0, end=1000, delta=60*60*2, is_test=False):
    F_LOC = '/mnt/raid0_24TB/datasets/pico/bro/'
    days = [os.path.join(F_LOC,d) for d in os.listdir(F_LOC)]
    days.sort()

    logs = []
    for d in days:
        kerb_logs = [os.path.join(d, l) for l in os.listdir(d) if 'kerb' in l]
        kerb_logs.sort()

        list_o_logs = [pico_file_loader(l) for l in kerb_logs]
        for l in list_o_logs:
            logs += l

    return pico_logs_to_graph(logs, delta, start, end)

if __name__ == '__main__':
    load_pico(8, **{'start': 0, 'end': float('inf'), 'delta': 7200, 'is_test': False})