from time import time 

import torch
from torch.optim import Adam
import pandas as pd

from loaders.tdata import TData 
from loaders.load_lanl import load_lanl_dist as ld 
from models.serial_models import \
    VGRNN, \
    SparseEGCN_H as EGCN_H, \
    SparseEGCN_O as EGCN_O

torch.set_num_threads(16)

def load_and_eval(constructor, delta, size, data=None):
    '''
    Assumes delta is in seconds, and size is number of deltas to load,
    alternatively can provide a data object 
    '''
    if data is None: 
        data = ld(8, start=0, end=delta*size, delta=delta)

    model = constructor(data.x_dim, 32, 16, pred=False)
    opt = Adam(model.parameters(), lr=0.01)

    print("Started... ", end='')
    start = time() 
    zs = model.forward(data, data.ALL)
    elapsed = time() - start 
    print("Finished %0.4f" % elapsed)

    print("Started loss...", end='')
    start = time() 
    p = [data.ei_masked(TData.TRAIN, i) for i in range(data.T)]
    n = data.get_negative_edges(TData.TRAIN, nratio=10)
    loss = model.calc_loss(p,n,zs)
    ltime = time() - start
    print("Finished %0.4f" % ltime)

    print("Started backward... ")
    start = time() 
    loss.backward()
    opt.step()
    btime = time() - start
    print("Finished %0.4f" % btime)
    

    return {'delta': delta, 'size': size, 'time': elapsed, 'loss_time': ltime, 'back_time': btime}


OUT_F = 'results/speedtest_vgrnn.txt'
def test(models, sizes, tests, d=0.5):
    for m in models: 
        for s in sizes:
            delta = d * (60**2) # Input is in human readable hour incriments
            data = ld(8, start=0, end=delta*s, delta=delta)

            results = [load_and_eval(m, delta, s, data=data) for _ in range(tests)]
            results = pd.DataFrame(results)
            results = results.mean()
            print(str(results))

            with open(OUT_F, 'a+') as f:
                f.write(str(m)+'\n')
                f.write('%0.1f\t%d\n' % (d, s))
                f.write(str(results))
                f.write('\n\n')

if __name__ == '__main__': 
    test(
        [VGRNN, EGCN_H, EGCN_O],
        [32,64,128,256,512,1024],
        5
    )
