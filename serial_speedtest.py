from time import time 

import torch
import pandas as pd

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
    
    print("Started... ", end='')
    start = time() 
    model.forward(data, data.ALL)
    elapsed = time() - start 
    print("Finished %0.4f" % elapsed)

    return {'delta': delta, 'size': size, 'time': elapsed}


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
        [EGCN_H, EGCN_O],
        [32,64,128,256,512,1024],
        5
    )
