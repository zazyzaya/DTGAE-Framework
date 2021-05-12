from argparse import ArgumentParser

import pandas as pd

from models.recurrent import GRU, LSTM 
from models.dist_static_modules import static_gcn_rref, static_gat_rref, static_sage_rref
from spinup_static import run_all

HOME = '/mnt/raid0_24TB/isaiah/code/DTGAE/'
def get_args():
    ap = ArgumentParser()

    ap.add_argument(
        '-d', '--delta',
        type=float, default=2.0
    )

    ap.add_argument(
        '-w', '--workers',
        type=int, default=4
    )

    ap.add_argument(
        '-T', '--threads',
        type=int, default=2
    )

    ap.add_argument(
        '-e', '--encoder',
        choices=['GCN', 'GAT', 'SAGE'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM'],
        type=str.upper,
        default="GRU"
    )

    ap.add_argument(
        '-H', '--hidden',
        type=int,
        default=32
    )

    ap.add_argument(
        '-z', '--zdim',
        type=int,
        default=16
    )

    ap.add_argument(
        '-n', '--ngrus',
        type=int,
        default=1
    )

    ap.add_argument(
        '-t', '--tests',
        type=int, 
        default=1
    )

    ap.add_argument(
        '-l', '--load',
        action='store_true'
    )

    ap.add_argument(
        '--fpweight',
        type=float,
        default=0.6
    )

    ap.add_argument(
        '--nowrite',
        action='store_true'
    )


    args = ap.parse_args()
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    readable = str(args)
    print(readable)

    # Convert from str to function pointer
    if args.encoder == 'GCN':
        args.encoder = static_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = static_gat_rref
    else:
        args.encoder = static_sage_rref

    if args.rnn == 'GRU':
        args.rnn = GRU
    else:
        args.rnn = LSTM 

    return args, readable

if __name__ == '__main__':
    args, argstr = get_args() 

    worker_args = [args.hidden, args.hidden]
    rnn_args = [args.hidden, args.hidden, args.zdim]

    stats = [
        run_all(
            args.workers, 
            args.rnn, 
            rnn_args,
            args.encoder, 
            worker_args, 
            args.delta,
            args.load,
            args.fpweight,
        )
        for _ in range(args.tests)
    ]

    df = pd.DataFrame(stats)
    compressed = pd.DataFrame(
        [df.mean(), df.sem()],
        index=['mean', 'stderr']
    ).to_csv().replace(',', ', ')

    full = df.to_csv(index=False, header=False)
    full = full.replace(',', ', ')

    # Don't write out if nowrite
    if args.nowrite:
        exit() 

    with open(HOME+'results/stats.txt', 'a') as f:
        f.write(str(argstr) + '\n\n')
        f.write(str(compressed) + '\n')
        f.write(full + '\n\n')
        
