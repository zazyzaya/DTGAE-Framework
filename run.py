from argparse import ArgumentParser

import pandas as pd

from models.recurrent import GRU, LSTM, Lin, EmptyModel
from models.embedders import \
    static_gcn_rref, static_gat_rref, static_sage_rref, static_gin_rref,\
    dynamic_gcn_rref, dynamic_gat_rref, dynamic_sage_rref, dynamic_gin_rref 

from spinup import run_all

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
        choices=['GCN', 'GAT', 'SAGE', 'GIN'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM', 'NONE', 'MLP'],
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

    ap.add_argument(
        '--single',
        action='store_true'
    )

    ap.add_argument(
        '--pred', '-p',
        action='store_true'
    )

    ap.add_argument(
        '--speedtest',
        type=int,
        default=-1
    )

    args = ap.parse_args()
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    readable = str(args)
    print(readable)

    static = not args.pred

    # Convert from str to function pointer
    if args.encoder == 'GCN':
        args.encoder = static_gcn_rref if static \
            else dynamic_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = static_gat_rref if static \
            else dynamic_gat_rref
    elif args.encoder == 'SAGE':
        args.encoder = static_sage_rref if static \
            else dynamic_sage_rref
    else:
        args.encoder = static_gin_rref if static \
            else dynamic_gin_rref

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM 
    elif args.rnn == 'MLP':
        args.rnn = Lin
    else:
        args.rnn = EmptyModel

    return args, readable

if __name__ == '__main__':
    args, argstr = get_args() 

    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        # Need to tell workers to output in embed dim
        worker_args = [args.hidden, args.zdim]
        rnn_args = [args.hidden, args.hidden, args.zdim]

    st_d = args.speedtest
    st = True if st_d > 0 else False
    st_d = st_d if st else None

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
            not args.pred,
            args.single,
            st,
            te_end=st_d
        )
        for _ in range(args.tests)
    ]

    # Don't write out if nowrite
    if args.nowrite:
        exit() 

    df = pd.DataFrame(stats)
    compressed = pd.DataFrame(
        [df.mean(), df.sem()],
        index=['mean', 'stderr']
    ).to_csv().replace(',', ', ')

    full = df.to_csv(index=False, header=False)
    full = full.replace(',', ', ')

    with open(HOME+'results/stats.txt', 'a') as f:
        if st:
            f.write(str(stats))
            f.write('\n')

        else:
            f.write(str(argstr) + '\n\n')
            f.write(str(compressed) + '\n')
            f.write(full + '\n\n')
        
