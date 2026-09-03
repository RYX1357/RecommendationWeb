import argparse


def parse():
    p = argparse.ArgumentParser('STSECL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # runtime
    p.add_argument('city', type=str, nargs='?', default='BER',
                   help='dataset name (BER,CHI,NYC,JK,KL,SP)')
    p.add_argument('--gpu', type=str, default='0', help='gpu id')
    p.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--wd', type=float, default=0.0, help='weight decay')
    p.add_argument('--seed', type=int, default=30100, help='random seed')
    p.add_argument('--patience', type=int, default=5, help='early stopping patience (epochs)')

    # model
    p.add_argument('--dim', type=int, default=64, help='embedding dimension d')
    p.add_argument('--nhead', type=int, default=8, help='number of hypergraph attention heads K')
    p.add_argument('--tau', type=float, default=0.2, help='InfoNCE temperature')
    p.add_argument('--lam', type=float, default=0.5, help='balance of the two contrastive losses')
    p.add_argument('--activation', type=str, default='prelu', choices=['relu', 'prelu'],
                   help='activation in aggregation')

    # data
    p.add_argument('--delta1', type=float, default=1.0,
                   help='L-L distance threshold (km; approx. via same category)')
    p.add_argument('--delta2', type=float, default=1.0,
                   help='proximity distance threshold (km; approx. via shared POIs)')
    p.add_argument('--rho', type=int, default=2,
                   help='min shared check-in POIs for a proximity edge')
    p.add_argument('--split', type=float, default=0.9, help='train ratio of friend edges')
    p.add_argument('--topk', type=int, default=None,
                   help='max friends per user in the final output (default: all)')

    args = p.parse_args()
    return args
