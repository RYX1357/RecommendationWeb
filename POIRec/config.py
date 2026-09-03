import argparse


def parse():
    p = argparse.ArgumentParser('POIRec', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # runtime
    p.add_argument('city', type=str, nargs='?', default='BER',
                   help='dataset name (BER,CHI,NYC,JK,KL,SP)')
    p.add_argument('--gpu', type=str, default='0', help='gpu id')
    p.add_argument('--epochs', type=int, default=30,
                   help='max epochs (early-stopped by --patience)')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--wd', type=float, default=0.0, help='weight decay')
    p.add_argument('--seed', type=int, default=30100, help='random seed')
    p.add_argument('--patience', type=int, default=5,
                   help='early stopping patience (epochs without NDCG@10 gain)')

    # model
    p.add_argument('--dim', type=int, default=64, help='embedding dimension d')
    p.add_argument('--activation', type=str, default='prelu', choices=['relu', 'prelu'],
                   help='activation in aggregation')
    p.add_argument('--tau', type=float, default=0.2, help='InfoNCE temperature')
    p.add_argument('--beta', type=float, default=0.5,
                   help='balance of the two meta-path-level contrast components (S41)')
    p.add_argument('--lam', type=float, default=0.5,
                   help='balance of the two symmetric graph-level contrast components (S42)')
    p.add_argument('--rho1', type=float, default=0.4,
                   help='weight of meta-path-level loss L_MP in total loss')
    p.add_argument('--rho2', type=float, default=0.3,
                   help='weight of graph-level enhanced-vs-metapath loss L_G-mp in total loss')
    p.add_argument('--kmax', type=int, default=5,
                   help='number of edge-weight bins when splitting meta-path instance subgraphs (S332)')

    # data (patent hyperparameters; real geographic distance thresholds in km)
    p.add_argument('--delta1', type=float, default=1.0,
                   help='U-L enhanced edge radius: user activity-center to POI (km)')
    p.add_argument('--delta2', type=float, default=15.0,
                   help='U-U enhanced edge radius: user activity-center to user (km)')
    p.add_argument('--delta3', type=float, default=1.0,
                   help='L-L type enhanced edge radius: same-category POIs (km)')
    p.add_argument('--delta4', type=float, default=1.0,
                   help='L-L visit enhanced edge radius: co-visited POIs (km)')
    p.add_argument('--split', type=float, default=0.9,
                   help='train ratio of check-ins')
    p.add_argument('--topk', type=int, default=None,
                   help='max POIs per user in the final output (default: all)')

    args = p.parse_args()
    return args
