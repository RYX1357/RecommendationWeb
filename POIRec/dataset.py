"""Load the POIRec dataset and build model inputs.

The meta-path instance graphs are precomputed in preprocessing. Here we only
bin each meta-path's edges by their weight (path count) into `kmax` subgraphs
for the multi-level aggregation of the meta-path enhanced view (S332).
"""

import os
from collections import defaultdict

import torch


def _bin_subgraphs(mp_edge, mp_weight, kmax):
    """Split a weighted meta-path graph into subgraphs by edge-weight bin.

    Subgraph k holds edges whose path count equals k (counts >= kmax are merged
    into the last bin). Returns {k: edge_index (2, E_k)}.
    """
    bins = defaultdict(list)
    for t in range(mp_edge.shape[1]):
        w = min(int(mp_weight[t]), kmax)
        bins[w].append(t)
    sub = {}
    for k in range(1, kmax + 1):
        idx = bins.get(k, [])
        if idx:
            sub[k] = torch.tensor(mp_edge[:, idx], dtype=torch.long)
        else:
            sub[k] = torch.empty((2, 0), dtype=torch.long)
    return sub


def load_poi(city, kmax=5):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city, 'poi_data.pt')
    raw = torch.load(path)

    nu = raw['num_user']
    nl = raw['num_loc']

    mp_edge = {}
    mp_sub = {}
    for p, (e, w) in raw['mp'].items():
        e_t = torch.tensor(e, dtype=torch.long)
        mp_edge[p] = e_t
        mp_sub[p] = _bin_subgraphs(e, w, kmax)

    data = {
        'num_user': nu,
        'num_loc': nl,
        'user_feat': torch.tensor(raw['user_feat']),
        'loc_feat': torch.tensor(raw['loc_feat']),
        'loc_cat': torch.tensor(raw['loc_cat'], dtype=torch.long),
        'loc_total': torch.tensor(raw['loc_total'], dtype=torch.long),
        'loc_name': raw['loc_name'],
        'poi_lat': raw['poi_lat'],
        'poi_lng': raw['poi_lng'],
        'net_edge_index': torch.tensor(raw['net_edge_index'], dtype=torch.long),
        'mp_edge': mp_edge,
        'mp_sub': mp_sub,
    }

    user_visited = raw['user_visited']
    test_checkins = torch.tensor(raw['test_checkins'], dtype=torch.long)
    return data, user_visited, test_checkins
