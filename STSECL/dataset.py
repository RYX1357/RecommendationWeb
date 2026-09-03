"""Load the STSECL dataset, split friend edges, and build model inputs.

The friend edges are the positive labels: 90% are kept for training and used to
build the friend-dependent structure (U-U edges, F_UUU meta-path, friendship
hyperedges); the remaining 10% are held out for evaluation.
"""

import os

import numpy as np
import torch

from metapath import twohop, undirected


def _build_hyper_incidence(hyper_friend, hyper_checkin, hyper_traj):
    node_list, edge_list, hyper_type = [], [], []

    n_f = len(hyper_friend)
    if n_f > 0:
        node_list.append(hyper_friend.reshape(-1))
        edge_list.append(np.repeat(np.arange(n_f), hyper_friend.shape[1]))
        hyper_type.append(np.zeros(n_f * hyper_friend.shape[1], dtype=np.int64))

    n_c = len(hyper_checkin)
    base_c = n_f
    if n_c > 0:
        node_list.append(hyper_checkin.reshape(-1))
        edge_list.append(base_c + np.repeat(np.arange(n_c), hyper_checkin.shape[1]))
        hyper_type.append(np.ones(n_c * hyper_checkin.shape[1], dtype=np.int64))

    base_t = base_c + n_c
    tid = 0
    for traj in hyper_traj:
        if len(traj) == 0:
            continue
        node_list.append(np.asarray(traj, dtype=np.int64))
        edge_list.append(np.full(len(traj), base_t + tid, dtype=np.int64))
        hyper_type.append(np.full(len(traj), 2, dtype=np.int64))
        tid += 1

    node_list = np.concatenate(node_list).astype(np.int64)
    edge_list = np.concatenate(edge_list).astype(np.int64)
    hyper_type = np.concatenate(hyper_type).astype(np.int64)
    return (torch.tensor(node_list), torch.tensor(edge_list), torch.tensor(hyper_type))


def load_stsecl(city, split=0.9, seed=30100):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city, 'stsecl_data.pt')
    raw = torch.load(path)

    nu = raw['num_user']
    nl = raw['num_loc']
    nc = raw['num_cat']
    nt = raw['num_time']

    # ---- friend split ----
    uu = raw['uu_edge']                                  # (2, E) undirected (single dir)
    uu_feat = raw['uu_feat']                             # (E, 5)
    n_friend = uu.shape[1]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_friend)
    n_train = int(n_friend * split)
    train_uu = uu[:, perm[:n_train]]
    test_uu = uu[:, perm[n_train:]]
    train_uu_feat = uu_feat[perm[:n_train]]

    # all friend pairs (for negative sampling exclusion)
    all_friends_set = {(min(int(uu[0, i]), int(uu[1, i])),
                        max(int(uu[0, i]), int(uu[1, i]))) for i in range(n_friend)}

    # ---- F_UUU rebuilt from TRAIN friends only (concatenated edge features) ----
    uu_tr, uu_tr_f = undirected(train_uu, train_uu_feat)
    F_UUU_edge, F_UUU_feat = twohop(uu_tr, uu_tr_f, uu_tr, uu_tr_f, nu, nu, K=20)

    # ---- meta-path edge_index (already directed) ----
    mp_edge_index, mp_edge_feat = {}, {}
    for mp, (e, f) in raw['mp'].items():
        mp_edge_index[mp] = torch.tensor(e, dtype=torch.long)
        mp_edge_feat[mp] = torch.tensor(f, dtype=torch.float32)
    mp_edge_index['F_UUU'] = torch.tensor(F_UUU_edge, dtype=torch.long)
    mp_edge_feat['F_UUU'] = torch.tensor(F_UUU_feat, dtype=torch.float32)

    # ---- hypergraph incidence (friend hyperedges use train friends) ----
    hyper_node, hyper_edge, hyper_type = _build_hyper_incidence(
        train_uu.T, raw['hyper_checkin'], raw['hyper_traj'])

    node_type = np.concatenate([
        np.zeros(nu, dtype=np.int64),
        np.ones(nl, dtype=np.int64),
        np.full(nc, 2, dtype=np.int64),
        np.full(nt, 3, dtype=np.int64),
    ])

    data = {
        'num_user': nu,
        'num_loc': nl,
        'num_cat': nc,
        'num_time': nt,
        'user_feat': torch.tensor(raw['user_feat']),
        'loc_feat': torch.tensor(raw['loc_feat']),
        'cat_feat': torch.tensor(raw['cat_feat']),
        'time_feat': torch.tensor(raw['time_feat']),
        'mp_edge_index': mp_edge_index,
        'mp_edge_feat': mp_edge_feat,
        'hyper_node': hyper_node,
        'hyper_edge': hyper_edge,
        'hyper_type': hyper_type,
        'node_type': torch.tensor(node_type),
    }

    train_friends = torch.tensor(train_uu, dtype=torch.long)
    test_friends = torch.tensor(test_uu, dtype=torch.long)
    return data, train_friends, test_friends, all_friends_set
