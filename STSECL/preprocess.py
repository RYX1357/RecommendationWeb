"""
STSECL data preprocessing (project-local only).

Reads the HHGNN-style .pkl files under `data/{city}/` and builds a uniform
STSECL dataset: node features, base edges, spatio-temporal edge augmentation,
three meta-path relationship graphs (Friend/Check-in/Proximity) and three
hyperedge types (friendship/check-in/trajectory).

The .pkl files store only index-level structure
  * friend_list_index.pkl      : {friend-edge-id : {user_i, user_j}}
  * trajectory_list_index.pkl  : {user-id : {visited-poi-global-ids}}
  * visit_list_edge_tensor.pkl : {checkin-id : {user, poi, class, day}}
(global index order: user -> poi -> class -> day). They do NOT contain
latitude/longitude or hour-of-day, so the geographic / fine-grained temporal
signals used in the paper are approximated as follows:

  * location-location (L-L) edges : same category (proxy for "nearby" + same type)
  * proximity U'-U' edges         : >= rho shared check-in POIs (proxy for co-location)
  * proximity U'-L' edges         : category affinity (user visited that category)
  * "4 time intervals" frequencies: replaced by total count in the first slot
  * distance features             : replaced by co-visitation count / 0.0

Meta-path relationship graphs are materialised with sparse matrix products
(path counts), which keeps the enumeration tractable for dense augmentation.

The model architecture (views + attention + InfoNCE contrast) is unchanged.
"""

import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from metapath import adj_matrix, bipartite, mat_to_edges, twohop, undirected

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
DIM_NODE = 64          # random node feature dim (paper)
NUM_INTERVAL = 4       # 5-10, 10-15, 15-22, 22-5 (hour-of-day, unavailable -> total in slot 0)

# node counts per city for decoding the global indices (user -> poi -> class -> day)
CITY_STATS = {
    'BER': dict(user=3545, poi=5874, cls=229, time=642),
    'CHI': dict(user=6444, poi=5807, cls=273, time=770),
    'NYC': dict(user=3754, poi=3626, cls=281, time=547),
    'JK':  dict(user=6184, poi=8805, cls=314, time=566),
    'KL':  dict(user=6324, poi=10804, cls=337, time=573),
    'SP':  dict(user=3811, poi=6255, cls=289, time=549),
}


# ----------------------------------------------------------------------------
# raw loader  (project-local .pkl only)
# ----------------------------------------------------------------------------
def load_pkl(city):
    st = CITY_STATS[city]
    nu, nl, nc, nt = st['user'], st['poi'], st['cls'], st['time']
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city)

    with open(os.path.join(data_dir, 'friend_list_index.pkl'), 'rb') as f:
        friend = pickle.load(f)
    with open(os.path.join(data_dir, 'visit_list_edge_tensor.pkl'), 'rb') as f:
        visit = pickle.load(f)
    with open(os.path.join(data_dir, 'trajectory_list_index.pkl'), 'rb') as f:
        traj = pickle.load(f)

    base_loc = nu
    base_cls = nu + nl
    base_time = nu + nl + nc

    # friends: {edge-id : {u, v}} -> (E, 2)
    friends = []
    for eid in range(len(friend)):
        a, b = sorted(friend[eid])
        friends.append((int(a), int(b)))
    friends = np.array(friends, dtype=np.int64).reshape(-1, 2)

    # check-ins: {checkin-id : {user, poi, class, day}} -> (N, 4) = (user, loc, day, cat)
    checkins = []
    for cid in range(len(visit)):
        s = sorted(visit[cid])
        checkins.append((s[0], s[1] - base_loc, s[3] - base_time, s[2] - base_cls))
    checkins = np.array(checkins, dtype=np.int64)

    # location category: a location's class index (from any of its check-ins)
    loc_cat = -np.ones(nl, dtype=np.int64)
    for cid in range(len(visit)):
        s = sorted(visit[cid])
        p = s[1] - base_loc
        if loc_cat[p] < 0:
            loc_cat[p] = s[2] - base_cls
    loc_cat[loc_cat < 0] = 0

    # trajectory: {user-id : {poi-global-ids}} -> list of (local) poi sets
    trajectories = []
    for u in range(nu):
        pts = sorted(traj.get(u, []))
        trajectories.append([p - base_loc for p in pts])

    return {
        'num_user': nu, 'num_loc': nl, 'num_cat': nc, 'num_day': nt,
        'loc_cat': loc_cat,
        'checkins': checkins, 'friends': friends, 'trajectories': trajectories,
    }


# ----------------------------------------------------------------------------
# base statistics
# ----------------------------------------------------------------------------
def _visit_stats(raw):
    """Per-(user,loc) visit count, distinct-day count, and user -> visited-loc sets."""
    checkins = raw['checkins']
    visit_cnt = defaultdict(int)
    visit_days = defaultdict(set)
    user_locs = defaultdict(set)
    for u, p, d, c in checkins:
        visit_cnt[(u, p)] += 1
        visit_days[(u, p)].add(int(d))
        user_locs[u].add(int(p))
    return visit_cnt, visit_days, user_locs


# ----------------------------------------------------------------------------
# feature builders
# ----------------------------------------------------------------------------
def _build_uu(raw, user_locs):
    """Friend edges U-U, feat (5,) = [shared-POI count, 0,0,0, 0]."""
    edges, feats = [], []
    for u, v in raw['friends']:
        u, v = int(u), int(v)
        shared = len(user_locs[u] & user_locs[v])
        edges.append((u, v))
        feats.append([float(shared), 0.0, 0.0, 0.0, 0.0])
    return np.array(edges, dtype=np.int64).T, np.array(feats, dtype=np.float32)


def _build_ul(raw, visit_cnt, visit_days):
    """Check-in edges U-L, feat (4,) = [visit count, distinct days, 0, 0]."""
    edges = sorted(visit_cnt.keys())
    feats = np.array([[float(visit_cnt[e]), float(len(visit_days[e])), 0.0, 0.0]
                      for e in edges], dtype=np.float32)
    return np.array(edges, dtype=np.int64).T, feats


def _build_ll(raw):
    """L-L edges (same category), feat (2,) = [1.0, 0.0].

    Same-category is the proxy for the paper's "same category AND distance <
    delta1" constraint, since lat/lon is not available.
    """
    nl = raw['num_loc']
    loc_cat = raw['loc_cat']
    by_cat = defaultdict(list)
    for a in range(nl):
        by_cat[loc_cat[a]].append(a)
    edges, feats = [], []
    for plist in by_cat.values():
        for a in range(len(plist)):
            for b in range(a + 1, len(plist)):
                x, y = plist[a], plist[b]
                edges.append((x, y)); edges.append((y, x))
                feats.append([1.0, 0.0]); feats.append([1.0, 0.0])
    return np.array(edges, dtype=np.int64).T, np.array(feats, dtype=np.float32)


def _build_proximity(raw, user_locs, rho=2):
    """U'-U' (>= rho shared POIs) and U'-L' (category affinity) proximity edges.

    feat (1,) = shared-POI count (U'-U') or 1.0 (U'-L').
    """
    nu, nl = raw['num_user'], raw['num_loc']
    loc_cat = raw['loc_cat']
    checkins = raw['checkins']

    # location -> users who visited it
    loc_users = defaultdict(set)
    for u, p, d, c in checkins:
        loc_users[p].add(u)

    # user-user shared-POI counts
    pair_cnt = defaultdict(int)
    for p, users in loc_users.items():
        ul = sorted(users)
        for a in range(len(ul)):
            for b in range(a + 1, len(ul)):
                pair_cnt[(ul[a], ul[b])] += 1

    uup, uup_f = [], []
    for (x, y), cnt in pair_cnt.items():
        if cnt < rho:
            continue
        uup.append((x, y)); uup.append((y, x))
        uup_f.append([float(cnt)]); uup_f.append([float(cnt)])

    # user -> set of visited categories
    user_cats = defaultdict(set)
    for u in user_locs:
        for p in user_locs[u]:
            user_cats[u].add(loc_cat[p])

    # U'-L': category affinity (same category as a visited location, not visited)
    ulp, ulp_f = [], []
    for u in range(nu):
        visited = user_locs[u]
        for c in user_cats[u]:
            for p in range(nl):
                if loc_cat[p] == c and p not in visited:
                    ulp.append((u, p))
                    ulp_f.append([1.0])

    return (np.array(uup, dtype=np.int64).T, np.array(uup_f, dtype=np.float32),
            np.array(ulp, dtype=np.int64).T, np.array(ulp_f, dtype=np.float32))


# ----------------------------------------------------------------------------
# meta-path materialization  (concatenated edge features, Eq. 3)
# ----------------------------------------------------------------------------
def _build_meta_paths(raw, uu_edge, uu_feat, ul_edge, ul_feat, ll_edge, ll_feat,
                      uup_edge, uup_feat, ulp_edge, ulp_feat):
    """Materialise the six meta-path relationship graphs.

    Two-hop meta-paths (F_UUU, C_ULU) use the paper's concatenated base-edge
    features averaged over the middle node (Eq. 3). The remaining paths fall
    back to log1p path counts with correctly padded feature dims (proximity
    graphs are too dense for exact enumeration, and U'-L' features are constant).
    """
    nu = raw['num_user']
    nl = raw['num_loc']

    uu_u, uu_f = undirected(uu_edge, uu_feat)               # friendship is undirected
    ul_rev = np.stack([ul_edge[1], ul_edge[0]], 0)          # L -> U (reverse check-in)

    out = {}
    out['F_UUU'] = twohop(uu_u, uu_f, uu_u, uu_f, nu, nu, K=20)            # U-U-U
    out['C_ULU'] = twohop(ul_edge, ul_feat, ul_rev, ul_feat, nu, nl, K=20) # U-L-U

    A_ul = bipartite(ul_edge, nu, nl)
    A_ll = adj_matrix(ll_edge, nl)
    A_uup = adj_matrix(uup_edge, nu)
    A_ulp = bipartite(ulp_edge, nu, nl)
    out['P_UUU'] = mat_to_edges(A_uup @ A_uup, 2, K=20)           # U'-U'-U'
    out['P_ULU'] = mat_to_edges(A_ulp @ A_ulp.T, 2, K=20)         # U'-L'-U'
    out['C_ULLU'] = mat_to_edges(A_ul @ A_ll @ A_ul.T, 10, K=20)  # U-L-L-U
    out['P_ULLU'] = mat_to_edges(A_ulp @ A_ll @ A_ulp.T, 4, K=20)  # U'-L'-L'-U'
    return out


# ----------------------------------------------------------------------------
# hyperedge builders
# ----------------------------------------------------------------------------
def _build_hyperedges(raw):
    nu, nl = raw['num_user'], raw['num_loc']
    nc, nt = raw['num_cat'], raw['num_day']
    base_loc = nu
    base_cls = nu + nl
    base_time = nu + nl + nc
    checkins = raw['checkins']

    hyper_friend = np.array([[int(u), int(v)] for u, v in raw['friends']], dtype=np.int64)

    hyper_checkin = []
    for u, p, d, c in checkins:
        cat = raw['loc_cat'][p]
        hyper_checkin.append([int(u), base_loc + int(p), base_time + int(d), base_cls + int(cat)])
    hyper_checkin = np.array(hyper_checkin, dtype=np.int64)

    hyper_traj = []
    for u in range(nu):
        hyper_traj.append(sorted(base_loc + p for p in raw['trajectories'][u]))

    return hyper_friend, hyper_checkin, hyper_traj


# ----------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------
def build_dataset(city, save=True, delta1=1.0, delta2=1.0, rho=2):
    print(f'[{city}] loading .pkl ...', flush=True)
    raw = load_pkl(city)
    nu, nl = raw['num_user'], raw['num_loc']
    nc, nt = raw['num_cat'], raw['num_day']
    print(f'  users={nu} locs={nl} cats={nc} days={nt} '
          f'checkins={raw["checkins"].shape[0]} friends={raw["friends"].shape[0]}', flush=True)

    rng = np.random.RandomState(30100)
    user_feat = rng.randint(0, 10, (nu, DIM_NODE)).astype(np.float32)
    loc_feat = rng.randint(0, 10, (nl, DIM_NODE)).astype(np.float32)
    cat_feat = np.eye(nc, dtype=np.float32)
    time_feat = np.eye(nt, dtype=np.float32)

    print('  building base edges ...', flush=True)
    visit_cnt, visit_days, user_locs = _visit_stats(raw)
    uu_edge, uu_feat = _build_uu(raw, user_locs)
    ul_edge, ul_feat = _build_ul(raw, visit_cnt, visit_days)
    ll_edge, ll_feat = _build_ll(raw)
    print(f'    U-U={uu_edge.shape[1]} (feat {uu_feat.shape[1]})  '
          f'U-L={ul_edge.shape[1]} (feat {ul_feat.shape[1]})  L-L={ll_edge.shape[1]}', flush=True)

    print(f'  building proximity edges (delta1={delta1}, delta2={delta2}, rho={rho}) ...', flush=True)
    uup_edge, uup_feat, ulp_edge, ulp_feat = _build_proximity(raw, user_locs, rho)
    print(f'    U\'-U\'={uup_edge.shape[1]}  U\'-L\'={ulp_edge.shape[1]}', flush=True)

    print('  building meta-path relationship graphs ...', flush=True)
    mp = _build_meta_paths(raw, uu_edge, uu_feat, ul_edge, ul_feat, ll_edge, ll_feat,
                           uup_edge, uup_feat, ulp_edge, ulp_feat)
    for k, (e, f) in mp.items():
        print(f'    {k}: edges={e.shape[1]} feat={f.shape[1]}', flush=True)

    print('  building hyperedges ...', flush=True)
    hyper_friend, hyper_checkin, hyper_traj = _build_hyperedges(raw)
    print(f'    friend={len(hyper_friend)} checkin={len(hyper_checkin)} traj={len(hyper_traj)}', flush=True)

    data = {
        'city': city,
        'delta1': delta1, 'delta2': delta2, 'rho': rho,
        'num_user': nu, 'num_loc': nl, 'num_cat': nc, 'num_time': nt,
        'user_feat': user_feat, 'loc_feat': loc_feat,
        'cat_feat': cat_feat, 'time_feat': time_feat,
        'uu_edge': uu_edge, 'uu_feat': uu_feat,
        'ul_edge': ul_edge, 'ul_feat': ul_feat,
        'll_edge': ll_edge, 'll_feat': ll_feat,
        'uup_edge': uup_edge, 'uup_feat': uup_feat,
        'ulp_edge': ulp_edge, 'ulp_feat': ulp_feat,
        'mp': mp,
        'hyper_friend': hyper_friend,
        'hyper_checkin': hyper_checkin,
        'hyper_traj': hyper_traj,
    }

    if save:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'stsecl_data.pt')
        torch.save(data, path)
        with open(os.path.join(out_dir, 'build_params.json'), 'w') as f:
            json.dump({'delta1': delta1, 'delta2': delta2, 'rho': rho}, f)
        print(f'  saved -> {path}', flush=True)

    return data


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser('STSECL preprocess',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('cities', type=str, nargs='*', default=['BER'],
                   help='city names (BER,CHI,NYC,JK,KL,SP)')
    p.add_argument('--delta1', type=float, default=1.0,
                   help='L-L distance threshold (km, paper; approx. via same category)')
    p.add_argument('--delta2', type=float, default=1.0,
                   help='proximity distance threshold (km, paper; approx. via shared POIs)')
    p.add_argument('--rho', type=int, default=2,
                   help='min shared check-in POIs for a U\'-U\' proximity edge')
    a = p.parse_args()
    for c in a.cities:
        build_dataset(c, save=True, delta1=a.delta1, delta2=a.delta2, rho=a.rho)
