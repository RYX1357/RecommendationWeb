"""POIRec data preprocessing (Gowalla CSV -> poi_data.pt).

Builds the LBSN (users + POIs), node initial features (6-dim), four
spatio-temporal enhanced edges from real geographic distance thresholds
(delta1..delta4) and three POI-anchored meta-path instance graphs
(LUL / LLL / LUUL), then saves everything to `data/{city}/poi_data.pt`.

Node initial features (patent):
  * user = [4 time-slot check-in counts, activity-center (lat, lng)]
  * POI  = [4 time-slot visit counts, POI (lat, lng)]
  time slots: night(22-5) / morning(5-10) / noon(10-15) / evening(15-22)

Enhanced edges (geographic distance thresholds):
  * U-L       (delta1) : user activity-center to POI distance < delta1 km
  * U-U       (delta2) : user activity-center to user activity-center < delta2 km
  * L-L type  (delta3) : same-category POIs within delta3 km
  * L-L visit (delta4) : co-visited POIs within delta4 km
"""

import ast
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from metapath import adj_matrix, bipartite, mat_to_weighted_edges

DIM_NODE = 6          # 4 time-slot counts + 2 coordinates
SEED = 30100

CITY_FILES = {
    'BER': dict(checkin='check_in_berlin_user_in_friend.csv',
                friend='friend_ship_berlin.csv',
                poi='berlin_poi_incheckin_and_friend.csv'),
    'CHI': dict(checkin='check_in_chi_user_in_friend.csv',
                friend='friend_ship_chi.csv',
                poi='chi_poi_incheckin_and_friend.csv'),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _time_slot(hour):
    if hour >= 22 or hour < 5:
        return 0   # night
    if hour < 10:
        return 1   # morning
    if hour < 15:
        return 2   # noon
    return 3       # evening


def _hour_of(dt_str):
    return int(dt_str[11:13])


def _category_name(spot_categories):
    return ast.literal_eval(spot_categories)[0]['name']


def _project(lat, lng):
    """(lat, lng) -> (x, y) in km (equirectangular) for cKDTree neighbour search."""
    lat = np.asarray(lat, dtype=np.float64)
    lng = np.asarray(lng, dtype=np.float64)
    x = lng * 111.32 * np.cos(np.deg2rad(lat))
    y = lat * 110.574
    return x, y


def _haversine(lat1, lng1, lat2, lng2):
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [float(lat1), float(lng1),
                                              float(lat2), float(lng2)])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _sym(edges):
    """Symmetrise a single-direction (2, E) edge tensor into both directions."""
    if edges.shape[1] == 0:
        return edges.astype(np.int64)
    return np.concatenate([edges, edges[[1, 0]]], axis=1).astype(np.int64)


def _cap_degree(edges, K):
    """Keep at most K edges incident to each node (undirected), preserving order.

    The geographic thresholds already bound each node's neighbourhood, but the
    larger delta2 radius can still be dense; this cap guarantees sparsity.
    """
    if edges.shape[1] == 0:
        return edges
    deg = defaultdict(int)
    keep = []
    for t in range(edges.shape[1]):
        a, b = int(edges[0, t]), int(edges[1, t])
        if deg[a] < K and deg[b] < K:
            keep.append(t)
            deg[a] += 1
            deg[b] += 1
    if not keep:
        return np.empty((2, 0), dtype=np.int64)
    return edges[:, keep]


# ----------------------------------------------------------------------------
# raw loader
# ----------------------------------------------------------------------------
def load_csv(city):
    names = CITY_FILES[city]
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city)

    ci = pd.read_csv(os.path.join(data_dir, names['checkin']))
    fr = pd.read_csv(os.path.join(data_dir, names['friend']))
    poi = pd.read_csv(os.path.join(data_dir, names['poi']))

    user_ids = sorted(ci['userid'].unique())
    poi_ids = sorted(poi['id'].unique())
    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    poi2idx = {int(p): i for i, p in enumerate(poi_ids)}

    nu, nl = len(user_ids), len(poi_ids)

    # POI metadata (aligned to poi_ids order)
    lat_map = dict(zip(poi['id'].astype(int), poi['lat'].astype(float)))
    lng_map = dict(zip(poi['id'].astype(int), poi['lng'].astype(float)))
    cat_map = dict(zip(poi['id'].astype(int), poi['spot_categories'].apply(_category_name)))
    cat_set = sorted({cat_map[p] for p in poi_ids})
    cat2idx = {c: i for i, c in enumerate(cat_set)}
    nc = len(cat_set)

    poi_lat = np.array([lat_map[p] for p in poi_ids], dtype=np.float64)
    poi_lng = np.array([lng_map[p] for p in poi_ids], dtype=np.float64)
    loc_cat = np.array([cat2idx[cat_map[p]] for p in poi_ids], dtype=np.int64)

    # check-ins -> (user, poi, hour)
    ci_u = ci['userid'].astype(int).map(user2idx).astype(np.int64).to_numpy()
    ci_p = ci['placeid'].astype(int).map(poi2idx).astype(np.int64).to_numpy()
    ci_h = ci['datetime'].apply(_hour_of).astype(np.int64).to_numpy()
    checkins = np.stack([ci_u, ci_p, ci_h], axis=1)

    # friendships -> (E, 2) user indices (both endpoints must be check-in users)
    fr_u1 = fr['userid1'].astype(int).map(user2idx)
    fr_u2 = fr['userid2'].astype(int).map(user2idx)
    ok = fr_u1.notna() & fr_u2.notna()
    friends = np.stack([fr_u1[ok].astype(np.int64).to_numpy(),
                        fr_u2[ok].astype(np.int64).to_numpy()], axis=1)

    return {
        'num_user': nu, 'num_loc': nl, 'num_cat': nc,
        'loc_cat': loc_cat, 'cat_names': cat_set,
        'poi_lat': poi_lat, 'poi_lng': poi_lng,
        'checkins': checkins, 'friends': friends,
    }


# ----------------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------------
def build_dataset(city, save=True, delta1=1.0, delta2=15.0, delta3=1.0, delta4=1.0,
                  split=0.9, topk=20, enh_topk=20):
    print(f'[{city}] loading CSV ...', flush=True)
    raw = load_csv(city)
    nu, nl = raw['num_user'], raw['num_loc']
    nc = raw['num_cat']
    loc_cat = raw['loc_cat']
    cat_names = raw['cat_names']
    poi_lat, poi_lng = raw['poi_lat'], raw['poi_lng']
    checkins = raw['checkins']
    friends = raw['friends']
    print(f'  users={nu} locs={nl} cats={nc} '
          f'checkins={checkins.shape[0]} friends={friends.shape[0]}', flush=True)

    # ---- split unique (user, poi) pairs into train / test ----
    pairs = sorted({(int(u), int(p)) for u, p, h in checkins})
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(pairs))
    n_train = int(len(pairs) * split)
    train_pairs = {pairs[i] for i in perm[:n_train]}
    train_mask = np.array([(int(u), int(p)) in train_pairs
                           for u, p, h in checkins], dtype=bool)
    train_ci = checkins[train_mask]
    test_ci = checkins[~train_mask]

    # ---- per-node visit statistics (train only) ----
    user_slot = np.zeros((nu, 4), dtype=np.float32)
    loc_slot = np.zeros((nl, 4), dtype=np.float32)
    visit_cnt = defaultdict(int)
    user_locs = defaultdict(set)
    for u, p, h in train_ci:
        u, p = int(u), int(p)
        s = _time_slot(int(h))
        user_slot[u, s] += 1
        loc_slot[p, s] += 1
        visit_cnt[(u, p)] += 1
        user_locs[u].add(p)

    loc_total = loc_slot.sum(1).astype(np.int64)

    # ---- node initial features (6-dim) ----
    # user activity center = mean (lat, lng) of visited POIs
    center_lat = np.zeros(nu, dtype=np.float64)
    center_lng = np.zeros(nu, dtype=np.float64)
    for u in range(nu):
        if user_locs[u]:
            ps = list(user_locs[u])
            center_lat[u] = poi_lat[ps].mean()
            center_lng[u] = poi_lng[ps].mean()

    lat_lo, lat_hi = poi_lat.min(), poi_lat.max()
    lng_lo, lng_hi = poi_lng.min(), poi_lng.max()

    user_feat = np.zeros((nu, DIM_NODE), dtype=np.float32)
    loc_feat = np.zeros((nl, DIM_NODE), dtype=np.float32)
    user_feat[:, :4] = user_slot
    user_feat[:, 4] = (center_lat - lat_lo) / (lat_hi - lat_lo)
    user_feat[:, 5] = (center_lng - lng_lo) / (lng_hi - lng_lo)
    loc_feat[:, :4] = loc_slot
    loc_feat[:, 4] = (poi_lat - lat_lo) / (lat_hi - lat_lo)
    loc_feat[:, 5] = (poi_lng - lng_lo) / (lng_hi - lng_lo)

    # ---- original edges ----
    ul_edge = np.array(sorted(visit_cnt.keys()), dtype=np.int64).T   # (2, E) U -> L
    uu_edge = friends.T.astype(np.int64)                             # (2, E) U-U single dir
    print(f'    U-L={ul_edge.shape[1]}  U-U={uu_edge.shape[1]}', flush=True)

    # ---- enhanced edges (geographic) ----
    poi_x, poi_y = _project(poi_lat, poi_lng)
    usr_x, usr_y = _project(center_lat, center_lng)
    poi_xy = np.stack([poi_x, poi_y], axis=1)
    usr_xy = np.stack([usr_x, usr_y], axis=1)
    tree_p = cKDTree(poi_xy)
    tree_u = cKDTree(usr_xy)

    # (1) U-L: user center -> unvisited POIs within delta1
    ul_enh = []
    for u in range(nu):
        visited = user_locs[u]
        for p in tree_p.query_ball_point(usr_xy[u], delta1):
            if p not in visited:
                ul_enh.append((u, p))
    ul_enh_edge = np.array(ul_enh, dtype=np.int64).T if ul_enh else np.empty((2, 0), dtype=np.int64)
    ul_enh_edge = _cap_degree(ul_enh_edge, enh_topk)

    # (2) U-U: user centers within delta2
    uu_enh = []
    for u in range(nu):
        for v in tree_u.query_ball_point(usr_xy[u], delta2):
            if v > u:
                uu_enh.append((u, v))
    uu_enh_edge = np.array(uu_enh, dtype=np.int64).T if uu_enh else np.empty((2, 0), dtype=np.int64)
    uu_enh_edge = _cap_degree(uu_enh_edge, enh_topk)

    # (3) L-L type: same-category POIs within delta3
    ll_type = []
    for p in range(nl):
        for q in tree_p.query_ball_point(poi_xy[p], delta3):
            if q > p and loc_cat[p] == loc_cat[q]:
                ll_type.append((p, q))
    ll_type_edge = np.array(ll_type, dtype=np.int64).T if ll_type else np.empty((2, 0), dtype=np.int64)
    ll_type_edge = _cap_degree(ll_type_edge, enh_topk)

    # (4) L-L visit: co-visited POIs within delta4
    A_ul = bipartite(ul_edge, nu, nl)
    co_visit = A_ul.T @ A_ul
    co_edges, _ = mat_to_weighted_edges(co_visit, topk=topk)
    ll_visit = []
    for t in range(co_edges.shape[1]):
        p, q = int(co_edges[0, t]), int(co_edges[1, t])
        if p != q and _haversine(poi_lat[p], poi_lng[p], poi_lat[q], poi_lng[q]) < delta4:
            ll_visit.append((p, q))
    ll_visit_edge = np.array(ll_visit, dtype=np.int64).T if ll_visit else np.empty((2, 0), dtype=np.int64)

    print(f'    U-L_enh={ul_enh_edge.shape[1]}  U-U_enh={uu_enh_edge.shape[1]}  '
          f'L-L_type={ll_type_edge.shape[1]}  L-L_visit={ll_visit_edge.shape[1]}', flush=True)

    # ---- meta-path instance graphs (LUL / LLL / LUUL), POI-POI with weights ----
    A_uu = adj_matrix(_sym(uu_edge), nu)                          # undirected friendship
    ll_type_bin = adj_matrix(_sym(ll_type_edge), nl)
    ll_visit_bin = adj_matrix(ll_visit_edge, nl)                  # already directed
    A_ll = (ll_type_bin + ll_visit_bin).tocsr()
    A_ll.data = np.ones_like(A_ll.data)                           # binarize combined L-L

    mp = {
        'LUL': mat_to_weighted_edges(A_ul.T @ A_ul, topk=topk),
        'LLL': mat_to_weighted_edges(A_ll @ A_ll, topk=topk),
        'LUUL': mat_to_weighted_edges(A_ul.T @ A_uu @ A_ul, topk=topk),
    }
    for k, (e, w) in mp.items():
        print(f'    {k}: edges={e.shape[1]}', flush=True)

    # ---- combined net adjacency (nodes: user 0..nu-1, loc nu..nu+nl-1) ----
    net_parts = []
    uu_all = _sym(np.concatenate([uu_edge, uu_enh_edge], axis=1) if uu_enh_edge.shape[1] else uu_edge)
    net_parts.append(uu_all)
    ul_all = np.concatenate([ul_edge, ul_enh_edge], axis=1)
    if ul_all.shape[1]:
        ul_all = _sym(np.stack([ul_all[0], nu + ul_all[1]], 0))
        net_parts.append(ul_all)
    if ll_type_edge.shape[1]:
        net_parts.append(_sym(np.stack([nu + ll_type_edge[0], nu + ll_type_edge[1]], 0)))
    if ll_visit_edge.shape[1]:
        net_parts.append(np.stack([nu + ll_visit_edge[0], nu + ll_visit_edge[1]], 0))
    net_edge_index = np.concatenate(net_parts, axis=1).astype(np.int64)
    print(f'    net_edges={net_edge_index.shape[1]}', flush=True)

    # ---- user -> visited POIs (train) for recommendation ----
    user_visited = [sorted(user_locs.get(u, ())) for u in range(nu)]

    # test check-ins as (user, poi, hour, cat)
    test_ci_4 = np.stack([test_ci[:, 0], test_ci[:, 1], test_ci[:, 2],
                          loc_cat[test_ci[:, 1].astype(np.int64)]], axis=1).astype(np.int64)

    data = {
        'city': city,
        'delta1': delta1, 'delta2': delta2, 'delta3': delta3, 'delta4': delta4,
        'split': split,
        'num_user': nu, 'num_loc': nl, 'num_cat': nc,
        'user_feat': user_feat, 'loc_feat': loc_feat,
        'loc_cat': loc_cat, 'loc_total': loc_total,
        'loc_name': [cat_names[int(loc_cat[p])] for p in range(nl)],
        'poi_lat': poi_lat, 'poi_lng': poi_lng,
        'net_edge_index': net_edge_index,
        'mp': {k: (e, w) for k, (e, w) in mp.items()},
        'user_visited': user_visited,
        'test_checkins': test_ci_4,
    }

    if save:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'poi_data.pt')
        torch.save(data, path)
        with open(os.path.join(out_dir, 'build_params.json'), 'w') as f:
            json.dump({'delta1': delta1, 'delta2': delta2,
                       'delta3': delta3, 'delta4': delta4}, f)
        print(f'  saved -> {path}', flush=True)

    return data


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser('POIRec preprocess',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('cities', type=str, nargs='*', default=['BER'],
                   help='city names (BER,CHI)')
    p.add_argument('--delta1', type=float, default=1.0, help='U-L enhanced edge radius (km)')
    p.add_argument('--delta2', type=float, default=15.0, help='U-U enhanced edge radius (km)')
    p.add_argument('--delta3', type=float, default=1.0, help='L-L type enhanced edge radius (km)')
    p.add_argument('--delta4', type=float, default=1.0, help='L-L visit enhanced edge radius (km)')
    p.add_argument('--split', type=float, default=0.9)
    p.add_argument('--enh-topk', type=int, default=20,
                   help='max incident enhanced edges per node (degree cap)')
    a = p.parse_args()
    for c in a.cities:
        build_dataset(c, save=True, delta1=a.delta1, delta2=a.delta2,
                      delta3=a.delta3, delta4=a.delta4, split=a.split,
                      enh_topk=a.enh_topk)
