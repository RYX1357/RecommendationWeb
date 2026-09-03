"""Losses and evaluation for POIRec."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

MP_NAMES = ['LUL', 'LLL', 'LUUL']


def info_nce(a, b, tau):
    """InfoNCE between two views; positive = same node (diagonal)."""
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    n = a.shape[0]
    sim = a @ b.T / tau
    eye = torch.eye(n, device=a.device, dtype=torch.bool)
    pos = sim.diag()
    denom = torch.logsumexp(sim.masked_fill(eye, float('-inf')), dim=1)
    return (-pos + denom).mean()


def total_loss(out, tau, beta, lam, rho1, rho2):
    """S4: L = rho1 * L_MP + rho2 * L_G-mp + (1 - rho1 - rho2) * L_G-ns."""
    H, Ytilde = out['H'], out['Ytilde']

    L_MP = 0.0
    for p in MP_NAMES:
        L_MP = L_MP + beta * info_nce(H[p], Ytilde[p], tau) \
                    + (1 - beta) * info_nce(Ytilde[p], H[p], tau)

    L_G_mp = lam * info_nce(out['me'], out['mp'], tau) \
             + (1 - lam) * info_nce(out['mp'], out['me'], tau)
    L_G_ns = lam * info_nce(out['me'], out['net'], tau) \
             + (1 - lam) * info_nce(out['net'], out['me'], tau)

    return rho1 * L_MP + rho2 * L_G_mp + (1 - rho1 - rho2) * L_G_ns


def evaluate(y_me, user_visited, test_checkins, nu, nl, device, k_list=(5, 10, 20)):
    """Recall@k / NDCG@k for POI recommendation.

    User preference = mean-pool of visited-POI embeddings; cosine ranking over
    all candidate POIs (visited ones masked to -inf).
    """
    y = F.normalize(y_me, dim=1)
    d = y.shape[1]

    pref = torch.zeros(nu, d, device=device, dtype=y.dtype)
    for u in range(nu):
        if user_visited[u]:
            idx = torch.tensor(user_visited[u], device=device)
            pref[u] = y[idx].mean(0)
    pref = F.normalize(pref, dim=1)

    scores = pref @ y.T                                    # (nu, nl)

    rows, cols = [], []
    for u in range(nu):
        for p in user_visited[u]:
            rows.append(u)
            cols.append(p)
    if rows:
        scores[torch.tensor(rows, device=device),
               torch.tensor(cols, device=device)] = -1e9

    test_by_user = defaultdict(list)
    for u, p, dd, c in test_checkins.tolist():
        test_by_user[u].append(p)

    kmax = max(k_list)
    recall, ndcg = {k: [] for k in k_list}, {k: [] for k in k_list}
    for u, ts in test_by_user.items():
        if not ts:
            continue
        ts_set = set(ts)
        top = torch.topk(scores[u], kmax)[1].tolist()
        for k in k_list:
            hit = sum(1 for t in top[:k] if t in ts_set)
            recall[k].append(hit / len(ts))
            dcg = sum(1.0 / np.log2(i + 2) for i, t in enumerate(top[:k]) if t in ts_set)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ts))))
            ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    recall = {k: float(np.mean(v)) if v else 0.0 for k, v in recall.items()}
    ndcg = {k: float(np.mean(v)) if v else 0.0 for k, v in ndcg.items()}
    return recall, ndcg


def recommend(y_me, user_visited, loc_cat, loc_total, loc_name, poi_lat, poi_lng,
              nu, nl, device, k=None):
    """POI recommendations per user, with POI metadata.

    User preference = mean-pool of visited-POI embeddings (enhanced view), then
    cosine ranking over all candidate POIs (visited ones masked to -inf).

    ``k`` caps the number of candidates per user; when None (or larger than nl)
    all unvisited POIs are emitted. Yields
    (user, rank, poi, category, name, lat, lng, visits, score) tuples.
    """
    y = F.normalize(y_me, dim=1)
    d = y.shape[1]

    pref = torch.zeros(nu, d, device=device, dtype=y.dtype)
    for u in range(nu):
        if user_visited[u]:
            idx = torch.tensor(user_visited[u], device=device)
            pref[u] = y[idx].mean(0)
    pref = F.normalize(pref, dim=1)

    scores = pref @ y.T                                    # (nu, nl)

    rows, cols = [], []
    for u in range(nu):
        for p in user_visited[u]:
            rows.append(u)
            cols.append(p)
    if rows:
        scores[torch.tensor(rows, device=device),
               torch.tensor(cols, device=device)] = -1e9

    if k is None:
        k = nl
    k = min(k, nl)
    topk_val, topk_idx = torch.topk(scores, k, dim=1)
    topk_val = topk_val.cpu().numpy()
    topk_idx = topk_idx.cpu().numpy()

    cat = loc_cat.cpu().numpy() if hasattr(loc_cat, 'cpu') else np.asarray(loc_cat)
    tot = loc_total.cpu().numpy() if hasattr(loc_total, 'cpu') else np.asarray(loc_total)
    lat = poi_lat if isinstance(poi_lat, np.ndarray) else np.asarray(poi_lat)
    lng = poi_lng if isinstance(poi_lng, np.ndarray) else np.asarray(poi_lng)

    for u in range(nu):
        rank = 0
        for r in range(k):
            s = float(topk_val[u, r])
            if s <= -1e8:                                # visited POI
                continue
            rank += 1
            p = int(topk_idx[u, r])
            yield (u, rank, p, int(cat[p]), loc_name[p],
                   float(lat[p]), float(lng[p]), int(tot[p]), s)
