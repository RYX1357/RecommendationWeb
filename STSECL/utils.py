"""Losses and evaluation for STSECL (no DGL dependency)."""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score


def info_nce(anchor, other, friend_index, tau):
    """InfoNCE with multiple positives (friends), mean over positives. Eq. (14)."""
    N = anchor.shape[0]
    anchor = F.normalize(anchor, dim=1)
    other = F.normalize(other, dim=1)
    mask = torch.zeros(N, N, device=anchor.device)
    mask[friend_index[0], friend_index[1]] = 1.0
    mask[friend_index[1], friend_index[0]] = 1.0

    sim = anchor @ other.T / tau                       # (N, N)
    eye = torch.eye(N, device=anchor.device, dtype=torch.bool)
    denom = torch.logsumexp(sim.masked_fill(eye, float('-inf')), dim=1)   # (N,)
    pos_count = mask.sum(1)                            # (N,)
    pos_mean = (sim * mask).sum(1) / pos_count.clamp(min=1)   # mean over positives

    has_friend = pos_count > 0
    loss = (-pos_mean + denom)[has_friend].mean()
    return loss


def contrastive_loss(z_em, z_hg, friend_index, tau, lam):
    l_eh = info_nce(z_em, z_hg, friend_index, tau)     # em as anchor
    l_he = info_nce(z_hg, z_em, friend_index, tau)     # hg as anchor
    return lam * l_eh + (1 - lam) * l_he


def _sample_scores(z, test_pos, all_friends_set, nu, device):
    """1:1 negative sampling and cosine scores for the test edges."""
    n_pos = test_pos.shape[1]
    g = torch.Generator().manual_seed(30100)
    neg_src = test_pos[0].cpu().numpy().copy()
    neg_dst = np.zeros(n_pos, dtype=np.int64)
    for i in range(n_pos):
        s = int(neg_src[i])
        d = int(torch.randint(0, nu, (1,), generator=g).item())
        while s == d or (min(s, d), max(s, d)) in all_friends_set:
            d = int(torch.randint(0, nu, (1,), generator=g).item())
        neg_dst[i] = d

    pos_score = (z[test_pos[0]] * z[test_pos[1]]).sum(-1)
    neg_dst_t = torch.tensor(neg_dst, device=device)
    neg_src_t = torch.tensor(neg_src, device=device)
    neg_score = (z[neg_src_t] * z[neg_dst_t]).sum(-1)

    labels = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    return labels, scores


def evaluate_auc(z_em, test_pos, all_friends_set, nu, device):
    """AUC only (no N x N matrix), for early-stopping monitoring."""
    z = F.normalize(z_em, dim=1)
    labels, scores = _sample_scores(z, test_pos, all_friends_set, nu, device)
    return roc_auc_score(labels, scores)


def evaluate(z_em, test_pos, all_friends_set, train_friends, nu, device,
             k_list=(1, 5, 10, 15, 20)):
    """AUC / AP / Top@k for friend recommendation using cosine similarity."""
    z = F.normalize(z_em, dim=1)

    labels, scores = _sample_scores(z, test_pos, all_friends_set, nu, device)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    # --- Top@k ---
    n_pos = test_pos.shape[1]
    N = z.shape[0]
    sim = z @ z.T                                    # (N, N)
    tmask = torch.zeros(N, N, device=device, dtype=torch.bool)
    if train_friends.shape[1] > 0:
        tmask[train_friends[0], train_friends[1]] = True
        tmask[train_friends[1], train_friends[0]] = True
    tmask.fill_diagonal_(True)
    sim = sim.masked_fill(tmask, -1.0)

    test_friends = defaultdict(set)
    for i in range(n_pos):
        u, v = int(test_pos[0][i]), int(test_pos[1][i])
        test_friends[u].add(v)
        test_friends[v].add(u)

    kmax = max(k_list)
    right = {k: 0 for k in k_list}
    total = 0
    for u, fs in test_friends.items():
        total += 1
        top = torch.topk(sim[u], kmax)[1].tolist()
        for k in k_list:
            if set(top[:k]) & fs:
                right[k] += 1
    top_k = [right[k] / max(total, 1) for k in k_list]
    return auc, ap, top_k


def recommend_friends(z_em, train_friends, nu, device, k=None):
    """Friend recommendations per user via cosine similarity.

    Masks self and already-known (training) friends, then yields every remaining
    candidate in similarity order. ``k`` caps the number of candidates per user;
    when None (or larger than nu) all candidates are emitted. Yields
    (user, rank, friend, score) tuples.
    """
    z = F.normalize(z_em, dim=1)
    sim = z @ z.T                                    # (nu, nu)
    mask = torch.zeros(nu, nu, device=device, dtype=torch.bool)
    mask.fill_diagonal_(True)
    if train_friends.shape[1] > 0:
        mask[train_friends[0], train_friends[1]] = True
        mask[train_friends[1], train_friends[0]] = True
    sim = sim.masked_fill(mask, float('-inf'))

    if k is None:
        k = nu
    k = min(k, nu)
    topk_val, topk_idx = torch.topk(sim, k, dim=1)
    topk_val = topk_val.cpu().numpy()
    topk_idx = topk_idx.cpu().numpy()

    for u in range(nu):
        rank = 0
        for r in range(k):
            s = float(topk_val[u, r])
            if s <= -1e30:                           # self / training friend
                continue
            rank += 1
            yield (u, rank, int(topk_idx[u, r]), s)
