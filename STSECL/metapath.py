"""Shared meta-path materialisation helpers for STSECL (numpy/scipy).

Kept in one module so both `preprocess.py` (full graph) and `dataset.py`
(train-only F_UUU rebuild) produce identical meta-path features.
"""

from collections import defaultdict

import numpy as np
import scipy.sparse as sp


def adj_matrix(edges, n, dtype=np.float32):
    """Binary (n, n) CSR adjacency from a (2, E) edge tensor."""
    if edges.shape[1] == 0:
        return sp.csr_matrix((n, n), dtype=dtype)
    data = np.ones(edges.shape[1], dtype=dtype)
    return sp.csr_matrix((data, (edges[0], edges[1])), shape=(n, n), dtype=dtype)


def bipartite(edges, n_row, n_col, dtype=np.float32):
    """Binary (n_row, n_col) CSR from a (2, E) edge tensor."""
    if edges.shape[1] == 0:
        return sp.csr_matrix((n_row, n_col), dtype=dtype)
    data = np.ones(edges.shape[1], dtype=dtype)
    return sp.csr_matrix((data, (edges[0], edges[1])), shape=(n_row, n_col), dtype=dtype)


def to_directed(edges, feats):
    """Symmetrise a (2, E) edge tensor into both directions."""
    src, dst = edges[0], edges[1]
    e = np.concatenate([np.stack([src, dst], 1), np.stack([dst, src], 1)], 0).T
    f = np.concatenate([feats, feats], 0)
    return e, f


def undirected(edges, feats):
    """Make an undirected edge tensor directed (add reversed copies)."""
    return to_directed(edges, feats)


def mat_to_edges(M, feat_dim, K=20):
    """Top-K sparse a user-user path-count matrix into a directed graph.

    feat (feat_dim,) = [log1p(path count), 0, ...].
    """
    M = M.tolil()
    M.setdiag(0)
    M = M.tocsr()
    M.eliminate_zeros()

    kept = []
    for i in range(M.shape[0]):
        start, end = M.indptr[i], M.indptr[i + 1]
        if start == end:
            continue
        idx = M.indices[start:end]
        val = M.data[start:end]
        if len(idx) > K:
            order = np.argpartition(val, -K)[-K:]
            idx = idx[order]
            val = val[order]
        for j, v in zip(idx, val):
            kept.append((i, int(j), float(v)))

    d = {}
    for i, j, v in kept:
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        d[key] = max(d.get(key, 0.0), v)

    keys = sorted(d.keys())
    edges = np.array(keys, dtype=np.int64).T
    counts = np.array([d[k] for k in keys], dtype=np.float32)
    feats = np.zeros((len(keys), feat_dim), dtype=np.float32)
    feats[:, 0] = np.log1p(counts)
    return to_directed(edges, feats)


def twohop(edge_A, feat_A, edge_B, feat_B, n_src, n_mid, K=20):
    """U -> M -> U meta-path with concatenated edge features (mean over M).

    edge_A: (2, Ea) source -> middle ; feat_A: (Ea, da)
    edge_B: (2, Eb) middle -> target ; feat_B: (Eb, db)
    Returns a directed (edge_index (2, E), feats (E, da + db)) graph.
    """
    A_bin = bipartite(edge_A, n_src, n_mid)
    B_bin = bipartite(edge_B, n_mid, n_src)
    P = A_bin @ B_bin
    P = P.tolil()
    P.setdiag(0)
    P = P.tocsr()
    P.eliminate_zeros()

    # top-K candidate pairs by path count (symmetrised + deduped)
    cand = {}
    for i in range(n_src):
        start, end = P.indptr[i], P.indptr[i + 1]
        if start == end:
            continue
        idx = P.indices[start:end]
        val = P.data[start:end]
        if len(idx) > K:
            order = np.argpartition(val, -K)[-K:]
            idx = idx[order]
            val = val[order]
        for j, v in zip(idx, val):
            j = int(j)
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            cand[key] = max(cand.get(key, 0.0), float(v))

    # edge feature lookups
    fa = {(int(edge_A[0, t]), int(edge_A[1, t])): feat_A[t] for t in range(edge_A.shape[1])}
    fb = {(int(edge_B[0, t]), int(edge_B[1, t])): feat_B[t] for t in range(edge_B.shape[1])}

    # source -> middles, and target -> middles (reverse of B)
    src_mid = defaultdict(set)
    for t in range(edge_A.shape[1]):
        src_mid[int(edge_A[0, t])].add(int(edge_A[1, t]))
    tgt_mid = defaultdict(set)
    for t in range(edge_B.shape[1]):
        tgt_mid[int(edge_B[1, t])].add(int(edge_B[0, t]))

    da, db = feat_A.shape[1], feat_B.shape[1]
    edges_out, feats_out = [], []
    for (i, j) in cand:
        for u, v in ((i, j), (j, i)):
            mids = src_mid.get(u, set()) & tgt_mid.get(v, set())
            if not mids:
                continue
            left = np.zeros(da, dtype=np.float32)
            right = np.zeros(db, dtype=np.float32)
            c = 0
            for m in mids:
                left += fa[(u, m)]
                right += fb[(m, v)]
                c += 1
            left /= c
            right /= c
            feats_out.append(np.concatenate([left, right]))
            edges_out.append((u, v))

    if not edges_out:
        return (np.empty((2, 0), dtype=np.int64),
                np.empty((0, da + db), dtype=np.float32))
    return np.array(edges_out, dtype=np.int64).T, np.array(feats_out, dtype=np.float32)
