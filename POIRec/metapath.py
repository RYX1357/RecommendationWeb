"""Sparse meta-path materialisation helpers for POIRec (numpy/scipy).

The three POI-anchored meta-paths (LUL / LLL / LUUL) produce POI-POI
instance graphs whose edge weights are the number of connecting meta-path
instances. This module builds those weighted graphs from sparse matrix
products and sparsifies them with a per-row top-K.
"""

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


def to_directed(edges, weights):
    """Symmetrise a (2, E) edge tensor (with per-edge weights) into both directions."""
    src, dst = edges[0], edges[1]
    e = np.concatenate([np.stack([src, dst], 1), np.stack([dst, src], 1)], 0).T
    w = np.concatenate([weights, weights], 0)
    return e, w


def mat_to_weighted_edges(M, topk=20):
    """Convert a symmetric (n, n) path-count matrix to directed edges + weights.

    Per-row top-K sparsification keeps the graph tractable; the returned graph is
    symmetrised and de-duplicated. Returns (edges (2, E), weights (E,)).
    """
    M = M.tolil()
    M.setdiag(0)
    M = M.tocsr()
    M.eliminate_zeros()

    kept = {}
    for i in range(M.shape[0]):
        start, end = M.indptr[i], M.indptr[i + 1]
        if start == end:
            continue
        idx = M.indices[start:end]
        val = M.data[start:end]
        if len(idx) > topk:
            order = np.argpartition(val, -topk)[-topk:]
            idx = idx[order]
            val = val[order]
        for j, v in zip(idx, val):
            j = int(j)
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            kept[key] = max(kept.get(key, 0.0), float(v))

    keys = sorted(kept.keys())
    edges = np.array(keys, dtype=np.int64).T
    weights = np.array([kept[k] for k in keys], dtype=np.float32)
    if edges.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return to_directed(edges, weights)
