"""POIRec: POI recommendation via meta-path enhanced view contrastive learning.

Three complementary views are built over an augmented LBSN and contrasted with
InfoNCE to learn POI embeddings:

  * network schema view (S31)      : one-hop GNN over the augmented adjacency
  * meta-path view (S32)           : GNN per meta-path instance graph + attention
  * meta-path enhanced view (S33)  : edge embedding -> weight-based subgraph split
                                     -> intra-subgraph node attention -> inter-subgraph
                                     attention -> inter-meta-path attention

Loss (S4): meta-path-level contrast (Hp vs Yp~) plus graph-level contrasts
(enhanced view vs meta-path view, enhanced view vs network schema view).
Recommendation (S5) pools a user's visited-POI embeddings from the enhanced view.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter

MP_NAMES = ['LUL', 'LLL', 'LUUL']


def _act(name):
    return nn.PReLU() if name == 'prelu' else nn.ReLU()


# ----------------------------------------------------------------------------
# one-hop message passing (S31 network schema view, S32 meta-path view)
# ----------------------------------------------------------------------------
class GNNLayer(nn.Module):
    def __init__(self, d, activation='prelu'):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(d))
        self.act = _act(activation)
        nn.init.xavier_uniform_(self.lin.weight, gain=1.414)

    def forward(self, x, edge_index, N):
        if edge_index.shape[1] == 0:
            agg = torch.zeros(N, x.shape[1], device=x.device, dtype=x.dtype)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg = scatter(x[src], dst, dim=0, reduce='sum', dim_size=N)
            deg = scatter(torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype),
                          dst, dim=0, reduce='sum', dim_size=N)
            agg = agg / deg.clamp(min=1).unsqueeze(-1)
        return self.act(self.lin(agg) + self.bias + x)   # residual acts as self-loop


# ----------------------------------------------------------------------------
# node-level attention combining node features + edge features (S333)
# ----------------------------------------------------------------------------
class IntraAttention(nn.Module):
    def __init__(self, d, edge_dim, activation='prelu'):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, d)
        self.att = nn.Parameter(torch.empty(3 * d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.act = _act(activation)
        nn.init.xavier_uniform_(self.att.view(1, -1))
        nn.init.xavier_uniform_(self.edge_proj.weight, gain=1.414)

    def forward(self, x, edge_index, edge_feat):
        if edge_index.shape[1] == 0:
            return self.act(self.bias).expand(x.shape[0], -1)
        src, dst = edge_index[0], edge_index[1]
        e = self.edge_proj(edge_feat)
        cat = torch.cat([x[dst], x[src], e], dim=-1)
        score = F.leaky_relu((cat * self.att).sum(-1))
        alpha = pyg_softmax(score, dst)
        out = scatter(x[src] * alpha.unsqueeze(-1), dst, dim=0,
                      reduce='sum', dim_size=x.shape[0])
        return self.act(out + self.bias)


# ----------------------------------------------------------------------------
# semantic-level attention over a list of (N, d) embeddings (S334/S335, S32)
# ----------------------------------------------------------------------------
class SemanticAttention(nn.Module):
    def __init__(self, d, num_slots):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_slots, d, d))
        self.q = nn.Parameter(torch.empty(num_slots, d))
        self.b = nn.Parameter(torch.zeros(num_slots, d))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q.view(self.q.shape[0], -1))

    def forward(self, hs):
        scores = []
        for k, h in enumerate(hs):
            w = torch.tanh(h @ self.W[k].T + self.b[k])
            scores.append((self.q[k] * w).sum(-1).mean())
        beta = torch.softmax(torch.stack(scores), dim=0)
        return sum(beta[k] * hs[k] for k in range(len(hs)))


# ----------------------------------------------------------------------------
# full model
# ----------------------------------------------------------------------------
class POIRec(nn.Module):
    def __init__(self, args, nu, nl):
        super().__init__()
        d = args.dim
        act = args.activation
        self.nu = nu
        self.nl = nl
        self.d = d
        self.kmax = args.kmax

        self.proj_user = nn.Linear(6, d)
        self.proj_loc = nn.Linear(6, d)

        self.gnn_net = GNNLayer(d, act)                 # S31
        self.gnn_mp = GNNLayer(d, act)                  # S32 (shared)
        self.sem_mp = SemanticAttention(d, 3)           # fuse LUL/LLL/LUUL

        self.intra = IntraAttention(d, d, act)          # S333 (edge feat is d-dim)
        self.sem_sub = SemanticAttention(d, self.kmax)  # S334 inter-subgraph
        self.sem_meta = SemanticAttention(d, 3)         # S335 inter-meta-path

        self.act = _act(act)

    def forward(self, data):
        nu, nl, d = self.nu, self.nl, self.d
        N = nu + nl

        h_user = self.act(self.proj_user(data['user_feat']))
        h_loc = self.act(self.proj_loc(data['loc_feat']))
        X = torch.cat([h_user, h_loc], dim=0)

        # S31 network schema view
        Z = self.gnn_net(X, data['net_edge_index'], N)
        Z_loc = Z[nu:]

        # S32 meta-path view
        H = {}
        for p in MP_NAMES:
            H[p] = self.gnn_mp(h_loc, data['mp_edge'][p], nl)
        Y_mp = self.sem_mp([H[p] for p in MP_NAMES])

        # S33 meta-path enhanced view
        Ytilde = {}
        for p in MP_NAMES:
            sub_embs = []
            for k in range(1, self.kmax + 1):
                ei = data['mp_sub'][p][k]
                e = (Z_loc[ei[0]] + Z_loc[ei[1]]) / 2     # S331 edge embedding
                sub_embs.append(self.intra(h_loc, ei, e))  # S333 intra-subgraph
            Ytilde[p] = self.sem_sub(sub_embs)             # S334 inter-subgraph
        y_me = self.sem_meta([Ytilde[p] for p in MP_NAMES])  # S335 inter-meta-path

        return {'net': Z_loc, 'mp': Y_mp, 'me': y_me, 'H': H, 'Ytilde': Ytilde}
