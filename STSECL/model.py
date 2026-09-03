"""STSECL: Spatio-Temporal Semantic-Enhanced Contrastive Learning.

Two complementary views are built over an augmented LBSN and contrasted with
InfoNCE to learn user embeddings for friend recommendation:

  * edge-enhanced meta-path view  (hierarchical attention)
      intra-meta-path (node-level) -> inter-meta-path (semantic-level)
      -> inter-relationship-type (relationship-level)
  * heterogeneous hypergraph view (two-step type-specific attention)
      node -> hyperedge -> node

Only user embeddings from the meta-path view are used for recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter


def _act(name):
    return nn.PReLU() if name == 'prelu' else nn.ReLU()


# ----------------------------------------------------------------------------
# intra-meta-path (node-level) attention  -- Eq. (4) (5)
# ----------------------------------------------------------------------------
class IntraMetaPathAttention(nn.Module):
    def __init__(self, d, edge_dim, activation='prelu'):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, d)
        self.att = nn.Parameter(torch.empty(3 * d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.act = _act(activation)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att.view(1, -1))
        nn.init.xavier_uniform_(self.edge_proj.weight, gain=1.414)

    def forward(self, x, edge_index, edge_feat):
        src, dst = edge_index[0], edge_index[1]
        e = self.edge_proj(edge_feat)                 # (E, d)
        xi = x[dst]                                    # target node feature
        xj = x[src]                                    # source node feature
        cat = torch.cat([xi, xj, e], dim=-1)           # (E, 3d)
        score = F.leaky_relu((cat * self.att).sum(-1))  # (E,)
        alpha = pyg_softmax(score, dst)                # node-level attention
        msg = xj * alpha.unsqueeze(-1)
        out = scatter(msg, dst, dim=0, reduce='sum', dim_size=x.shape[0])
        return self.act(out + self.bias)


# ----------------------------------------------------------------------------
# semantic-level attention (inter-meta-path / inter-relationship)  -- Eq. (7) (9)
# ----------------------------------------------------------------------------
class SemanticAttention(nn.Module):
    """Fuse a list of (N, d) embeddings with a learned scalar weight per slot."""

    def __init__(self, d, num_slots):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_slots, d, d))
        self.q = nn.Parameter(torch.empty(num_slots, d))
        self.b = nn.Parameter(torch.zeros(num_slots, d))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q.view(self.q.shape[0], -1))

    def forward(self, hs):
        scores = []
        for k, h in enumerate(hs):
            w = torch.tanh(h @ self.W[k].T + self.b[k])   # (N, d)
            scores.append((self.q[k] * w).sum(-1).mean())  # scalar
        beta = torch.softmax(torch.stack(scores), dim=0)   # (num_slots,)
        z = sum(beta[k] * hs[k] for k in range(len(hs)))
        return z


# ----------------------------------------------------------------------------
# heterogeneous hypergraph view  -- Eq. (10)-(13)
# ----------------------------------------------------------------------------
class HypergraphView(nn.Module):
    def __init__(self, d, K, num_node_types=4, num_hyper_types=3, activation='prelu'):
        super().__init__()
        self.d = d
        self.K = K
        # hyperedge attention vector per hyperedge type, shape (num_hyper_types, K, d)
        self.a = nn.Parameter(torch.empty(num_hyper_types, K, d))
        # node attention vector per node type, shape (num_node_types, (1+K)*d)
        self.q = nn.Parameter(torch.empty(num_node_types, (1 + K) * d))
        self.act = _act(activation)
        self.lin = nn.Linear(K * d, d)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.a.view(self.a.shape[0], -1))
        nn.init.xavier_uniform_(self.q.view(self.q.shape[0], -1))
        nn.init.xavier_uniform_(self.lin.weight, gain=1.414)

    def forward(self, x, node_list, edge_list, hyper_type, node_type):
        # ---- node -> hyperedge (hyperedge attention) ----
        # score[i, k] = <x[node_list[i]], a[hyper_type[i], k]> without materialising
        # the (E_inc, K, d) intermediate that blows up GPU memory.
        x_a = torch.einsum('nd,tkd->ntk', x, self.a)            # (N, num_hyper_types, K)
        score = x_a[node_list, hyper_type]                      # (E_inc, K)
        eps = pyg_softmax(F.leaky_relu(score), edge_list)       # per-hyperedge softmax

        E = int(edge_list.max()) + 1
        hm = x[node_list]                                       # (E_inc, d)
        he = torch.zeros(E, self.K, self.d, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            he[:, k] = scatter(eps[:, k, None] * hm, edge_list, dim=0,
                               reduce='sum', dim_size=E)        # (E, d)
        he = self.act(he)

        # ---- hyperedge -> node (node attention) ----
        he_full = he.reshape(E, self.K * self.d)                # (E, Kd)
        node_q = x @ self.q[:, :self.d].T                       # (N, num_node_types)
        he_q = he_full @ self.q[:, self.d:].T                   # (E, num_node_types)
        nt = node_type[node_list]                               # (E_inc,)
        score2 = node_q[node_list, nt] + he_q[edge_list, nt]    # (E_inc,)
        delta = pyg_softmax(F.leaky_relu(score2), node_list)    # (E_inc,)

        z = torch.zeros(x.shape[0], self.K, self.d, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            z[:, k] = scatter(delta[:, None] * he[edge_list, k], node_list, dim=0,
                              reduce='sum', dim_size=x.shape[0])  # (N, d)
        z = self.act(z).reshape(x.shape[0], self.K * self.d)
        return self.lin(z)


# ----------------------------------------------------------------------------
# full model
# ----------------------------------------------------------------------------
class STSECL(nn.Module):
    def __init__(self, args, nu, nl, nc, nt):
        super().__init__()
        d = args.dim
        K = args.nhead
        act = args.activation

        # type-specific node projections  -- Eq. (1)
        self.proj_user = nn.Linear(64, d)
        self.proj_loc = nn.Linear(64, d)
        self.proj_cat = nn.Linear(nc, d)
        self.proj_time = nn.Linear(nt, d)

        # meta-path relationship feature dims (concatenated base edge features)
        meta_dims = {'F_UUU': 10, 'C_ULU': 8, 'C_ULLU': 10,
                     'P_UUU': 2, 'P_ULU': 2, 'P_ULLU': 4}
        self.meta_layers = nn.ModuleDict({
            mp: IntraMetaPathAttention(d, dim, act) for mp, dim in meta_dims.items()
        })
        # inter-meta-path aggregation per relationship type  -- Eq. (6) (7)
        self.sem_F = SemanticAttention(d, 1)
        self.sem_C = SemanticAttention(d, 2)
        self.sem_P = SemanticAttention(d, 3)
        # inter-relationship-type aggregation  -- Eq. (8) (9)
        self.relationship = SemanticAttention(d, 3)

        # heterogeneous hypergraph view  -- Eq. (10)-(13)
        self.hyper = HypergraphView(d, K, 4, 3, act)

        self.act = _act(act)

    def forward(self, data):
        # --- type projection ---
        h_user = self.act(self.proj_user(data['user_feat']))
        h_loc = self.act(self.proj_loc(data['loc_feat']))
        h_cat = self.act(self.proj_cat(data['cat_feat']))
        h_time = self.act(self.proj_time(data['time_feat']))

        # --- edge-enhanced meta-path view ---
        mp_emb = {}
        for mp, layer in self.meta_layers.items():
            mp_emb[mp] = layer(h_user, data['mp_edge_index'][mp], data['mp_edge_feat'][mp])

        z_F = self.sem_F([mp_emb['F_UUU']])
        z_C = self.sem_C([mp_emb['C_ULU'], mp_emb['C_ULLU']])
        z_P = self.sem_P([mp_emb['P_UUU'], mp_emb['P_ULU'], mp_emb['P_ULLU']])
        z_em = self.relationship([z_F, z_C, z_P])            # (nu, d)

        # --- heterogeneous hypergraph view ---
        h_all = torch.cat([h_user, h_loc, h_cat, h_time], dim=0)
        z_hg_all = self.hyper(h_all, data['hyper_node'], data['hyper_edge'],
                              data['hyper_type'], data['node_type'])
        z_hg = z_hg_all[:data['num_user']]                    # (nu, d)

        return z_em, z_hg
