import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Graph Encoder ---
class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_residual=False):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # GraphNorm instead of LayerNorm (better for graph data)
        self.norm_nodes = GraphNorm(hidden_dim)
        self.norm_edges = GraphNorm(hidden_dim)

        self.use_residual = use_residual

    def forward(self, node_feats, edge_feats, edge_index):
        h_nodes = F.relu(self.node_proj(node_feats))
        h_edges = F.relu(self.edge_proj(edge_feats))
        h_nodes = self.dropout(h_nodes)
        h_edges = self.dropout(h_edges)

        src, dst = edge_index
        for _ in range(2):  # message passing iterations
            # Node update
            messages = h_edges + h_nodes[src]
            h_nodes_new = torch.zeros_like(h_nodes).index_add(0, dst, messages)
            if self.use_residual:
                h_nodes_new = h_nodes + h_nodes_new
            h_nodes = self.norm_nodes(F.relu(h_nodes_new))

            # Edge update
            h_edges_new = h_edges + (h_nodes[src] + h_nodes[dst]) / 2
            if self.use_residual:
                h_edges_new = h_edges + h_edges_new
            h_edges = self.norm_edges(F.relu(h_edges_new))

        return h_nodes, h_edges


class GraphNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


# --- Autoregressive Transformer Decoder ---
class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dropout=0.1,
            norm_first=True  # Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None):
        return self.decoder(tgt, memory, tgt_mask=tgt_mask)


# --- Pointer Head for Actions ---
class PointerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, keys, mask=None):
        q = self.q_proj(query)  # [B,1,H]
        k = self.k_proj(keys)   # [B,N,H]
        logits = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


# --- Full Model ---
class WarlightAutoregModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_residual=False):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim, use_residual=use_residual)
        self.decoder = ActionDecoder(hidden_dim)
        self.pointer = PointerHead(hidden_dim)

    def forward(self, node_feats, edge_feats, edge_index, tgt_seq, tgt_mask=None):
        h_nodes, h_edges = self.encoder(node_feats, edge_feats, edge_index)
        memory = h_nodes.unsqueeze(1)  # [B,N,H] → [B,N,H]
        decoded = self.decoder(tgt_seq, memory, tgt_mask)
        probs, logits = self.pointer(decoded, h_nodes)
        return probs, logits
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, dropout=0.1, use_residual=False):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.node_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

        self.edge_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        for l in range(self.num_layers):
            row, col = edge_index

            # Message passing: aggregate neighbor info
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, h[col])

            node_input = torch.cat([h, agg], dim=-1)
            new_h = self.node_updates[l](node_input)

            edge_input = torch.cat([h[row], h[col]], dim=-1)
            new_e = self.edge_updates[l](edge_input)

            if self.use_residual:
                h = h + new_h
                e = e + new_e
            else:
                h = new_h
                e = new_e

        return h, e

class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_actions, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation="gelu", norm_first=True  # Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, memory, action_queries, tgt_mask=None):
        decoded = self.decoder(action_queries, memory, tgt_mask=tgt_mask)
        logits = self.action_head(decoded)
        return logits

class AutoregressiveWarlightModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_actions, gnn_layers=3, gnn_dropout=0.1, use_residual=False):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim,
                                  num_layers=gnn_layers, dropout=gnn_dropout,
                                  use_residual=use_residual)
        self.decoder = ActionDecoder(hidden_dim, num_actions)

    def forward(self, x, edge_index, edge_attr, action_queries, tgt_mask=None):
        node_repr, edge_repr = self.encoder(x, edge_index, edge_attr)
        memory = node_repr  # memory passed to the decoder
        logits = self.decoder(memory, action_queries, tgt_mask)
        return logits
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, dropout=0.1, use_residual=False):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.node_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

        self.edge_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        for l in range(self.num_layers):
            row, col = edge_index

            # Message passing: aggregate neighbor info
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, h[col])

            node_input = torch.cat([h, agg], dim=-1)
            new_h = self.node_updates[l](node_input)

            edge_input = torch.cat([h[row], h[col]], dim=-1)
            new_e = self.edge_updates[l](edge_input)

            if self.use_residual:
                h = h + new_h
                e = e + new_e
            else:
                h = new_h
                e = new_e

        return h, e

class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_actions, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation="gelu", norm_first=True  # Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, memory, action_queries, tgt_mask=None):
        decoded = self.decoder(action_queries, memory, tgt_mask=tgt_mask)
        logits = self.action_head(decoded)
        return logits

class AutoregressiveWarlightModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_actions, gnn_layers=3, gnn_dropout=0.1, use_residual=False):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim,
                                  num_layers=gnn_layers, dropout=gnn_dropout,
                                  use_residual=use_residual)
        self.decoder = ActionDecoder(hidden_dim, num_actions)

    def forward(self, x, edge_index, edge_attr, action_queries, tgt_mask=None):
        node_repr, edge_repr = self.encoder(x, edge_index, edge_attr)
        memory = node_repr  # memory passed to the decoder
        logits = self.decoder(memory, action_queries, tgt_mask)
        return logits
