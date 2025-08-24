import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class NodeEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.ln(x)
        return self.net(x)

class EdgeEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.ln(x)
        return self.net(x)

class TransformerActorCritic(nn.Module):
    def __init__(self,
                 num_regions=42,
                 node_feature_dim=17,
                 edge_feature_dim=5,
                 army_options=4,
                 hidden_dim=128,
                 num_layers=4,
                 num_heads=4,
                 dropout=0.15):
        super().__init__()
        self.num_regions = num_regions
        self.edge_tensor = None  # Placeholder for edge tensor, will be registered later
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.army_options = army_options
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout_embed = nn.Dropout(dropout)
        self.dropout_head = nn.Dropout(dropout)

        self.node_embed = NodeEmbedding(node_feature_dim, hidden_dim, dropout)
        self.edge_embed = EdgeEmbedding(edge_feature_dim, hidden_dim, dropout)

        # Pre-LN Transformer Encoder for nodes
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_transformer_ln = nn.LayerNorm(hidden_dim)

        # Heads
        self.placement_head = nn.Sequential(
            nn.RMSNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        self.edge_scorer = nn.Sequential(
            nn.RMSNorm(3*hidden_dim),
            nn.Linear(3*hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.army_scorer = nn.Sequential(
            nn.RMSNorm(3*hidden_dim),
            nn.Linear(3*hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, army_options)
        )
        self.value_head = nn.Sequential(
            nn.RMSNorm(3*hidden_dim),
            nn.Linear(3*hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )



    @property
    def recommended_weight_decay(self):
        return 1e-5

    def forward(self, node_features, action_edges=None, action=None, edge_mask=None, edge_features=None):
        # node_features: [batch, num_nodes, node_feature_dim] or [num_nodes, node_feature_dim]
        # action_edges: [batch, num_edges, 2] or [num_edges, 2]
        # edge_features: [batch, num_edges, edge_feature_dim] or [num_edges, edge_feature_dim]
        is_batch = node_features.dim() == 3

        # Node embedding and transformer
        node_x = self.node_embed(node_features)
        node_x = self.dropout_embed(node_x)
        node_x = self.transformer_encoder(node_x)
        node_x = self.post_transformer_ln(node_x)
        node_x = self.dropout(node_x)

        src_idx = action_edges[..., 0]
        tgt_idx = action_edges[..., 1]
        edge_emb = self.edge_embed(edge_features)

        # Mask for valid edges
        valid_mask = (src_idx != -1) & (tgt_idx != -1)
        if not is_batch:
            valid_mask = torch.tensor(valid_mask, device=src_idx.device)

        # Get source/target node embeddings, fully vectorized
        if is_batch:
            safe_src_idx = torch.where(valid_mask, src_idx, torch.zeros_like(src_idx))
            safe_tgt_idx = torch.where(valid_mask, tgt_idx, torch.zeros_like(tgt_idx))
            src_emb = torch.gather(node_x, 1, safe_src_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
            tgt_emb = torch.gather(node_x, 1, safe_tgt_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        else:
            safe_src_idx = torch.where(valid_mask, src_idx, torch.zeros_like(src_idx))
            safe_tgt_idx = torch.where(valid_mask, tgt_idx, torch.zeros_like(tgt_idx))
            src_emb = node_x[safe_src_idx]
            tgt_emb = node_x[safe_tgt_idx]

        src_emb = src_emb * valid_mask.unsqueeze(-1)
        tgt_emb = tgt_emb * valid_mask.unsqueeze(-1)

        edge_pair = torch.cat([src_emb, tgt_emb, edge_emb], dim=-1)
        edge_pair = self.dropout(edge_pair)
        attack_logits = self.edge_scorer(edge_pair)
        army_logits = self.army_scorer(edge_pair)
        placement_logits = self.placement_head(node_x).squeeze(-1)  # [batch, num_regions] or [num_regions]

        # Mask invalid edges in attack_logits and army_logits
        mask_value = -1e9
        attack_logits = attack_logits.masked_fill(~valid_mask[..., None], mask_value).squeeze(-1)  # [batch, num_edges] or [num_edges]
        army_logits = army_logits.masked_fill(~valid_mask.unsqueeze(-1), mask_value)

        # Placement head (per node)
        placement_logits = self.placement_head(node_x).squeeze(-1)  # [batch, num_regions] or [num_regions]

        # Value head (global)
        value = self.value_head(edge_pair)

        return placement_logits, attack_logits, army_logits, value

    def get_value(self, node_features, edge_features, action_edges):

        is_batch = node_features.dim() == 3

        # Node embedding and transformer
        node_x = self.node_embed(node_features)
        node_x = self.dropout_embed(node_x)
        node_x = self.transformer_encoder(node_x)
        node_x = self.post_transformer_ln(node_x)
        node_x = self.dropout(node_x)

        src_idx = action_edges[..., 0]
        tgt_idx = action_edges[..., 1]
        edge_emb = self.edge_embed(edge_features)

        # Mask for valid edges
        valid_mask = (src_idx != -1) & (tgt_idx != -1)
        if not is_batch:
            valid_mask = torch.tensor(valid_mask, device=src_idx.device)

        # Get source/target node embeddings, fully vectorized
        if is_batch:
            safe_src_idx = torch.where(valid_mask, src_idx, torch.zeros_like(src_idx))
            safe_tgt_idx = torch.where(valid_mask, tgt_idx, torch.zeros_like(tgt_idx))
            src_emb = torch.gather(node_x, 1, safe_src_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
            tgt_emb = torch.gather(node_x, 1, safe_tgt_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        else:
            safe_src_idx = torch.where(valid_mask, src_idx, torch.zeros_like(src_idx))
            safe_tgt_idx = torch.where(valid_mask, tgt_idx, torch.zeros_like(tgt_idx))
            src_emb = node_x[safe_src_idx]
            tgt_emb = node_x[safe_tgt_idx]

        src_emb *= valid_mask.unsqueeze(-1)
        tgt_emb *= valid_mask.unsqueeze(-1)

        edge_pair = torch.cat([src_emb, tgt_emb, edge_emb], dim=-1)
        edge_pair = self.dropout(edge_pair)

        edge_pair_mean = edge_pair.mean(dim=1) if is_batch else edge_pair.mean(dim=0)
        value = self.value_head(edge_pair_mean)

        if is_batch:
            value = value.squeeze(-1)
        return value
