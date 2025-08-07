import torch
import torch.nn.functional as f
import torch.nn as nn
import math
from src.game.Phase import Phase

class PositionalEncoding(nn.Module):
    """Add positional encoding based on region coordinates"""
    def __init__(self, d_model, max_regions=200):
        super().__init__()
        self.position_embedding = nn.Embedding(max_regions, d_model)
        
    def forward(self, x, region_ids=None):
        if region_ids is None:
            region_ids = torch.arange(x.size(-2), device=x.device)
        pos_embed = self.position_embedding(region_ids)
        return x + pos_embed

class StableMultiHeadAttention(nn.Module):
    """Multi-head attention with gradient clipping and stability"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, adjacency_mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Clip attention scores to prevent explosion
        scores = torch.clamp(scores, min=-20.0, max=20.0)
        
        # Apply adjacency mask if provided (only attend to neighboring regions)
        if adjacency_mask is not None:
            scores = scores.masked_fill(~adjacency_mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        attn_weights = f.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        out = self.out_proj(out)
        
        return out, attn_weights

class TransformerBlock(nn.Module):
    """Transformer block with layer normalization and residual connections"""
    def __init__(self, embed_dim, num_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or 2 * embed_dim
        
        self.attention = StableMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, adjacency_mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(self.norm1(x), adjacency_mask)
        x = x + attn_out
        
        # Feedforward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x

class WarlightPolicyNetTransformer(nn.Module):
    """Transformer-based Warlight policy network"""
    def __init__(self, node_feat_dim, embed_dim=64, num_heads=4, num_layers=3, max_army_send=50):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output heads - much simpler than GNN versions
        self.placement_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        # Attack head - use attention to score edge importance
        self.edge_attention = StableMultiHeadAttention(embed_dim, num_heads=2)
        self.edge_scorer = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        self.army_scorer = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, max_army_send)
        )
        
        # Very small value head for stability
        self.value_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )
        
        self.max_army_send = max_army_send
        self.edge_tensor: torch.Tensor = None
        
    def create_adjacency_mask(self, batch_size, num_nodes, edge_index, device):
        """Create adjacency mask for attention (only attend to neighbors)"""
        adj_mask = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=device)
        
        # Allow self-attention
        for i in range(num_nodes):
            adj_mask[:, i, i] = True
            
        # Allow attention to neighbors
        if edge_index.numel() > 0:
            src, tgt = edge_index
            adj_mask[:, src, tgt] = True
            adj_mask[:, tgt, src] = True  # Undirected graph
            
        return adj_mask

    def get_node_embeddings(self, x, edge_index):
        """Process node features through transformer layers"""
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, num_nodes, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Create adjacency mask for local attention
        adj_mask = self.create_adjacency_mask(batch_size, num_nodes, edge_index, x.device)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, adj_mask)
            # Clip embeddings after each layer
            x = torch.clamp(x, min=-3.0, max=3.0)
        
        if squeeze_output:
            x = x.squeeze(0)
            
        return x

    def get_value(self, node_features: torch.Tensor):
        if self.edge_tensor is None:
            # Fallback: create empty edge tensor
            edge_tensor = torch.empty((2, 0), dtype=torch.long, device=node_features.device)
        else:
            edge_tensor = self.edge_tensor.to(node_features.device)
            
        node_embeddings = self.get_node_embeddings(node_features, edge_tensor)
        
        if node_embeddings.dim() == 2:
            graph_embedding = node_embeddings.mean(dim=0)
        else:
            graph_embedding = node_embeddings.mean(dim=1)
        
        value = self.value_head(graph_embedding)
        # Conservative value clipping
        value = torch.clamp(value, min=-30.0, max=30.0)
        return value.squeeze(-1)

    def forward(self, x, action_edges, army_counts, action: str=None, edge_mask=None):
        if self.edge_tensor is None:
            edge_tensor = torch.empty((2, 0), dtype=torch.long, device=x.device)
        else:
            edge_tensor = self.edge_tensor.to(x.device)
            
        node_embeddings = self.get_node_embeddings(x, edge_tensor)
        
        placement_logits = torch.tensor([])
        attack_logits = torch.tensor([])
        army_logits = torch.tensor([])

        if action == Phase.PLACE_ARMIES or action is None:
            if node_embeddings.dim() == 3:
                # Batch case
                placement_logits = self.placement_head(node_embeddings).squeeze(-1)
            else:
                # Single sample
                placement_logits = self.placement_head(node_embeddings).squeeze(-1)
            placement_logits = torch.clamp(placement_logits, min=-10.0, max=10.0)

        if action == Phase.ATTACK_TRANSFER or action is None:
            # Handle batch dimensions
            if action_edges.dim() == 2:
                batch_size = 1
                action_edges = action_edges.unsqueeze(0)
                squeeze_output = True
            else:
                batch_size = action_edges.size(0)
                squeeze_output = False
                
            if node_embeddings.dim() == 2:
                node_embeddings = node_embeddings.unsqueeze(0)
                
            num_edges = action_edges.shape[-2]
            embed_dim = node_embeddings.shape[-1]
            num_nodes = node_embeddings.shape[-2]
            
            src = torch.clamp(action_edges[..., 0], min=0, max=num_nodes - 1)
            tgt = torch.clamp(action_edges[..., 1], min=0, max=num_nodes - 1)
            
            # Gather embeddings for edge endpoints
            src_embed = torch.gather(node_embeddings, 1, 
                                   src.unsqueeze(-1).expand(-1, -1, embed_dim))
            tgt_embed = torch.gather(node_embeddings, 1,
                                   tgt.unsqueeze(-1).expand(-1, -1, embed_dim))
            
            edge_embed = torch.cat([src_embed, tgt_embed], dim=-1)
            edge_embed_flat = edge_embed.view(-1, 2 * embed_dim)
            
            # Conservative clipping
            edge_embed_flat = torch.clamp(edge_embed_flat, min=-2.0, max=2.0)
            
            attack_logits_flat = self.edge_scorer(edge_embed_flat).squeeze(-1)
            army_logits_flat = self.army_scorer(edge_embed_flat)
            
            attack_logits = attack_logits_flat.view(batch_size, num_edges)
            army_logits = army_logits_flat.view(batch_size, num_edges, self.max_army_send)
            
            # Conservative output clipping
            attack_logits = torch.clamp(attack_logits, min=-10.0, max=10.0)
            army_logits = torch.clamp(army_logits, min=-10.0, max=10.0)
            
            if squeeze_output:
                attack_logits = attack_logits.squeeze(0)
                army_logits = army_logits.squeeze(0)
            
            if edge_mask is not None:
                attack_logits = attack_logits.masked_fill(~edge_mask, -1e9)
                army_logits = army_logits.masked_fill(~edge_mask.unsqueeze(-1), -1e9)

        return placement_logits, attack_logits, army_logits
