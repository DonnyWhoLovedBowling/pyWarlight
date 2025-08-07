import torch
import torch.nn.functional as f
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from src.game.Phase import Phase

class StableSAGEConv(nn.Module):
    """SAGE convolution with stability improvements"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sage = SAGEConv(in_channels, out_channels, normalize=True)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        x = self.norm(x)
        x = self.dropout(x)
        return f.relu(x)

class WarlightPolicyNetSAGE(nn.Module):
    """Warlight policy network using GraphSAGE (more stable than GCN)"""
    def __init__(self, node_feat_dim, embed_dim=64, max_army_send=50):
        super().__init__()
        
        # GraphSAGE layers (inherently more stable than GCN)
        self.sage1 = StableSAGEConv(node_feat_dim, embed_dim)
        self.sage2 = StableSAGEConv(embed_dim, embed_dim)
        self.sage3 = StableSAGEConv(embed_dim, embed_dim)

        # Smaller, more stable heads
        self.placement_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # Simplified attack head
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        self.army_scorer = nn.Sequential(
            nn.Linear(2 * embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, max_army_send)
        )

        # Much smaller value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),  # High dropout for value stability
            nn.Linear(16, 1)
        )
        
        self.max_army_send = max_army_send
        self.edge_tensor: torch.Tensor = None

    def get_node_embeddings(self, x, edge_index):
        """Get node embeddings with stability checks"""
        x = self.sage1(x, edge_index)
        x = self.sage2(x, edge_index)
        x = self.sage3(x, edge_index)
        
        # Clip embeddings to prevent explosion
        x = torch.clamp(x, min=-5.0, max=5.0)
        return x

    def get_value(self, node_features: torch.Tensor):
        edge_tensor = self.edge_tensor.to(node_features.device)
        node_embeddings = self.get_node_embeddings(node_features, edge_tensor)

        if node_embeddings.dim() == 2:
            graph_embedding = node_embeddings.mean(dim=0)
        else:
            graph_embedding = node_embeddings.mean(dim=1)
        
        value = self.value_head(graph_embedding)
        # Tight value clipping for stability
        value = torch.clamp(value, min=-50.0, max=50.0)
        return value.squeeze(-1)

    def forward(self, x, action_edges, army_counts, action: str=None, edge_mask=None):
        edge_index = self.edge_tensor.to(x.device)
        node_embeddings = self.get_node_embeddings(x, edge_index)
        
        placement_logits = torch.tensor([])
        attack_logits = torch.tensor([])
        army_logits = torch.tensor([])

        if action == Phase.PLACE_ARMIES or action is None:
            placement_logits = self.placement_head(node_embeddings).squeeze(-1)
            placement_logits = torch.clamp(placement_logits, min=-15.0, max=15.0)

        if action == Phase.ATTACK_TRANSFER or action is None:
            # Simplified attack logic
            if action_edges.dim() == 2:
                batch_size = 1
                action_edges = action_edges.unsqueeze(0)
                squeeze_output = True
            else:
                batch_size = action_edges.size(0)
                squeeze_output = False
                
            num_edges = action_edges.shape[-2]
            src = torch.clamp(action_edges[..., 0], min=0, max=node_embeddings.size(-2) - 1)
            tgt = torch.clamp(action_edges[..., 1], min=0, max=node_embeddings.size(-2) - 1)
            
            if node_embeddings.dim() == 2:
                node_embeddings = node_embeddings.unsqueeze(0)
            
            src_embed = torch.gather(node_embeddings, 1, 
                                   src.unsqueeze(-1).expand(-1, -1, node_embeddings.size(-1)))
            tgt_embed = torch.gather(node_embeddings, 1,
                                   tgt.unsqueeze(-1).expand(-1, -1, node_embeddings.size(-1)))
            
            edge_embed = torch.cat([src_embed, tgt_embed], dim=-1)
            edge_embed_flat = edge_embed.view(-1, edge_embed.shape[-1])
            
            # Stability: clip inputs
            edge_embed_flat = torch.clamp(edge_embed_flat, min=-3.0, max=3.0)
            
            attack_logits_flat = self.edge_scorer(edge_embed_flat).squeeze(-1)
            army_logits_flat = self.army_scorer(edge_embed_flat)
            
            attack_logits = attack_logits_flat.view(batch_size, num_edges)
            army_logits = army_logits_flat.view(batch_size, num_edges, self.max_army_send)
            
            # Clip outputs
            attack_logits = torch.clamp(attack_logits, min=-15.0, max=15.0)
            army_logits = torch.clamp(army_logits, min=-15.0, max=15.0)
            
            if squeeze_output:
                attack_logits = attack_logits.squeeze(0)
                army_logits = army_logits.squeeze(0)
            
            if edge_mask is not None:
                attack_logits = attack_logits.masked_fill(~edge_mask, -1e9)
                army_logits = army_logits.masked_fill(~edge_mask.unsqueeze(-1), -1e9)

        return placement_logits, attack_logits, army_logits
