import torch
import torch.nn.functional as f
import torch.nn as nn
from torch_geometric.nn import GCNConv
from src.game.Phase import Phase

class ResidualGCNLayer(nn.Module):
    """GCN layer with residual connection and layer normalization"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
        
        # Projection layer if dimensions don't match
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x, edge_index):
        identity = x
        
        # Apply GCN
        out = self.gcn(x, edge_index)
        out = self.norm(out)
        out = self.dropout(out)
        
        # Add residual connection
        if self.projection is not None:
            identity = self.projection(identity)
        
        out = out + identity
        return f.relu(out)

class StabilizedTerritoryGNN(nn.Module):
    """Territory GNN with residual connections and normalization"""
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(ResidualGCNLayer(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(ResidualGCNLayer(hidden_channels, hidden_channels))
    
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

class StabilizedAttackHead(nn.Module):
    """Attack head with gradient clipping and normalization"""
    def __init__(self, embed_dim, hidden_dim, n_army_options=4):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.army_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, n_army_options)
        )
        self.n_army_options = n_army_options
        self.army_percentages = torch.tensor([n/n_army_options for n in range(1, n_army_options + 1)], dtype=torch.float32)

    def forward(self, node_embeddings: torch.Tensor, action_edges: torch.Tensor):
        """
        Forward pass for the attack head, handling both single samples and batched samples.

        Args:
            node_embeddings: Node embeddings from GNN
                - Single sample: [num_nodes, embed_dim]
                - Batched: [batch_size, num_nodes, embed_dim]
            action_edges: Edge indices for possible actions
                - Single sample: [num_edges, 2]
                - Batched: [batch_size, num_edges, 2]

        Returns:
            tuple containing:
                - edge_logits: Logits for edge selection (same batch structure as input)
                - army_logits: Logits for army percentage selection (same batch structure as input)
        """
        # Handle both single samples and batched samples
        if node_embeddings.dim() == 2:
            # Single sample: [num_nodes, embed_dim]
            batch_size = 1
            num_nodes, embed_dim = node_embeddings.shape
            node_embeddings = node_embeddings.unsqueeze(0)  # [1, num_nodes, embed_dim]
            action_edges = action_edges.unsqueeze(0)  # [1, num_edges, 2]
            squeeze_output = True
        else:
            # Batched samples: [batch_size, num_nodes, embed_dim]
            batch_size, num_nodes, embed_dim = node_embeddings.shape
            squeeze_output = False

        num_edges = action_edges.shape[-2]

        # Extract source and target indices
        src = action_edges[..., 0].to(node_embeddings.device)  # [batch_size, num_edges]
        tgt = action_edges[..., 1].to(node_embeddings.device)  # [batch_size, num_edges]

        # Handle padded elements (-1) by clamping to valid indices
        src_clamped = torch.clamp(src, min=0, max=num_nodes - 1)
        tgt_clamped = torch.clamp(tgt, min=0, max=num_nodes - 1)

        # Gather node embeddings for source and target nodes
        src_embed = torch.gather(node_embeddings, 1,
                                src_clamped.unsqueeze(-1).expand(-1, -1, embed_dim))
        tgt_embed = torch.gather(node_embeddings, 1,
                                tgt_clamped.unsqueeze(-1).expand(-1, -1, embed_dim))

        # Concatenate source and target embeddings
        edge_embed = torch.cat([src_embed, tgt_embed], dim=-1)
        edge_embed_flat = edge_embed.view(-1, edge_embed.shape[-1])

        # Apply gradient clipping to embeddings
        edge_embed_flat = torch.clamp(edge_embed_flat, min=-10.0, max=10.0)

        # Compute edge and army logits
        edge_logits_flat = self.edge_scorer(edge_embed_flat).squeeze(-1)
        army_logits_flat = self.army_scorer(edge_embed_flat)

        # Reshape to batch format
        edge_logits = edge_logits_flat.view(batch_size, num_edges)
        army_logits = army_logits_flat.view(batch_size, num_edges, self.n_army_options)

        # Clip logits to prevent explosion
        edge_logits = torch.clamp(edge_logits, min=-20.0, max=20.0)
        army_logits = torch.clamp(army_logits, min=-20.0, max=20.0)

        if squeeze_output:
            edge_logits = edge_logits.squeeze(0)
            army_logits = army_logits.squeeze(0)

        return edge_logits, army_logits

class WarlightPolicyNetResidual(nn.Module):
    """Warlight policy network with residual connections and stabilization"""
    def __init__(self, node_feat_dim, embed_dim=64, n_army_options=4):
        super().__init__()
        self.gnn = StabilizedTerritoryGNN(node_feat_dim, embed_dim, num_layers=3)

        self.placement_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.attack_head = StabilizedAttackHead(embed_dim, 64, n_army_options)

        # Stabilized value head with smaller network
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Higher dropout for value head
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.edge_tensor: torch.Tensor = None

    def get_value(self, node_features: torch.Tensor):
        edge_tensor = self.edge_tensor.to(node_features.device)
        node_embeddings = self.gnn(node_features, edge_tensor)

        if node_embeddings.dim() == 2:
            graph_embedding = node_embeddings.mean(dim=0)
        else:
            graph_embedding = node_embeddings.mean(dim=1)
        
        # Clip graph embedding to prevent value explosion
        graph_embedding = torch.clamp(graph_embedding, min=-5.0, max=5.0)
        value = self.value_head(graph_embedding)
        
        # Clip final value prediction
        value = torch.clamp(value, min=-100.0, max=100.0)
        return value.squeeze(-1)

    def forward(self, x, action_edges, action: str=None, edge_mask=None):
        edge_index = self.edge_tensor.to(x.device)
        node_embeddings = self.gnn(x, edge_index)
        
        placement_logits = torch.tensor([])
        attack_logits = torch.tensor([])
        army_logits = torch.tensor([])

        if action == Phase.PLACE_ARMIES or action is None:
            placement_logits = self.placement_head(node_embeddings).squeeze(-1)
            # Clip placement logits
            placement_logits = torch.clamp(placement_logits, min=-20.0, max=20.0)

        if action == Phase.ATTACK_TRANSFER or action is None:
            attack_logits, army_logits = self.attack_head(node_embeddings, action_edges)
            
            if edge_mask is not None:
                attack_logits = attack_logits.masked_fill(~edge_mask, -1e9)
                army_logits = army_logits.masked_fill(~edge_mask.unsqueeze(-1), -1e9)

        return placement_logits, attack_logits, army_logits
