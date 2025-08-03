import torch
import torch.nn.functional as f
import torch.nn as nn
from torch_geometric.nn import GCNConv

from src.game.Phase import Phase

class TerritoryGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = f.relu(conv(x, edge_index))
        return x  # [num_nodes, hidden_dim]


class AttackHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim, max_army_send):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.army_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Linear(128, max_army_send)
        )
        self.max_army_send = max_army_send

    def forward(self, node_embeddings: torch.Tensor, action_edges: torch.Tensor, army_counts: torch.Tensor):
        src, tgt = action_edges[:, 0].to(node_embeddings.device), action_edges[:, 1].to(node_embeddings.device)
        edge_embed = torch.cat([node_embeddings[src], node_embeddings[tgt]], dim=-1)

        edge_logits = self.edge_scorer(edge_embed).squeeze(-1)  # [num_edges]
        army_logits = self.army_scorer(edge_embed)  # [num_edges, max_army_send]

        # ====== Soft discouragement for unlikely attacks ======
        src_armies = army_counts[src]
        tgt_armies = army_counts[tgt]

        bad_edges = (src_armies <= 2) | (tgt_armies >= 3 * src_armies)
        edge_logits = edge_logits - bad_edges.float() * 1.0  # subtract 1.0 as soft penalty

        invalid_self = src == tgt
        edge_logits[invalid_self] -= 100.0  # or -1e9 if you want hard mask

        # ====== Hard mask invalid army amounts per edge ======
        max_sendable = src_armies - 1
        army_mask = (
            torch.arange(self.max_army_send, device=army_logits.device)
            .unsqueeze(0)
        )  # [1, max_army_send]

        valid_mask = army_mask <= max_sendable.unsqueeze(1)  # [num_edges, max_army_send]
        army_logits[~valid_mask] = -1e9  # Mask out too-large moves

        return edge_logits, army_logits


class WarlightPolicyNet(nn.Module):
    def __init__(self, node_feat_dim, embed_dim=64, max_army_send=50):
        super().__init__()
        self.gnn1 = GCNConv(node_feat_dim, embed_dim)
        self.gnn2 = GCNConv(embed_dim, embed_dim)

        self.placement_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)  # One logit per node
        )

        self.attack_head = AttackHead(embed_dim, 64, max_army_send)

        # Value head: input is aggregated graph embedding
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.edge_tensor: torch.Tensor = None

    def get_value(self, node_features: torch.Tensor):
        edge_tensor = self.edge_tensor

        x = f.relu(self.gnn1(node_features, edge_tensor))
        node_embeddings = self.gnn2(x, edge_tensor)

        # Handle both single samples and batched samples
        if node_embeddings.dim() == 2:
            # Single sample: [num_nodes, embed_dim]
            graph_embedding = node_embeddings.mean(dim=0)
        else:
            # Batched samples: [batch_size, num_nodes, embed_dim]
            graph_embedding = node_embeddings.mean(dim=1)
        
        value = self.value_head(graph_embedding)
        return value.squeeze(-1)

    def forward(self, x, action_edges, army_counts, action: str=None):
        """
        x: [num_nodes, node_feat_dim]       # node features
        edge_index: [2, num_edges]          # graph structure
        action_edges: [num_actions, 2]      # list of (src, tgt) edges for attacks
        army_counts: [num_nodes]            # current army count on each node
        """
        # GNN
        edge_index = self.edge_tensor
        x = f.relu(self.gnn1(x, edge_index))
        node_embeddings = self.gnn2(x, edge_index)
        placement_logits = torch.tensor([])
        attack_logits = torch.tensor([])
        army_logits = torch.tensor([])

        if action == Phase.PLACE_ARMIES or action is None:
            # Placement
            placement_logits = self.placement_head(node_embeddings).squeeze(
                -1
            )  # [num_nodes]

        if action == Phase.ATTACK_TRANSFER or action is None:
            # Attack
            attack_logits, army_logits = self.attack_head(
                node_embeddings, action_edges, army_counts
            )

        return placement_logits, attack_logits, army_logits

