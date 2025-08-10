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
    def __init__(self, embed_dim, hidden_dim, n_army_options=4):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.army_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Linear(128, n_army_options)
        )
        self.n_army_options = n_army_options
        # Army percentage options: 25%, 50%, 75%, 100%
        self.army_percentages = torch.tensor([n/n_army_options for n in range(1, n_army_options + 1)], dtype=torch.float32)


    def forward(self, node_embeddings: torch.Tensor, action_edges: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
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
                                 src_clamped.unsqueeze(-1).expand(-1, -1,
                                                                  embed_dim))  # [batch_size, num_edges, embed_dim]
        tgt_embed = torch.gather(node_embeddings, 1,
                                 tgt_clamped.unsqueeze(-1).expand(-1, -1,
                                                                  embed_dim))  # [batch_size, num_edges, embed_dim]
        edge_embed = torch.cat([src_embed, tgt_embed], dim=-1)  # [batch_size, num_edges, 2*embed_dim]
        # Reshape for linear layers: [batch_size * num_edges, 2*embed_dim]
        edge_embed_flat = edge_embed.view(-1, edge_embed.shape[-1])

        # Compute logits
        # Debug: Check tensor shapes and bounds
        if torch.isnan(edge_embed_flat).any() or torch.isinf(edge_embed_flat).any():
            print("WARNING: NaN or Inf detected in edge_embed_flat")
            try:
                edge_logits_flat = self.edge_scorer(edge_embed_flat).squeeze(-1)  # [batch_size * num_edges]
            except RuntimeError as e:
                print(f"CUDA error in edge_scorer: {e}")
                print(f"Input shape: {edge_embed_flat.shape}")
                torch.cuda.empty_cache()  # Clear GPU memory
                raise e

        edge_logits_flat = self.edge_scorer(edge_embed_flat).squeeze(-1)  # [batch_size * num_edges]
        army_logits_flat = self.army_scorer(edge_embed_flat)  # [batch_size * num_edges, n_army_options]

        # Reshape back to batched format
        edge_logits = edge_logits_flat.view(batch_size, num_edges)  # [batch_size, num_edges]
        army_logits = army_logits_flat.view(batch_size, num_edges, self.n_army_options)  # [batch_size, num_edges, n_army_options]

        # Use original indices for valid edge detection
        valid_edges = (src >= 0) & (tgt >= 0)  # Exclude padded edges (-1)

        # Hard penalty for self-attacks, only for valid edges
        invalid_self = (src == tgt)
        invalid_self_valid = invalid_self & valid_edges
        edge_logits = edge_logits - invalid_self_valid.float() * 100.0  # Use subtraction instead of indexing


        # ====== Hard mask invalid army amounts per edge ======
        # Get army percentages tensor on the correct device
        army_percentages = self.army_percentages.to(army_logits.device)  # [n_army_options]
                
        # Squeeze outputs if input was single sample
        if squeeze_output:
            edge_logits = edge_logits.squeeze(0)
            army_logits = army_logits.squeeze(0)

        return edge_logits, army_logits

    def get_army_count_from_percentage_choice(self, src_armies: torch.Tensor, army_choice_idx: torch.Tensor) -> torch.Tensor:
        """
        Convert army percentage choice index to actual army count.
        
        Args:
            src_armies: [batch_size, num_edges] or [num_edges] - armies available at source
            army_choice_idx: [batch_size, num_edges] or [num_edges] - index of chosen percentage (0-3)
            
        Returns:
            actual_armies: [batch_size, num_edges] or [num_edges] - actual army count to send
        """
        # Get army percentages on correct device
        army_percentages = self.army_percentages.to(src_armies.device)
        
        # Get the chosen percentage for each edge
        chosen_percentages = army_percentages[army_choice_idx]  # Same shape as army_choice_idx
        
        # Calculate actual armies to send
        actual_armies = torch.floor(src_armies * chosen_percentages)
        
        return actual_armies.long()

    def get_percentage_choice_from_army_count(self, src_armies: torch.Tensor, actual_armies: torch.Tensor) -> torch.Tensor:
        """
        Convert actual army count back to percentage choice index for log probability calculation.
        
        Args:
            src_armies: [batch_size, num_edges] or [num_edges] - armies available at source  
            actual_armies: [batch_size, num_edges] or [num_edges] - actual armies sent
            
        Returns:
            choice_idx: [batch_size, num_edges] or [num_edges] - index of closest percentage choice (0-3)
        """
        # Get army percentages on correct device
        army_percentages = self.army_percentages.to(src_armies.device)
        
        # Calculate actual percentage used
        actual_percentage = actual_armies.float() / src_armies.float()
        
        # Find closest percentage option
        # army_percentages: [4], actual_percentage: [...] -> differences: [..., 4]
        differences = torch.abs(actual_percentage.unsqueeze(-1) - army_percentages.unsqueeze(0))
        choice_idx = torch.argmin(differences, dim=-1)
        
        return choice_idx


class WarlightPolicyNet(nn.Module):
    def __init__(self, node_feat_dim, embed_dim=64, n_army_options=4):
        super().__init__()
        self.gnn1 = GCNConv(node_feat_dim, embed_dim)
        self.gnn2 = GCNConv(embed_dim, embed_dim)

        self.placement_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)  # One logit per node
        )

        self.attack_head = AttackHead(embed_dim, 64, n_army_options)

        # Value head: input is aggregated graph embedding
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.edge_tensor: torch.Tensor = None

    def get_value(self, node_features: torch.Tensor):
        edge_tensor = self.edge_tensor.to(node_features.device)

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

    def forward(self, x, action_edges, action: str=None, edge_mask=None):
        """
        x: [num_nodes, node_feat_dim]       # node features
        edge_index: [2, num_edges]          # graph structure
        action_edges: [num_actions, 2]      # list of (src, tgt) edges for attacks
        army_counts: [num_nodes]            # current army count on each node
        edge_mask: [num_actions]            # mask for valid edges (True = valid, False = padded)
        """
        # GNN
        edge_index = self.edge_tensor.to(x.device)
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
                node_embeddings, action_edges
            )
            
            # Apply edge mask to attack logits if provided
            if edge_mask is not None:
                attack_logits = attack_logits.masked_fill(~edge_mask, -1e9)
                army_logits = army_logits.masked_fill(~edge_mask.unsqueeze(-1), -1e9)

        return placement_logits, attack_logits, army_logits
