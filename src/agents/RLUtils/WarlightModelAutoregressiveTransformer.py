import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =========================
# GNN Encoder
# =========================
class GNNLayer(nn.Module):
    def __init__(self, hidden_dim, residual=True, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.graph_norm = nn.LayerNorm(hidden_dim)  # simplified GraphNorm
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_h, edge_index, edge_h):
        src, dst = edge_index
        messages = node_h[src] + edge_h
        agg = torch.zeros_like(node_h)
        agg.index_add_(0, dst, messages)
        out = self.linear(agg)
        out = self.dropout(F.relu(out))
        out = self.graph_norm(out)
        if self.residual:
            out = out + node_h
        return self.norm(out)

class GNNEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, depth=3, skip_residuals=False):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([GNNLayer(hidden_dim, residual=not skip_residuals) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, edge_index, edge_feats):
        h = self.node_proj(node_feats)
        e = self.edge_proj(edge_feats)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return self.final_norm(h), e

# =========================
# Placement Head
# =========================
class PlacementHead(nn.Module):
    def __init__(self, hidden_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_nodes)
        self.norm = nn.LayerNorm(hidden_dim*2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embs, edge_embs, n_available_armies):
        edge_summary = edge_embs.mean(dim=0, keepdim=True).expand(node_embs.shape[0], -1)
        h = torch.cat([node_embs, edge_summary], dim=-1)
        h = self.norm(h)
        h = F.relu(self.fc(h))
        h = self.dropout(h)
        logits = self.out(h).mean(dim=0)
        dist = Categorical(logits=logits)
        placements = [dist.sample() for _ in range(n_available_armies)]
        logp = torch.stack([dist.log_prob(a) for a in placements]).sum()
        entropy = dist.entropy().sum()
        return placements, logp, entropy

# =========================
# Hierarchical Attack Decoder
# =========================
class AttackDecoder(nn.Module):
    def __init__(self, hidden_dim, num_edges, max_steps=20, nhead=4, num_layers=2, dropout=0.1, frac_bins=4):
        super().__init__()
        self.max_steps = max_steps
        self.num_edges = num_edges
        self.frac_bins = frac_bins

        self.input_fc = nn.Linear(hidden_dim*2, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.edge_fc = nn.Linear(hidden_dim, num_edges)
        self.frac_fc = nn.Linear(hidden_dim, frac_bins)
        self.stop_fc = nn.Linear(hidden_dim,1)

    def forward(self, node_embs, edge_embs, edge_index, armies_left):
        actions = {"edge_idx": [], "army_idx": []}
        logps, entropies = [], []

        memory = torch.cat([node_embs.mean(dim=0, keepdim=True), edge_embs.mean(dim=0, keepdim=True)], dim=-1)
        memory = self.input_fc(memory).unsqueeze(0)  # (1,1,H)
        tgt = torch.zeros((1,0,memory.size(-1)), device=memory.device)

        fraction_ranges = [(0.0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.0)]

        for _ in range(self.max_steps):
            out = self.transformer_decoder(tgt, memory)
            last_out = out[:,-1,:] if out.size(1)>0 else memory[:,0,:]

            edge_logits = self.edge_fc(last_out)
            stop_logits = self.stop_fc(last_out)
            logits = torch.cat([edge_logits, stop_logits], dim=-1)
            dist_edge = Categorical(logits=logits)
            edge_choice = dist_edge.sample()
            logp_edge = dist_edge.log_prob(edge_choice)
            entropy_edge = dist_edge.entropy()

            if edge_choice.item() == logits.shape[-1]-1:  # stop-token
                break

            source_region = edge_index[0, edge_choice]
            available_armies = armies_left[source_region]
            if available_armies <= 0:
                continue  # skip, or sample another edge

            # Hierarchical fraction selection
            frac_logits = self.frac_fc(last_out)
            dist_frac = Categorical(logits=frac_logits)
            frac_choice = dist_frac.sample()
            logp_frac = dist_frac.log_prob(frac_choice)
            entropy_frac = dist_frac.entropy()

            low, high = fraction_ranges[frac_choice.item()]
            # stochastic selection within fraction
            army_choice = max(1, min(available_armies, int(round(available_armies*(low + (high-low)*torch.rand(1).item())))))
            armies_left[source_region] -= army_choice

            actions["edge_idx"].append(edge_choice.item())
            actions["army_idx"].append(army_choice)
            
            logps.append(logp_edge + logp_frac)
            entropies.append(entropy_edge + entropy_frac)

            tgt = torch.cat([tgt, last_out.unsqueeze(1)], dim=1)

        return actions, torch.stack(logps).sum() if logps else torch.tensor(0.0), \
               torch.stack(entropies).sum() if entropies else torch.tensor(0.0)

# =========================
# Multi-layer Value Head
# =========================
class ValueHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self, node_embs):
        g = node_embs.mean(dim=0, keepdim=True)
        g = self.norm1(g)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        g = self.norm2(g)
        value = self.fc2(g)
        return value.squeeze()

# =========================
# Full Warlight Policy
# =========================
class WarlightPolicy(nn.Module):
    def __init__(self, node_dim, edge_dim, num_nodes, num_edges, hidden_dim=128, skip_residuals=False):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim, depth=3, skip_residuals=skip_residuals)
        self.placement_head = PlacementHead(hidden_dim, num_nodes)
        self.attack_decoder = AttackDecoder(hidden_dim, num_edges)
        self.value_head = ValueHead(hidden_dim)

    def forward(self, node_feats, edge_index, edge_feats, n_available_armies, armies_left):
        node_embs, edge_embs = self.encoder(node_feats, edge_index, edge_feats)
        placements, logp_p, ent_p = self.placement_head(node_embs, edge_embs, n_available_armies)
        attacks, logp_a, ent_a = self.attack_decoder(node_embs, edge_embs, edge_index, armies_left)
        value = self.value_head(node_embs)
        return {
            "placements": placements,
            "attacks": attacks,
            "logp": logp_p + logp_a,
            "entropy": ent_p + ent_a,
            "value": value
        }

# =========================
# Minimal Example
# =========================
if __name__ == "__main__":
    num_nodes, num_edges, node_dim, edge_dim = 10, 15, 8, 5
    node_feats = torch.randn(num_nodes, node_dim)
    edge_feats = torch.randn(num_edges, edge_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    n_available_armies = 5
    armies_left = torch.randint(1,5,(num_nodes,))  # armies currently available per node

    policy = WarlightPolicy(node_dim, edge_dim, num_nodes, num_edges, hidden_dim=64, skip_residuals=False)
    out = policy(node_feats, edge_index, edge_feats, n_available_armies, armies_left.tolist())

    print("Placements:", out["placements"])
    print("Attacks:", out["attacks"])
    print("Logp:", out["logp"].item())
    print("Entropy:", out["entropy"].item())
    print("Value:", out["value"].item())
