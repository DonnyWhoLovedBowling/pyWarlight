from dataclasses import dataclass
import logging
import os
import numpy as np
import torch
import torch.nn.functional as f
from src.game import Game


class StatTracker:
    s0 = 0
    s1 = 0
    s2 = 0
    min = 1e9
    max = -1e9

    def log(self, value):
        self.s0 += 1
        self.s1 += value
        self.s2 += value * value
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

    def std(self):
        n = self.s0
        if n > 0:
            std = np.sqrt((n * self.s2 - self.s1 * self.s1) / (n * (n - 1)))
        else:
            std = 0.0
        return std

    def mean(self):
        n = self.s0
        if n > 0:
            mean = self.s1 / n
        else:
            mean = 0.0
        return mean

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max


class RewardNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-8

    def update(self, rewards: torch.Tensor):

        batch = torch.tensor(rewards, dtype=torch.float32, device=rewards.device)
        batch_mean = batch.mean().item()
        batch_var = batch.var(unbiased=False).item()
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        return (rewards - self.mean) / (self.var ** 0.5 + 1e-8)

@dataclass
class PrevStateBuffer:    
    def __init__(self, prev_state: Game, player_id: int):
        self.regions = set([r.get_id() for r in prev_state.regions_owned_by(player_id)])
        self.prev_continents = prev_state.get_bonus_armies(player_id)
        self.prev_armies = prev_state.number_of_armies_owned(player_id)
        self.prev_armies_enemy = sum(
                [prev_state.number_of_armies_owned(pid) for pid in range(1, prev_state.config.num_players + 1) if
                 pid != player_id]
                )
    
class RolloutBuffer:
    def __init__(self):
        self.edges = []
        self.attacks = []
        self.placements = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        # Store as list of individual tensors instead of concatenated
        self.starting_node_features_list = []
        self.post_placement_node_features_list = []

    def pad_tensor_list(self, tensor_list, pad_value=-1):
        """Pad list of tensors to same shape"""
        if not tensor_list:
            return torch.tensor([])
            
        # Find max dimensions
        max_dim0 = max(t.size(0) if t.numel() > 0 else 0 for t in tensor_list)
        if len(tensor_list[0].shape) > 1:
            max_dim1 = max(t.size(1) if t.numel() > 0 else 0 for t in tensor_list)
            padded = torch.full((len(tensor_list), max_dim0, max_dim1), pad_value, dtype=tensor_list[0].dtype)
        else:
            padded = torch.full((len(tensor_list), max_dim0), pad_value, dtype=tensor_list[0].dtype)
            
        for i, tensor in enumerate(tensor_list):
            if tensor.numel() > 0:
                if len(tensor.shape) > 1:
                    padded[i, :tensor.size(0), :tensor.size(1)] = tensor
                else:
                    padded[i, :tensor.size(0)] = tensor
                    
        return padded

    def get_edges(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Pad all edge tensors to 42 edges
        padded_edges = []
        for edges in self.edges:
            if len(edges) < 42:
                padding = torch.full((42 - len(edges), 2), -1, dtype=edges.dtype)
                padded = torch.cat([edges, padding], dim=0)
            else:
                padded = edges[:42]  # Truncate if somehow > 42
            padded_edges.append(padded)
        return torch.stack(padded_edges).to(device)

    def get_attacks(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        padded = self.pad_tensor_list(self.attacks)
        return padded.to(device)

    def get_placements(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        padded = self.pad_tensor_list(self.placements)
        return padded.to(device)

    def get_log_probs(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.log_probs, device=device)

    def get_rewards(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.rewards, device=device)

    def get_values(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.values, device=device)

    def get_dones(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.dones, device=device)

    def get_starting_node_features(self):
        """Return properly batched node features [batch_size, num_nodes, features]"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.starting_node_features_list:
            return torch.empty((0, 0, 8), dtype=torch.float32, device=device)
        
        # Stack individual tensors into batch
        return torch.stack(self.starting_node_features_list).to(device)

    def get_post_placement_node_features(self):
        """Return properly batched node features [batch_size, num_nodes, features]"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.post_placement_node_features_list:
            return torch.empty((0, 0, 8), dtype=torch.float32, device=device)
        
        return torch.stack(self.post_placement_node_features_list).to(device)

    def add(self, edges, attacks, placements, log_prob, reward, value, done, starting_node_features, post_placement_node_features):
        self.edges.append(edges)
        # Convert attacks and placements to torch.Tensor for compatibility with compute_log_probs
        attacks_tensor = torch.tensor(attacks, dtype=torch.long)
        placements_tensor = torch.tensor(placements, dtype=torch.long)
        self.attacks.append(attacks_tensor)
        self.placements.append(placements_tensor)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        # Store individual tensors instead of concatenating
        self.starting_node_features_list.append(starting_node_features)
        self.post_placement_node_features_list.append(post_placement_node_features)

    def clear(self):
        self.__init__()

    # ... rest of methods remain the same ...

def load_checkpoint(policy, optimizer, path="checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded checkpoint from", path)
    else:
        print("No checkpoint found, starting fresh.")


def compute_gae(rewards, values, last_value, dones, gamma=0.95, lam=0.95):
    advantages = []
    gae = 0
    values = torch.cat([values, torch.tensor([last_value], dtype=values.dtype, device=values.device)])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages), torch.tensor(returns)


def compute_entropy(placement_logits, edge_logits, army_logits):
    if type(placement_logits) == int:
        return torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float), torch.tensor(0,  dtype=torch.float)
    try:
        placement_probs = f.softmax(placement_logits, dim=-1)
    except RuntimeError as re:
        logging.error(re)
        raise re


    placement_entropy = -(
            placement_probs * f.log_softmax(placement_logits, dim=-1)
    ).sum()

    edge_probs = f.softmax(edge_logits, dim=-1)
    edge_entropy = -(edge_probs * f.log_softmax(edge_logits, dim=-1)).sum()

    army_entropy = torch.zeros(1)
    for logits in army_logits:
        probs = f.softmax(logits, dim=-1)
        log_probs = f.log_softmax(logits, dim=-1)
        army_entropy += -(probs * log_probs).sum()
    return placement_entropy, edge_entropy, army_entropy


def compute_log_probs(
    attacks: torch.Tensor,            # [batch_size, max_attacks, 3] (src, tgt, armies)
    attack_logits: torch.Tensor,      # [batch_size, 42] - PADDED
    army_logits: torch.Tensor,        # [batch_size, 42, max_army_send] - PADDED  
    placements: torch.Tensor,         # [batch_size, max_placements]
    placement_logits: torch.Tensor,   # [batch_size, num_nodes]
    action_edges: torch.Tensor        # [batch_size, 42, 2] - PADDED with (-1,-1)
) -> torch.Tensor:
    """
    Fully vectorized with consistent 42-edge padding across all batches
    """
    device = placement_logits.device
    
    # Handle non-batch inputs by adding batch dimension
    if placement_logits.dim() == 1:
        # Single sample case - convert all inputs to batch format
        num_nodes = placement_logits.size(0)
        placement_logits = placement_logits.unsqueeze(0)  # [1, num_nodes]
        
        # Handle other tensors that might also be non-batched
        if attack_logits.dim() == 1:
            attack_logits = attack_logits.unsqueeze(0)  # [1, 42]
        if army_logits.dim() == 2:
            army_logits = army_logits.unsqueeze(0)  # [1, 42, max_army_send]
        if placements.dim() == 1:
            placements = placements.unsqueeze(0)  # [1, max_placements]
        if attacks.dim() == 2:
            attacks = attacks.unsqueeze(0)  # [1, max_attacks, 3]
        if action_edges.dim() == 2:
            action_edges = action_edges.unsqueeze(0)  # [1, 42, 2]
            
        batch_size = 1
        is_single_sample = True
    else:
        # Batch case - placement_logits is [batch_size, num_nodes]
        num_nodes = placement_logits.size(1)
        batch_size = placement_logits.size(0)
        is_single_sample = False
    
    # --- Placement log-probs (vectorized) ---
    placement_log_probs = f.log_softmax(placement_logits, dim=-1)
    placement_mask = placements >= 0
    placement_log_prob = torch.gather(placement_log_probs, 1, placements.clamp(min=0))
    placement_log_prob = (placement_log_prob * placement_mask.float()).sum(dim=1)
    
    # --- Attack log-probs (fully vectorized) ---
    edge_log_probs = f.log_softmax(attack_logits, dim=-1)  # [batch_size, 42]
    
    # Create vectorized edge lookup
    action_edges_flat = action_edges[:, :, 0] * num_nodes + action_edges[:, :, 1]  # [batch_size, 42]
    attacks_flat = attacks[:, :, 0] * num_nodes + attacks[:, :, 1]  # [batch_size, max_attacks]
    
    # Find matching edges using broadcasting
    edge_matches = (attacks_flat.unsqueeze(2) == action_edges_flat.unsqueeze(1))  # [batch_size, max_attacks, 42]
    edge_indices = edge_matches.to(torch.long).argmax(dim=2)  # Convert to long, then argmax
    
    # Mask valid attacks and edges
    attack_mask = (attacks[:, :, 0] >= 0)
    edge_mask = (action_edges[:, :, 0] >= 0)  # Mask padded (-1,-1) edges
    valid_match_mask = edge_matches.any(dim=2) & attack_mask  # [batch_size, max_attacks]
    
    # Gather edge and army log-probs
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    edge_lp = edge_log_probs[batch_idx, edge_indices] * valid_match_mask.float()
    
    # Army log-probs - FIX: Handle negative army values
    army_logits_selected = army_logits[batch_idx, edge_indices]  # [batch_size, max_attacks, max_army_send]
    army_log_probs = f.log_softmax(army_logits_selected, dim=-1)
    
    # Clamp army indices to valid range and only gather for valid attacks
    army_indices = attacks[:, :, 2].clamp(min=0, max=army_log_probs.size(2)-1)  # Clamp to [0, max_army_send-1]
    army_lp = torch.gather(army_log_probs, 2, army_indices.unsqueeze(-1)).squeeze(-1)
    
    # Apply valid mask to army log-probs (zeros out padded attacks)
    army_lp = army_lp * valid_match_mask.float()
    
    attack_log_prob = (edge_lp + army_lp).sum(dim=1)
    
    result = placement_log_prob + attack_log_prob
    
    # If input was single sample, return scalar instead of batch
    if is_single_sample:
        result = result.squeeze(0)
    
    return result
