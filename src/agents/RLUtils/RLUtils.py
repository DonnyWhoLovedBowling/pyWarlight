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
        if n > 1:
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

        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()
        batch_count = rewards.numel()

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


def pad_tensor_list(tensor_list, pad_value=-1, target_device=None):
    """Pad list of tensors to same shape"""
    if not tensor_list:
        return torch.tensor([])

    # Determine target device
    if target_device is None:
        target_device = tensor_list[0].device if tensor_list[0].numel() > 0 else 'cpu'

    # Find max dimensions
    max_dim0 = max(t.size(0) if t.numel() > 0 else 0 for t in tensor_list)
    # Check if we're dealing with 2D tensors by examining non-empty tensors
    is_2d = any(len(t.shape) > 1 and t.numel() > 0 for t in tensor_list)

    if is_2d:
        max_dim1 = max(t.size(1) if len(t.shape) > 1 and t.numel() > 0 else 0 for t in tensor_list)
        padded = torch.full((len(tensor_list), max_dim0, max_dim1), pad_value,
                            dtype=tensor_list[0].dtype, device=target_device)
    else:
        padded = torch.full((len(tensor_list), max_dim0), pad_value,
                            dtype=tensor_list[0].dtype, device=target_device)

    for i, tensor in enumerate(tensor_list):
        if tensor.numel() > 0:
            if len(tensor.shape) > 1:
                padded[i, :tensor.size(0), :tensor.size(1)] = tensor
            else:
                padded[i, :tensor.size(0)] = tensor

    return padded


class RolloutBuffer:
    def __init__(self):
        self.edges = []
        self.attacks = []
        self.placements = []
        self.placement_log_probs = []  # Store individual placement log probs
        self.attack_log_probs = []     # Store individual attack log probs
        self.rewards = []
        self.values = []
        self.dones = []
        # Store as list of individual tensors instead of concatenated
        self.starting_node_features_list = []
        self.post_placement_node_features_list = []
        self.end_features_list = []
        # Store region ownership for proper masking during PPO updates
        self.owned_regions_list = []

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
        padded = pad_tensor_list(self.attacks, pad_value=-1, target_device=device)
        return padded

    def get_placements(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        padded = pad_tensor_list(self.placements, pad_value=-1, target_device=device)
        return padded

    def get_placement_log_probs(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.placement_log_probs:
            return torch.tensor([], dtype=torch.float, device=device)
        return pad_tensor_list(self.placement_log_probs, pad_value=0.0, target_device=device)

    def get_attack_log_probs(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.attack_log_probs:
            return torch.tensor([], dtype=torch.float, device=device)
        return pad_tensor_list(self.attack_log_probs, pad_value=0.0, target_device=device)

    def get_rewards(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.rewards, dtype=torch.float, device=device)

    def get_values(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.values, dtype=torch.float, device=device)

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

    def get_end_features(self):
        """Return properly batched node features [batch_size, num_nodes, features]"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.end_features_list:
            return torch.empty((0, 0, 8), dtype=torch.float32, device=device)
        
        return torch.stack(self.end_features_list).to(device)

    def add(self, edges, attacks, placements, placement_log_probs, attack_log_probs, reward, value, done, starting_node_features, post_placement_node_features, end_features, owned_regions=None):
        self.edges.append(edges)
        attacks_tensor = torch.tensor(attacks, dtype=torch.long)
        placements_tensor = torch.tensor(placements, dtype=torch.long)
        self.attacks.append(attacks_tensor)
        self.placements.append(placements_tensor)
        
        # Store owned regions for masking during PPO updates
        if owned_regions is not None:
            self.owned_regions_list.append(torch.tensor(owned_regions, dtype=torch.long))
        else:
            self.owned_regions_list.append(None)
        
        # Store individual log probabilities
        if isinstance(placement_log_probs, torch.Tensor):
            self.placement_log_probs.append(placement_log_probs.detach().cpu())
        else:
            self.placement_log_probs.append(torch.tensor(placement_log_probs, dtype=torch.float))
            
        if isinstance(attack_log_probs, torch.Tensor):
            self.attack_log_probs.append(attack_log_probs.detach().cpu())
        else:
            self.attack_log_probs.append(torch.tensor(attack_log_probs, dtype=torch.float))
            
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        # Store individual tensors instead of concatenating
        self.starting_node_features_list.append(starting_node_features)
        self.post_placement_node_features_list.append(post_placement_node_features)
        self.end_features_list.append(end_features)

    def clear(self):
        self.__init__()
        
    def get_owned_regions(self):
        """Return the list of owned regions for each episode"""
        return self.owned_regions_list

    # ... rest of methods remain the same ...

def apply_placement_masking(placement_logits, owned_regions_list):
    """
    Apply the same masking to placement logits as used during action selection.
    
    Args:
        placement_logits: [batch_size, num_nodes] - raw placement logits from model
        owned_regions_list: list of tensors, each containing region IDs owned by agent
        
    Returns:
        masked_placement_logits: [batch_size, num_nodes] - logits with non-owned regions masked to -inf
    """
    if placement_logits.dim() != 2:
        raise ValueError(f"Expected placement_logits to be 2D [batch_size, num_nodes], got {placement_logits.shape}")
    
    batch_size, num_nodes = placement_logits.shape
    device = placement_logits.device
    
    # Clone to avoid modifying the original
    masked_logits = placement_logits.clone()
    
    for batch_idx, owned_regions in enumerate(owned_regions_list):
        if owned_regions is not None:
            # Create mask for all regions
            all_regions = set(range(num_nodes))
            owned_regions_set = set(owned_regions.tolist() if isinstance(owned_regions, torch.Tensor) else owned_regions)
            not_owned = list(all_regions.difference(owned_regions_set))
            
            # Mask non-owned regions to -inf
            if not_owned:
                masked_logits[batch_idx, not_owned] = float('-inf')
    
    return masked_logits


def load_checkpoint(policy, optimizer, path="checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        pass  # Starting fresh


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, last_value: torch.Tensor, dones: torch.Tensor, gamma=0.95, lam=0.95):
    device = values.device
    advantages = []
    gae = 0
    values = torch.cat([values, torch.tensor([last_value], dtype=values.dtype, device=device)])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages, device=device), torch.tensor(returns, device=device)


def compute_entropy(placement_logits, edge_logits, army_logits):
    if type(placement_logits) == int:
        return torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float), torch.tensor(0,  dtype=torch.float)
    
    # Check for NaN in logits
    if torch.isnan(placement_logits).any():
        placement_entropy = torch.tensor(0.0, device=placement_logits.device)
    else:
        try:
            # Check for problematic distributions (all -inf rows)
            all_inf_mask = torch.isinf(placement_logits).all(dim=-1)
            
            if all_inf_mask.any():
                # This should be rare now that we pre-fix logits in PPOAgent
                placement_logits_safe = placement_logits.clone()
                placement_logits_safe[all_inf_mask, 0] = 0.0
            else:
                placement_logits_safe = placement_logits
                
            # Use log_softmax for numerical stability - this handles -inf values correctly
            # -inf values will become 0 probability, finite values will be normalized
            placement_log_probs = f.log_softmax(placement_logits_safe, dim=-1)
            placement_probs = torch.exp(placement_log_probs)
            
            # Compute entropy only over valid (non-zero probability) regions
            # This preserves the entropy signal from the model's uncertainty over valid choices
            valid_mask = placement_probs > 1e-10  # Avoid log(0) issues
            masked_probs = placement_probs * valid_mask.float()
            masked_log_probs = placement_log_probs * valid_mask.float()
            
            # Renormalize to ensure it's still a valid probability distribution
            prob_sum = masked_probs.sum(dim=-1, keepdim=True)
            prob_sum = torch.clamp(prob_sum, min=1e-10)  # Avoid division by zero
            normalized_probs = masked_probs / prob_sum
            
            # Recompute log probs for normalized distribution
            normalized_log_probs = torch.log(torch.clamp(normalized_probs, min=1e-10))
            
            # Compute entropy: -sum(p * log(p)) over valid regions only
            placement_entropy = -(normalized_probs * normalized_log_probs).sum()
            
            # Check if entropy computation resulted in NaN
            if torch.isnan(placement_entropy):
                placement_entropy = torch.tensor(0.0, device=placement_logits.device)
                
        except RuntimeError as re:
            logging.error(re)
            placement_entropy = torch.tensor(0.0, device=placement_logits.device)

    if torch.isnan(edge_logits).any():
        edge_entropy = torch.tensor(0.0, device=edge_logits.device)
    else:
        try:
            # Check for problematic distributions (all -inf or mostly -inf with extreme values)
            all_inf_mask = torch.isinf(edge_logits).all(dim=-1)
            mostly_inf_mask = torch.isinf(edge_logits).sum(dim=-1) > (edge_logits.size(-1) * 0.8)
            
            if all_inf_mask.any() or mostly_inf_mask.any():
                # Create a safe version of the logits
                edge_logits_safe = edge_logits.clone()
                
                # For all -inf rows, set one element to 0
                if all_inf_mask.any():
                    edge_logits_safe[all_inf_mask, 0] = 0.0
                
                # For mostly -inf rows with extreme finite values, clamp the finite values
                if mostly_inf_mask.any():
                    finite_mask = ~torch.isinf(edge_logits_safe)
                    edge_logits_safe[finite_mask] = torch.clamp(edge_logits_safe[finite_mask], -10, 10)
            else:
                edge_logits_safe = edge_logits
                
            # Use log_softmax for numerical stability
            edge_log_probs = f.log_softmax(edge_logits_safe, dim=-1)
            edge_probs = torch.exp(edge_log_probs)
            edge_entropy = -(edge_probs * edge_log_probs).sum()
            
            # Check if entropy computation resulted in NaN
            if torch.isnan(edge_entropy):
                edge_entropy = torch.tensor(0.0, device=edge_logits.device)
                
        except RuntimeError as re:
            logging.error(re)
            edge_entropy = torch.tensor(0.0, device=edge_logits.device)

    army_entropy = torch.zeros(1, device=placement_logits.device)
    for logits in army_logits:
        probs = f.softmax(logits, dim=-1)
        log_probs = f.log_softmax(logits, dim=-1)
        army_entropy += -(probs * log_probs).sum()
    return placement_entropy, edge_entropy, army_entropy


def apply_placement_mask(placement_logits, owned_regions_list, num_regions):
    """
    Apply the same masking to placement logits that was used during action selection
    
    Args:
        placement_logits: [batch_size, num_regions] - raw placement logits
        owned_regions_list: List of owned regions for each batch item
        num_regions: Total number of regions
    
    Returns:
        masked_placement_logits: [batch_size, num_regions] - masked placement logits
    """
    masked_logits = placement_logits.clone()
    
    for batch_idx, owned_regions in enumerate(owned_regions_list):
        if owned_regions is not None:
            # Create mask: set non-owned regions to -inf
            all_regions = set(range(num_regions))
            not_owned = list(all_regions.difference(set(owned_regions.tolist())))
            if not_owned:
                masked_logits[batch_idx, not_owned] = float('-inf')
    
    return masked_logits


def compute_individual_log_probs(
    attacks: torch.Tensor,            # [batch_size, max_attacks, 3] (src, tgt, armies)
    attack_logits: torch.Tensor,      # [batch_size, 42] - PADDED
    army_logits: torch.Tensor,        # [batch_size, 42, max_army_send] - PADDED  
    placements: torch.Tensor,         # [batch_size, max_placements]
    placement_logits: torch.Tensor,   # [batch_size, num_nodes]
    action_edges: torch.Tensor        # [batch_size, 42, 2] - PADDED with (-1,-1)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute individual action log probabilities for per-action PPO ratio calculation.
    Returns separate placement and attack log probabilities for each action.
    
    Returns:
        placement_log_probs: [batch_size, max_placements] - log prob for each placement
        attack_log_probs: [batch_size, max_attacks] - log prob for each attack (edge + army)
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
    
    # --- Individual placement log-probs ---
    # Check for problematic values before log_softmax
    if torch.isnan(placement_logits).any():
        placement_logits = torch.where(torch.isnan(placement_logits), torch.tensor(-1e9, device=placement_logits.device), placement_logits)
    
    # Check for all-inf rows which would cause log_softmax to produce NaN
    all_inf_mask = torch.isinf(placement_logits).all(dim=-1)
    if all_inf_mask.any():
        # For all-inf rows, set one element to 0 to make log_softmax work
        placement_logits[all_inf_mask, 0] = 0.0
    
    placement_log_probs_full = f.log_softmax(placement_logits, dim=-1)  # [batch_size, num_nodes]
    
    # Check for NaN after log_softmax
    if torch.isnan(placement_log_probs_full).any():
        # Replace NaN with very negative value
        placement_log_probs_full = torch.where(torch.isnan(placement_log_probs_full), 
                                             torch.tensor(-1e9, device=placement_log_probs_full.device), 
                                             placement_log_probs_full)
    
    placement_mask = placements >= 0
    placement_log_probs = torch.gather(placement_log_probs_full, 1, placements.clamp(min=0))  # [batch_size, max_placements]
    
    # Replace -inf values after gather to prevent NaN propagation
    if torch.isinf(placement_log_probs).any():
        # CRITICAL FIX: Replace -inf with very negative but finite value to prevent NaN propagation
        # This happens when placement indices point to masked regions or padding
        placement_log_probs = torch.where(torch.isinf(placement_log_probs), 
                                        torch.tensor(-100.0, device=placement_log_probs.device), 
                                        placement_log_probs)
    
    # Check for NaN after gather
    if torch.isnan(placement_log_probs).any():
        # Replace NaN with very negative value
        placement_log_probs = torch.where(torch.isnan(placement_log_probs), 
                                        torch.tensor(-1e9, device=placement_log_probs.device), 
                                        placement_log_probs)
    
    placement_log_probs = placement_log_probs * placement_mask.float()  # Zero out padded placements
    
    # --- Individual attack log-probs ---
    # Handle case where there are no attacks
    if isinstance(attacks, list):
        has_attacks = len(attacks) > 0
    else:
        has_attacks = attacks.numel() > 0

    if not has_attacks:
        # No attacks case - create empty tensors with proper shapes
        max_attacks = attacks.shape[1] if attacks.numel() > 0 else 1
        attack_log_probs = torch.zeros(batch_size, max_attacks, dtype=torch.float, device=device)
    else:
        # Create vectorized edge lookup
        action_edges_flat = action_edges[:, :, 0] * num_nodes + action_edges[:, :, 1]  # [batch_size, 42]
        attacks_flat = attacks[:, :, 0] * num_nodes + attacks[:, :, 1]  # [batch_size, max_attacks]
        
        # Find matching edges using broadcasting
        edge_matches = (attacks_flat.unsqueeze(2) == action_edges_flat.unsqueeze(1))  # [batch_size, max_attacks, 42]
        edge_indices = edge_matches.to(torch.long).argmax(dim=2)  # Convert to long, then argmax

        # Mask valid attacks and edges
        attack_mask = (attacks[:, :, 0] >= 0)
        valid_match_mask = edge_matches.any(dim=2) & attack_mask  # [batch_size, max_attacks]

        # --- Individual attack log-probs (edge + army) ---
        edge_log_probs = f.log_softmax(attack_logits, dim=-1)  # [batch_size, 42]

        # Gather edge and army log-probs
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        edge_lp = edge_log_probs[batch_idx, edge_indices] * valid_match_mask.float()  # [batch_size, max_attacks]

        # Army log-probs
        army_logits_selected = army_logits[batch_idx, edge_indices]  # [batch_size, max_attacks, max_army_send]
        army_log_probs = f.log_softmax(army_logits_selected, dim=-1)

        # Clamp army indices to valid range and only gather for valid attacks
        # Army count in attacks is 1-indexed, but log probs are 0-indexed, so subtract 1
        army_indices = (attacks[:, :, 2] - 1).clamp(min=0, max=army_log_probs.size(2) - 1)  # Convert to 0-indexed
        army_lp = torch.gather(army_log_probs, 2, army_indices.unsqueeze(-1)).squeeze(-1)  # [batch_size, max_attacks]

        # Apply valid mask to army log-probs (zeros out padded attacks)
        army_lp = army_lp * valid_match_mask.float()

        attack_log_probs = edge_lp + army_lp  # [batch_size, max_attacks] - individual attack log probs

    # If input was single sample, squeeze batch dimension
    if is_single_sample:
        placement_log_probs = placement_log_probs.squeeze(0)
        attack_log_probs = attack_log_probs.squeeze(0)

    return placement_log_probs, attack_log_probs


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
    Returns total log probability by summing individual action log probabilities
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
    

    # Handle case where there are no attacks
    if isinstance(attacks, list):
        has_attacks = len(attacks) > 0
    else:
        has_attacks = attacks.numel() > 0

    if not has_attacks:
        # No attacks case - create empty tensors with proper shapes
        attack_log_prob = torch.zeros(batch_size, dtype=torch.float, device=device)
    else:
        # Create vectorized edge lookup
        action_edges_flat = action_edges[:, :, 0] * num_nodes + action_edges[:, :, 1]  # [batch_size, 42]

        attacks_flat = attacks[:, :, 0] * num_nodes + attacks[:, :, 1]  # [batch_size, max_attacks]
        # Find matching edges using broadcasting
        edge_matches = (
                    attacks_flat.unsqueeze(2) == action_edges_flat.unsqueeze(1))  # [batch_size, max_attacks, 42]
        edge_indices = edge_matches.to(torch.long).argmax(dim=2)  # Convert to long, then argmax

        # Mask valid attacks and edges
        attack_mask = (attacks[:, :, 0] >= 0)
        edge_mask = (action_edges[:, :, 0] >= 0)  # Mask padded (-1,-1) edges
        valid_match_mask = edge_matches.any(dim=2) & attack_mask  # [batch_size, max_attacks]

        # --- Attack log-probs (fully vectorized) ---
        edge_log_probs = f.log_softmax(attack_logits, dim=-1)  # [batch_size, 42]

        # Gather edge and army log-probs
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        edge_lp = edge_log_probs[batch_idx, edge_indices] * valid_match_mask.float()

        # Army log-probs - FIX: Handle negative army values
        army_logits_selected = army_logits[batch_idx, edge_indices]  # [batch_size, max_attacks, max_army_send]
        army_log_probs = f.log_softmax(army_logits_selected, dim=-1)

        # Clamp army indices to valid range and only gather for valid attacks
        army_indices = attacks[:, :, 2].clamp(min=0,
                                              max=army_log_probs.size(2) - 1)  # Clamp to [0, max_army_send-1]
        army_lp = torch.gather(army_log_probs, 2, army_indices.unsqueeze(-1)).squeeze(-1)

        # Apply valid mask to army log-probs (zeros out padded attacks)
        army_lp = army_lp * valid_match_mask.float()

        attack_log_prob = (edge_lp + army_lp).sum(dim=1)

    result = placement_log_prob + attack_log_prob

    # If input was single sample, return scalar instead of batch
    if is_single_sample:
        result = result.squeeze(0)

    return result
