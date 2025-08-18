from dataclasses import dataclass
import logging
import os
import numpy as np
import torch
import torch.nn.functional as f
from src.game import Game


class StatTracker:
    def __init__(self, alpha=0.7):
        self.s0 = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.min = 1e9
        self.max = -1e9
        self.mean_val = 0.0
        self.var_val = 0.0
        self.alpha = alpha

    def log(self, value):
        self.s0 += 1
        self.s1 += value
        self.s2 += value * value
        if value > self.max:
            self.max = value
        if value < self.min:
            self.min = value

        # Update moving average and variance
        if self.s0 == 1:
            self.mean_val = value
            self.var_val = 0.0
        else:
            prev_mean = self.mean_val
            self.mean_val = (1 - self.alpha) * self.mean_val + self.alpha * value
            self.var_val = (1 - self.alpha) * self.var_val + self.alpha * (value - prev_mean) ** 2

    def std(self):
        if self.s0 > 1:
            return np.sqrt(self.var_val)
        else:
            return 0.0

    def mean(self):
        return self.mean_val

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

class RewardNormalizer:
    def __init__(self, alpha=0.1):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-8
        self.alpha = alpha

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

        # Exponential moving average update
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * batch_var
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
        self.starting_edge_features = []
        self.end_edge_features = []

    def get_edges(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Pad all edge tensors to num_edges edges
        # padded_edges = []
        # for edges in self.edges:
        #     if len(edges) < num_edges:
        #         padding = torch.full((num_edges - len(edges), 2), -1, dtype=edges.dtype)
        #         padded = torch.cat([edges, padding], dim=0)
        #     else:
        #         padded = edges[:num_edges]  # Truncate if somehow > num_edges
        #     padded_edges.append(padded)
        return torch.stack(self.edges).to(device)

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

    def get_starting_edge_features(self):
        """Return properly batched node features [batch_size, num_nodes, features]"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.starting_edge_features:
            return torch.empty((0, 0, 3), dtype=torch.float32, device=device)

        return torch.stack(self.starting_edge_features).to(device)

    def get_end_edge_features(self):
        """Return properly batched node features [batch_size, num_nodes, features]"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.end_edge_features:
            return torch.empty((0, 0, 3), dtype=torch.float32, device=device)

        return torch.stack(self.end_edge_features).to(device)

    def add(self, edges, attacks, placements, placement_log_probs, attack_log_probs, reward, value, done, starting_node_features, post_placement_node_features, end_features, owned_regions=None, starting_edge_features=None, end_edge_features=None):
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
        self.starting_edge_features.append(starting_edge_features)
        self.end_edge_features.append(end_edge_features)


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
    """
    Compute Generalized Advantage Estimation (GAE) for batch of episodes.
    
    In PPO, we typically have a batch of episodes, each contributing one reward/value/done per update.
    This is different from a single episode with multiple timesteps.
    
    Args:
        rewards: [B] - rewards for each episode in batch
        values: [B] - value estimates for each episode  
        last_value: [B] or scalar - value estimate for final state of each episode
        dones: [B] - done flags for each episode
        gamma: discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: [B] - computed advantages
        returns: [B] - computed returns (advantages + values)
    """
    device = values.device
    
    # Ensure all inputs are 1D tensors with same length (batch size)
    if rewards.dim() != 1 or values.dim() != 1 or dones.dim() != 1:
        raise ValueError(f"Expected 1D tensors for batch processing. Got shapes: rewards={rewards.shape}, values={values.shape}, dones={dones.shape}")
    
    batch_size = rewards.size(0)
    if values.size(0) != batch_size or dones.size(0) != batch_size:
        raise ValueError(f"Batch size mismatch: rewards={rewards.size(0)}, values={values.size(0)}, dones={dones.size(0)}")
    
    # Handle last_value - ensure it matches batch size
    if isinstance(last_value, (int, float)):
        last_value = torch.full((batch_size,), last_value, dtype=values.dtype, device=device)
    elif last_value.dim() == 0:
        last_value = last_value.unsqueeze(0).expand(batch_size)
    elif last_value.numel() == 1:
        last_value = last_value.expand(batch_size)
    elif last_value.size(0) != batch_size:
        raise ValueError(f"last_value batch size {last_value.size(0)} doesn't match expected {batch_size}")
    
    # For batch of episodes (not sequential timesteps), GAE simplifies to:
    # delta = reward + gamma * next_value * (1 - done) - current_value
    # advantage = delta (since there's no temporal sequence within each episode)
    # return = advantage + current_value
    
    # Compute TD error (delta)
    next_values = last_value  # For episode endings, next value is the terminal value
    deltas = rewards + gamma * next_values * (1.0 - dones.float()) - values
    
    # For single-step episodes, advantages equal deltas
    advantages = deltas
    
    # Returns are advantages + current values
    returns = advantages + values
    
    # Ensure proper dtype and device
    advantages = advantages.to(dtype=torch.float32, device=device)
    returns = returns.to(dtype=torch.float32, device=device)
    
    # Validate outputs for NaN/inf values
    if torch.isnan(advantages).any() or torch.isinf(advantages).any():
        print("🚨 WARNING: NaN/inf detected in advantages computation")
        print(f"  rewards: {rewards}")
        print(f"  values: {values}")
        print(f"  last_value: {last_value}")
        print(f"  dones: {dones}")
        print(f"  deltas: {deltas}")
        print(f"  gamma: {gamma}, lam: {lam}")
        # Replace problematic values with zeros
        advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages), 
                               torch.zeros_like(advantages), advantages)
        
    if torch.isnan(returns).any() or torch.isinf(returns).any():
        print("🚨 WARNING: NaN/inf detected in returns computation")
        print(f"  returns before fix: {returns}")
        returns = torch.where(torch.isnan(returns) | torch.isinf(returns), 
                            values, returns)  # Fallback to original values
        print(f"  returns after fix: {returns}")
    
    return advantages, returns


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
            
            # Compute entropy: -sum(p * log(p)) over spatial dimension, then mean over batch
            batch_entropy = -(normalized_probs * normalized_log_probs).sum(dim=-1)  # [batch_size]
            placement_entropy = batch_entropy.mean()  # Average over batch dimension
            
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
            
            # Compute entropy per sample, then mean over batch
            batch_entropy = -(edge_probs * edge_log_probs).sum(dim=-1)  # [batch_size]
            edge_entropy = batch_entropy.mean()  # Average over batch dimension
            
            # Check if entropy computation resulted in NaN
            if torch.isnan(edge_entropy):
                edge_entropy = torch.tensor(0.0, device=edge_logits.device)
                
        except RuntimeError as re:
            logging.error(re)
            edge_entropy = torch.tensor(0.0, device=edge_logits.device)

    if torch.isnan(army_logits).any():
        army_entropy = torch.tensor(0.0, device=army_logits.device)
    else:
        try:
            # Reshape to [batch_size * num_edges, num_army_options] for categorical distribution
            batch_size, num_edges, num_army_options = army_logits.shape
            army_logits_reshaped = army_logits.view(-1, num_army_options)
            
            # Create categorical distribution from logits and compute entropy
            categorical = torch.distributions.Categorical(logits=army_logits_reshaped)
            entropy_per_edge = categorical.entropy()  # [batch_size * num_edges]
            
            # Reshape back to [batch_size, num_edges] and average
            entropy_per_sample = entropy_per_edge.view(batch_size, num_edges).mean(dim=-1)  # [batch_size]
            army_entropy = entropy_per_sample.mean()  # Average over batch dimension
            
            # Check if entropy computation resulted in NaN
            if torch.isnan(army_entropy):
                army_entropy = torch.tensor(0.0, device=army_logits.device)
                
        except RuntimeError as re:
            logging.error(re)
            army_entropy = torch.tensor(0.0, device=army_logits.device)
    
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
    attacks: torch.Tensor,            # [batch_size, max_attacks, 4] (src, tgt, used_armies, available_armies)
    attack_logits: torch.Tensor,      # [batch_size, num_edges] - PADDED
    army_logits: torch.Tensor,        # [batch_size, num_edges, n_army_options] - PADDED  
    placements: torch.Tensor,         # [batch_size, max_placements]
    placement_logits: torch.Tensor,   # [batch_size, num_nodes]
    action_edges: torch.Tensor        # [batch_size, num_edges, 2] - PADDED with (-1,-1)
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
            attack_logits = attack_logits.unsqueeze(0)  # [1, num_edges]
        if army_logits.dim() == 2:
            army_logits = army_logits.unsqueeze(0)  # [1, num_edges, n_army_options]
        if placements.dim() == 1:
            placements = placements.unsqueeze(0)  # [1, max_placements]
        if attacks.dim() == 2:
            attacks = attacks.unsqueeze(0)  # [1, max_attacks, 3]
        if action_edges.dim() == 2:
            action_edges = action_edges.unsqueeze(0)  # [1, num_edges, 2]
            
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
        action_edges_flat = action_edges[:, :, 0] * num_nodes + action_edges[:, :, 1]  # [batch_size, num_edges]
        attacks_flat = attacks[:, :, 0] * num_nodes + attacks[:, :, 1]  # [batch_size, max_attacks]
        
        # Find matching edges using broadcasting
        edge_matches = (attacks_flat.unsqueeze(2) == action_edges_flat.unsqueeze(1))  # [batch_size, max_attacks, num_edges]
        edge_indices = edge_matches.to(torch.long).argmax(dim=2)  # Convert to long, then argmax

        # Mask valid attacks and edges
        attack_mask = (attacks[:, :, 0] >= 0)
        valid_match_mask = edge_matches.any(dim=2) & attack_mask  # [batch_size, max_attacks]

        # --- Individual attack log-probs (edge + army) ---
        edge_log_probs = f.log_softmax(attack_logits, dim=-1)  # [batch_size, num_edges]

        # Gather edge and army log-probs
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        edge_lp = edge_log_probs[batch_idx, edge_indices] * valid_match_mask.float()  # [batch_size, max_attacks]

        # Army log-probs - UPDATED: Handle percentage-based army selection system
        army_logits_selected = army_logits[batch_idx, edge_indices]  # [batch_size, max_attacks, n_army_options]
        army_log_probs = f.log_softmax(army_logits_selected, dim=-1)

        # NEW PERCENTAGE SYSTEM: The army logits represent percentage choices
        # Choice i (0-indexed) means using (i+1)/n_army_options * 100% of available armies
        # attacks tensor now contains: [src, tgt, used_armies, available_armies]
        used_armies = attacks[:, :, 2]  # [batch_size, max_attacks]
        available_armies = attacks[:, :, 3]  # [batch_size, max_attacks]
        
        # IMPORTANT: Multiple logits can map to the same army count!
        # We need to sum probabilities for all logits that lead to the same army count
        n_army_options = army_log_probs.size(2)  # Get number of army options from logits shape
        
        # Convert log probabilities to probabilities for summing
        army_probs = torch.exp(army_log_probs)  # [batch_size, max_attacks, n_army_options]
        
        # For each attack, calculate which army count each logit choice leads to
        batch_size, max_attacks, _ = army_probs.shape
        army_lp_list = []
        
        for b in range(batch_size):
            attack_lps = []
            for a in range(max_attacks):
                if valid_match_mask[b, a]:  # Only process valid attacks
                    used = used_armies[b, a].item()
                    available = available_armies[b, a].item()
                    
                    # Calculate what army count each choice leads to
                    total_prob = 0.0
                    for choice_idx in range(n_army_options):
                        choice_army_count = round((choice_idx + 1) / n_army_options * available)
                        if choice_army_count == used:
                            total_prob += army_probs[b, a, choice_idx].item()
                    
                    # Convert back to log probability
                    attack_lp = torch.log(torch.clamp(torch.tensor(total_prob), min=1e-10))
                    attack_lps.append(attack_lp)
                else:
                    attack_lps.append(torch.tensor(0.0))  # Invalid attack
            army_lp_list.append(torch.stack(attack_lps))
        
        army_lp = torch.stack(army_lp_list).to(device=attack_logits.device)  # [batch_size, max_attacks]

        # Apply valid mask to army log-probs (zeros out padded attacks)
        army_lp = army_lp * valid_match_mask.float()

        attack_log_probs = edge_lp + army_lp  # [batch_size, max_attacks] - individual attack log probs

    # If input was single sample, squeeze batch dimension
    if is_single_sample:
        placement_log_probs = placement_log_probs.squeeze(0)
        attack_log_probs = attack_log_probs.squeeze(0)

    return placement_log_probs, attack_log_probs

