import torch
import torch.nn.functional as f
from typing import Optional, TYPE_CHECKING
from src.agents.RLUtils import WarlightModelAutoregressiveTransformer
from src.config.training_config import PPOConfig, VerificationConfig
from src.game.Phase import Phase
from src.agents.RLUtils.RLUtils import RolloutBuffer, compute_entropy, compute_gae, compute_individual_log_probs
from src.agents.RLUtils.PPOVerification import PPOVerifier
from src.agents.RLUtils.PopArt import PopArt
import time

if TYPE_CHECKING:
    from src.agents.RLUtils.CheckpointManager import CheckpointManager


class PPOAgent:
    def __init__(
            self,
            policy: WarlightModelAutoregressiveTransformer.WarlightPolicy,
            optimizer: torch.optim.Optimizer,
            ppo_config: PPOConfig,
            verification_config: VerificationConfig):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = ppo_config.gamma
        self.lam = ppo_config.lam
        self.clip_eps = ppo_config.clip_eps
        self.ppo_epochs = ppo_config.ppo_epochs
        self.gradient_clip_norm = ppo_config.gradient_clip_norm
        self.value_loss_coeff = ppo_config.value_loss_coeff
        self.value_clip_range = ppo_config.value_clip_range
        self.kl_threshold = ppo_config.kl_threshold
        self.monitor_gradient_norm = ppo_config.monitor_gradient_norm
        self.normalize_gradients = ppo_config.normalize_gradients
        self.target_grad_norm = ppo_config.target_grad_norm
        self._grad_norm_epsilon = 1e-8

        # Entropy configuration
        self.entropy_coeff_start = ppo_config.entropy_coeff_start
        self.entropy_coeff_decay = ppo_config.entropy_coeff_decay
        self.entropy_decay_episodes = ppo_config.entropy_decay_episodes
        self.placement_entropy_coeff = ppo_config.placement_entropy_coeff
        self.edge_entropy_coeff = ppo_config.edge_entropy_coeff
        self.army_entropy_coeff = ppo_config.army_entropy_coeff
        
        self.popart = PopArt(self.policy.value_head)

        # Initialize verification system
        self.verifier = PPOVerifier(verification_config=verification_config)
        
        # Initialize checkpoint manager (will be set by agent)
        self.checkpoint_manager: Optional['CheckpointManager'] = None
        
        # For weight change tracking
        self.prev_weights = None
        
        # Verbose losses configuration (enabled for analysis)
        self.verbose_losses = True

    def _pad_log_prob_tensors_to_match(self, new_tensor, old_tensor, tensor_name=""):
        """
        Ensure two log probability tensors have matching shapes by padding to maximum size.
        
        Args:
            new_tensor: New log probabilities tensor [batch_size, num_actions]
            old_tensor: Old log probabilities tensor [batch_size, num_actions]  
            tensor_name: Name for debugging purposes
            
        Returns:
            Tuple of (padded_new_tensor, padded_old_tensor) with matching shapes
        """
        if new_tensor.numel() == 0 or old_tensor.numel() == 0:
            return new_tensor, old_tensor
            
        max_size = max(new_tensor.size(1), old_tensor.size(1))
        
        # Safety check for NaN in new tensor before padding
        if torch.isnan(new_tensor).any():
            new_tensor = torch.where(torch.isnan(new_tensor), 
                                   torch.tensor(-1e9, device=new_tensor.device), 
                                   new_tensor)
        
        # Pad or truncate new tensor
        if new_tensor.size(1) < max_size:
            padding = torch.zeros(new_tensor.size(0), 
                                max_size - new_tensor.size(1), 
                                device=new_tensor.device)
            new_tensor = torch.cat([new_tensor, padding], dim=1)
        elif new_tensor.size(1) > max_size:
            new_tensor = new_tensor[:, :max_size]
            
        # Pad or truncate old tensor
        if old_tensor.size(1) < max_size:
            padding = torch.zeros(old_tensor.size(0), 
                                max_size - old_tensor.size(1), 
                                device=old_tensor.device)
            old_tensor = torch.cat([old_tensor, padding], dim=1)
        elif old_tensor.size(1) > max_size:
            old_tensor = old_tensor[:, :max_size]
            
        return new_tensor, old_tensor

    def _align_vector_length(self, tensor: torch.Tensor, target_length: int, device: torch.device,
                              fill_value: float = 1.0, tensor_name: str = "") -> torch.Tensor:
        """
        Ensure a 1D tensor matches the desired length by padding or trimming.

        Args:
            tensor: Input tensor expected to be 1D.
            target_length: Desired length of the output tensor.
            device: Target device for the resulting tensor.
            fill_value: Value used for padding when tensor is shorter than target_length.
            tensor_name: Optional name for debugging/logging.

        Returns:
            Tensor with shape [target_length].
        """
        if target_length <= 0:
            return torch.empty((0,), device=device, dtype=tensor.dtype if tensor.numel() > 0 else torch.float32)

        if tensor is None or tensor.numel() == 0:
            return torch.full((target_length,), fill_value, device=device,
                              dtype=tensor.dtype if tensor is not None and tensor.numel() > 0 else torch.float32)

        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)
        current_length = tensor.size(0)

        if current_length == target_length:
            return tensor

        if current_length < target_length:
            pad_shape = (target_length - current_length,)
            padding = torch.full(pad_shape, fill_value, device=device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)

        # current_length > target_length: trim extra entries
        return tensor[:target_length]
    
    def _compute_kl_divergence(self, old_log_probs, new_log_probs):
        """
        Compute mean KL-divergence between old and new log probabilities.
        Args:
            old_log_probs: [batch_size, num_actions]
            new_log_probs: [batch_size, num_actions]
        Returns:
            Mean KL-divergence (scalar)
        """
        # Convert log probs to probs
        old_probs = torch.exp(old_log_probs)
        new_probs = torch.exp(new_log_probs)
        # Avoid log(0) and division by zero
        eps = 1e-8
        kl = old_probs * (old_log_probs - new_log_probs)
        kl = kl.sum(dim=1)  # Sum over actions
        return kl.mean().item()

    @staticmethod
    def _compute_grad_norm(parameters, norm_type: float = 2.0) -> float:
        total = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(norm_type)
            total += param_norm.item() ** norm_type
        if total == 0.0:
            return 0.0
        return total ** (1.0 / norm_type)

    def update(self, buffer: RolloutBuffer, last_value, agent):
        ppo_update_start_time = time.time()
        epoch_times = []

        torch.autograd.set_detect_anomaly(False)

        if buffer.size() == 0:
            return

        device = next(self.policy.parameters()).device

        # DIAGNOSTIC: Check buffer consistency before processing
        buffer_diagnostics = {
            'starting_node_features': len(buffer.starting_node_features_list),
            'starting_edge_features': len(buffer.starting_edge_features),
            'end_features': len(buffer.end_features_list),
            'end_edge_features': len(buffer.end_edge_features),
            'rewards': len(buffer.rewards),
            'values': len(buffer.values),
            'dones': len(buffer.dones),
            'placements': len(buffer.placements),
            'attacks': len(buffer.attacks),
            'placement_log_probs': len(buffer.placement_log_probs),
            'attack_log_probs': len(buffer.attack_log_probs),
            'ownership_masks': len(buffer.ownership_masks),
            'armies_left': len(buffer.armies_left),
        }
        
        # Check for inconsistencies
        sizes = set(buffer_diagnostics.values())
        if len(sizes) > 1:
            print(f"\n[PPO] ERROR: Buffer size mismatch detected!")
            for key, size in sorted(buffer_diagnostics.items()):
                print(f"  {key}: {size}")
            print()

        rewards = buffer.get_rewards().to(device)
        values = buffer.get_values().to(device)
        dones = buffer.get_dones().to(device)

        # Validate rewards, values, dones have consistent lengths
        if not (rewards.shape[0] == values.shape[0] == dones.shape[0]):
            raise RuntimeError(
                f"Buffer consistency error: rewards ({rewards.shape[0]}), values ({values.shape[0]}), "
                f"and dones ({dones.shape[0]}) must have the same length"
            )

        if isinstance(last_value, torch.Tensor):
            last_value_tensor = last_value.detach().to(device=device, dtype=values.dtype)
        else:
            last_value_tensor = torch.tensor(last_value, device=device, dtype=values.dtype)

        advantages, returns = compute_gae(rewards, values, last_value_tensor, dones, gamma=self.gamma, lam=self.lam)
        advantages = advantages.to(device)
        returns = returns.to(device)

        if advantages.numel() > 1 and torch.var(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_mean = self.popart.mean
        old_std = self.popart.std
        self.popart.update(returns.detach())
        self.popart.adjust_weights(old_mean, old_std)
        normalized_returns = self.popart.normalize(returns).detach()

        batch_size = advantages.shape[0]

        starting_node_features = buffer.get_starting_node_features().to(device)
        starting_edge_features = buffer.get_starting_edge_features().to(device)
        end_node_features = buffer.get_end_features().to(device)
        end_edge_features = buffer.get_end_edge_features().to(device)
        
        # Validate feature dimensions match batch_size
        if starting_node_features.shape[0] != batch_size:
            raise RuntimeError(
                f"Buffer consistency error: starting_node_features has {starting_node_features.shape[0]} entries "
                f"but batch_size (from rewards) is {batch_size}"
            )
        if starting_edge_features.shape[0] != batch_size:
            raise RuntimeError(
                f"Buffer consistency error: starting_edge_features has {starting_edge_features.shape[0]} entries "
                f"but batch_size is {batch_size}"
            )
        if end_node_features.shape[0] != batch_size:
            raise RuntimeError(
                f"Buffer consistency error: end_node_features has {end_node_features.shape[0]} entries "
                f"but batch_size is {batch_size}"
            )
        if end_edge_features.shape[0] != batch_size:
            raise RuntimeError(
                f"Buffer consistency error: end_edge_features has {end_edge_features.shape[0]} entries "
                f"but batch_size is {batch_size}"
            )

        ownership_mask = buffer.get_ownership_mask()
        if ownership_mask is None:
            ownership_mask = torch.ones((batch_size, starting_node_features.size(1)), device=device)
        else:
            ownership_mask = ownership_mask.to(device)
            if ownership_mask.dim() == 1 and batch_size > 1:
                ownership_mask = ownership_mask.unsqueeze(0).expand(batch_size, -1)
            # Validate ownership mask matches batch_size
            if ownership_mask.shape[0] != batch_size:
                raise RuntimeError(
                    f"Buffer consistency error: ownership_mask has {ownership_mask.shape[0]} entries "
                    f"but batch_size is {batch_size}"
                )

        armies_left = buffer.get_armies_left()
        if armies_left.numel() == 0:
            armies_left_tensor = torch.zeros((batch_size, starting_node_features.size(1)), device=device, dtype=starting_node_features.dtype)
        else:
            armies_left_tensor = armies_left.to(device=device, dtype=starting_node_features.dtype)
            armies_left_tensor = torch.where(armies_left_tensor < 0, torch.zeros(1, device=device, dtype=armies_left_tensor.dtype), armies_left_tensor)
            # Validate armies_left matches batch_size
            if armies_left_tensor.shape[0] != batch_size:
                raise RuntimeError(
                    f"Buffer consistency error: armies_left_tensor has {armies_left_tensor.shape[0]} entries "
                    f"but batch_size is {batch_size}"
                )

        placements_tensor = buffer.get_placements()
        if placements_tensor.numel() == 0:
            placements_list = [[] for _ in range(batch_size)]
        else:
            placements_cpu = placements_tensor.cpu()
            # Trim placements_tensor if needed
            if placements_cpu.shape[0] > batch_size:
                print(f"[PPO] WARNING: Trimming placements from {placements_cpu.shape[0]} to {batch_size}")
                placements_cpu = placements_cpu[:batch_size]
            placements_list = []
            for row in placements_cpu:
                valid = row[row >= 0].tolist()
                placements_list.append(valid)

        attacks_data = buffer.get_attacks()
        # Trim attacks_data if it's a list and too long
        if isinstance(attacks_data, list) and len(attacks_data) > batch_size:
            print(f"[PPO] WARNING: Trimming attacks_data from {len(attacks_data)} to {batch_size}")
            attacks_data = attacks_data[:batch_size]
        
        if batch_size == 1:
            if isinstance(attacks_data, list):
                attacks_for_policy = attacks_data[0] if len(attacks_data) > 0 else {}
            else:
                attacks_for_policy = attacks_data if isinstance(attacks_data, dict) else {}
        else:
            if isinstance(attacks_data, list):
                attacks_for_policy = attacks_data
            elif isinstance(attacks_data, dict):
                attacks_for_policy = [attacks_data for _ in range(batch_size)]
            else:
                attacks_for_policy = [{} for _ in range(batch_size)]

        old_placement_log_probs = buffer.get_placement_log_probs()
        old_attack_log_probs = buffer.get_attack_log_probs()
        
        # Validate old log probs match batch_size
        if old_placement_log_probs.numel() > 0 and old_placement_log_probs.shape[0] != batch_size:
            raise RuntimeError(
                f"Buffer consistency error: old_placement_log_probs has {old_placement_log_probs.shape[0]} entries "
                f"but batch_size is {batch_size}"
            )
        if old_attack_log_probs.numel() > 0 and old_attack_log_probs.shape[0] != batch_size:
            raise RuntimeError(
                f"Buffer consistency error: old_attack_log_probs has {old_attack_log_probs.shape[0]} entries "
                f"but batch_size is {batch_size}"
            )

        if old_placement_log_probs.requires_grad:
            raise RuntimeError(
                "Expected placement log probabilities from buffer to be detached before PPO update. "
                "Ensure action selection runs under torch.no_grad() and stored tensors are cloned/detached."
            )
        if old_attack_log_probs.requires_grad:
            raise RuntimeError(
                "Expected attack log probabilities from buffer to be detached before PPO update. "
                "Ensure action selection runs under torch.no_grad() and stored tensors are cloned/detached."
            )

        def _sum_log_probs(log_prob_tensor: torch.Tensor, target_batch: int) -> torch.Tensor:
            if not isinstance(log_prob_tensor, torch.Tensor) or log_prob_tensor.numel() == 0:
                return torch.zeros(target_batch, device=device, dtype=returns.dtype)
            log_prob_tensor = log_prob_tensor.to(device=device, dtype=returns.dtype)
            if log_prob_tensor.dim() == 0:
                return log_prob_tensor.reshape(1).expand(target_batch)
            if log_prob_tensor.dim() == 1:
                if log_prob_tensor.size(0) == target_batch:
                    return log_prob_tensor
                return torch.zeros(target_batch, device=device, dtype=returns.dtype)
            return log_prob_tensor.sum(dim=1)

        joint_logp_old = _sum_log_probs(old_placement_log_probs, batch_size) + _sum_log_probs(old_attack_log_probs, batch_size)

        self.policy.train()

        # Track KL divergence for early stopping
        kl_divergences = []
        early_stop_triggered = False

        for epoch in range(self.ppo_epochs):
            epoch_start_time = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            # CRITICAL FIX: Use eval mode for forward pass to eliminate dropout noise
            # Gradients still flow because we're not in torch.no_grad()
            # This ensures deterministic policy evaluation for stable training
            self.policy.eval()

            recompute = self.policy.recompute_turn_logprobs(
                starting_node_features,
                starting_edge_features,
                ownership_mask,
                placements_list,
                attacks_for_policy,
                armies_left_tensor,
                action=None,
            )

            new_place_logps = recompute.get("logp", torch.empty((batch_size, 0), device=device, dtype=returns.dtype))
            if isinstance(new_place_logps, torch.Tensor):
                new_place_logps = new_place_logps.to(device=device, dtype=returns.dtype)
            else:
                new_place_logps = torch.empty((batch_size, 0), device=device, dtype=returns.dtype)

            new_attack_logps = recompute.get("attack_logp", torch.empty((batch_size, 0), device=device, dtype=returns.dtype))
            if isinstance(new_attack_logps, torch.Tensor):
                new_attack_logps = new_attack_logps.to(device=device, dtype=returns.dtype)
            else:
                new_attack_logps = torch.empty((batch_size, 0), device=device, dtype=returns.dtype)

            joint_logp_new = _sum_log_probs(new_place_logps, batch_size) + _sum_log_probs(new_attack_logps, batch_size)

            logp_diff = torch.clamp(joint_logp_new - joint_logp_old, min=-20.0, max=20.0)
            ratio = torch.exp(logp_diff)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            policy_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()

            value_pred = self.policy.get_value(end_node_features, end_edge_features).to(device=device, dtype=returns.dtype)
            normalized_value_pred = self.popart.normalize(value_pred)
            value_loss = f.mse_loss(normalized_value_pred, normalized_returns)

            placement_entropy = torch.tensor(recompute.get("placement_entropy", 0.0), device=device, dtype=returns.dtype)
            attack_entropy = torch.tensor(recompute.get("attack_entropy", 0.0), device=device, dtype=returns.dtype)

            # Calculate weighted entropy components
            weighted_placement_entropy = self.placement_entropy_coeff * placement_entropy
            weighted_attack_entropy = self.edge_entropy_coeff * attack_entropy

            entropy_loss = (
                weighted_placement_entropy
                + weighted_attack_entropy
            )

            total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff_start * entropy_loss

            # Switch to train mode for backward pass (allows dropout in gradient computation if needed)
            self.policy.train()
            total_loss.backward()

            parameters = [p for p in self.policy.parameters() if p.grad is not None]
            grad_norm_raw = 0.0
            grad_norm_post_scale = 0.0
            grad_scale = 1.0
            clipped_norm = 0.0

            if parameters:
                if self.monitor_gradient_norm or self.normalize_gradients:
                    grad_norm_raw = self._compute_grad_norm(parameters)
                    grad_norm_post_scale = grad_norm_raw

                if self.normalize_gradients and grad_norm_raw > self._grad_norm_epsilon:
                    grad_scale = self.target_grad_norm / (grad_norm_raw + self._grad_norm_epsilon)
                    for p in parameters:
                        p.grad.mul_(grad_scale)
                    grad_norm_post_scale = grad_norm_raw * grad_scale

                if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                    clipped_tensor_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.gradient_clip_norm)
                    clipped_norm = float(clipped_tensor_norm)
                else:
                    clipped_norm = grad_norm_post_scale

            self.optimizer.step()

            # Compute KL divergence AFTER the optimizer step to measure actual policy change
            # Use the same forward pass results (already in eval mode, no dropout)
            with torch.no_grad():
                self.policy.eval()

                recompute_after_update = self.policy.recompute_turn_logprobs(
                    starting_node_features,
                    starting_edge_features,
                    ownership_mask,
                    placements_list,
                    attacks_for_policy,
                    armies_left_tensor,
                    action=None,
                )

                new_place_logps_after = recompute_after_update.get("logp", torch.empty((batch_size, 0), device=device, dtype=returns.dtype))
                if isinstance(new_place_logps_after, torch.Tensor):
                    new_place_logps_after = new_place_logps_after.to(device=device, dtype=returns.dtype)
                else:
                    new_place_logps_after = torch.empty((batch_size, 0), device=device, dtype=returns.dtype)

                new_attack_logps_after = recompute_after_update.get("attack_logp", torch.empty((batch_size, 0), device=device, dtype=returns.dtype))
                if isinstance(new_attack_logps_after, torch.Tensor):
                    new_attack_logps_after = new_attack_logps_after.to(device=device, dtype=returns.dtype)
                else:
                    new_attack_logps_after = torch.empty((batch_size, 0), device=device, dtype=returns.dtype)

                joint_logp_after_update = _sum_log_probs(new_place_logps_after, batch_size) + _sum_log_probs(new_attack_logps_after, batch_size)

                # KL divergence: measures change from original policy to updated policy
                kl_div = (joint_logp_old - joint_logp_after_update).mean().item()
                kl_divergences.append(kl_div)

            # Timing for this epoch
            epoch_duration = time.time() - epoch_start_time
            epoch_times.append(epoch_duration)
            print(f"[PPO] Epoch {epoch+1}/{self.ppo_epochs} took {epoch_duration:.4f} seconds (avg per step: {epoch_duration/batch_size:.6f} s), KL div: {kl_div:.6f}")

            # Log losses and stats per epoch
            if hasattr(agent, "total_rewards"):
                agent.total_rewards["policy_loss"] = policy_loss.item()
                agent.total_rewards["value_loss"] = value_loss.item()
                agent.total_rewards["total_loss"] = total_loss.item()
                agent.total_rewards["ppo_ratio"] = ratio.mean().item()
                agent.total_rewards["kl_divergence"] = kl_div

                # Log entropy components and their weighted versions
                placement_entropy_val = placement_entropy.item()
                attack_entropy_val = attack_entropy.item()

                agent.total_rewards["placement_entropy"] = placement_entropy_val
                agent.total_rewards["attack_entropy"] = attack_entropy_val
                agent.total_rewards["weighted_placement_entropy"] = weighted_placement_entropy.item()
                agent.total_rewards["weighted_attack_entropy"] = weighted_attack_entropy.item()
                agent.total_rewards["total_entropy_loss"] = entropy_loss.item()

                # Log entropy coefficients for reference
                agent.total_rewards["placement_entropy_coeff"] = self.placement_entropy_coeff
                agent.total_rewards["edge_entropy_coeff"] = self.edge_entropy_coeff

                if self.monitor_gradient_norm:
                    agent.total_rewards["grad_norm_raw"] = grad_norm_raw
                    agent.total_rewards["grad_norm_post_scale"] = grad_norm_post_scale
                    agent.total_rewards["grad_norm_clipped"] = clipped_norm
                    agent.total_rewards["grad_scale_factor"] = grad_scale

            # Check if KL divergence exceeds threshold AFTER logging
            if kl_div > self.kl_threshold:
                print(f"[PPO] Early stopping at epoch {epoch+1}: KL divergence {kl_div:.6f} exceeds threshold {self.kl_threshold}")
                early_stop_triggered = True
                # Log the early stopping event
                if hasattr(agent, "total_rewards"):
                    agent.total_rewards["early_stop_epoch"] = epoch + 1
                    agent.total_rewards["final_kl_divergence"] = kl_div
                break

        # Log summary statistics about KL divergence and early stopping
        if hasattr(agent, "total_rewards"):
            agent.total_rewards["mean_kl_divergence"] = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
            agent.total_rewards["max_kl_divergence"] = max(kl_divergences) if kl_divergences else 0.0
            agent.total_rewards["epochs_completed"] = len(kl_divergences)
            agent.total_rewards["early_stopped"] = early_stop_triggered

        # PPO update timing
        total_ppo_update_time = time.time() - ppo_update_start_time
        print(f"[PPO] Total PPO update took {total_ppo_update_time:.4f} seconds (avg per step: {total_ppo_update_time/batch_size:.6f} s)")
        if early_stop_triggered:
            print(f"[PPO] Early stopping summary: Completed {len(kl_divergences)} epochs, mean KL: {sum(kl_divergences) / len(kl_divergences):.6f}")
