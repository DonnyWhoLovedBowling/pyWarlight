import datetime
import logging
import os
import torch
import torch.nn.functional as f
from typing import Optional, TYPE_CHECKING
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.game.Phase import Phase
from src.agents.RLUtils.RLUtils import RewardNormalizer, RolloutBuffer, StatTracker, compute_entropy, compute_gae, compute_individual_log_probs
from src.agents.RLUtils.PPOVerification import PPOVerifier
from src.config.training_config import VerificationConfig
from src.agents.RLUtils.RLUtils import apply_placement_masking
from src.agents.RLUtils.PopArt import PopArt

if TYPE_CHECKING:
    from src.agents.RLUtils.CheckpointManager import CheckpointManager


class PPOAgent:
    def __init__(
            self,
            policy,
            placement_optimizer: torch.optim.Optimizer,
            edge_optimizer: torch.optim.Optimizer,
            army_optimizer: torch.optim.Optimizer,
            value_optimizer: torch.optim.Optimizer,
            gamma=0.95,
            lam=0.95,
            clip_eps=0.30,
            ppo_epochs=1,
            enable_verification=False,  # Legacy parameter for backwards compatibility
            adaptive_epochs=True,  # Enable adaptive epoch adjustment
            verification_config=None,  # New parameter for granular verification control
            gradient_clip_norm=0.1,  # Gradient clipping norm
            value_loss_coeff=0.5,  # Value loss coefficient
            value_clip_range=None,  # Value clipping range (None = no clipping)
            verbose_losses=False,  # Print detailed loss information
            # Entropy configuration parameters
            entropy_coeff_start=0.5,  # Initial entropy coefficient
            entropy_coeff_decay=0.3,  # How much entropy decays
            entropy_decay_episodes=15000,  # Episodes over which to decay
            placement_entropy_coeff=0.1,  # Placement entropy coefficient
            edge_entropy_coeff=0.1,  # Edge entropy coefficient
            army_entropy_coeff=0.03,  # Army entropy coefficient
    ):
        self.policy: WarlightPolicyNet = policy
        self.placement_optimizer = placement_optimizer
        self.edge_optimizer = edge_optimizer
        self.army_optimizer = army_optimizer
        self.value_optimizer = value_optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.adaptive_epochs = adaptive_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.value_loss_coeff = value_loss_coeff
        self.value_clip_range = value_clip_range
        self.verbose_losses = verbose_losses
        
        # Entropy configuration
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_decay = entropy_coeff_decay
        self.entropy_decay_episodes = entropy_decay_episodes
        self.placement_entropy_coeff = placement_entropy_coeff
        self.edge_entropy_coeff = edge_entropy_coeff
        self.army_entropy_coeff = army_entropy_coeff
        
        self.popart = PopArt(self.policy.value_head)
        self.adv_tracker = StatTracker()
        self.loss_tracker = StatTracker()
        self.act_loss_tracker = StatTracker()
        self.crit_loss_tracker = StatTracker()
        self.ratio_tracker = StatTracker()
        self.placement_entropy_tracker = StatTracker()
        self.edge_entropy_tracker = StatTracker()
        self.army_entropy_tracker = StatTracker()
        self.entropy_loss_tracker = StatTracker()
        self.value_tracker = StatTracker()
        self.value_pred_tracker = StatTracker()
        self.returns_tracker = StatTracker()
        
        # Initialize verification system
        self.verifier = PPOVerifier(verification_config=verification_config)
        
        # Initialize checkpoint manager (will be set by agent)
        self.checkpoint_manager: Optional['CheckpointManager'] = None
        
        # For weight change tracking
        self.prev_weights = None
        
        # For adaptive epochs
        self.gradient_history = []
        self.current_adaptive_epochs = ppo_epochs
        

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
    
    def adjust_epochs_based_on_gradients(self, gradient_norm, agent):
        """
        Dynamically adjust the number of PPO epochs based on gradient magnitude and training stability
        """
        if not self.adaptive_epochs:
            return self.ppo_epochs
            
        # Store gradient history
        self.gradient_history.append(gradient_norm)
        if len(self.gradient_history) > 10:  # Keep last 10 updates
            self.gradient_history.pop(0)
        
        # Calculate gradient stability (coefficient of variation)
        if len(self.gradient_history) >= 3:
            grad_mean = sum(self.gradient_history) / len(self.gradient_history)
            grad_std = (sum((g - grad_mean) ** 2 for g in self.gradient_history) / len(self.gradient_history)) ** 0.5
            grad_cv = grad_std / (grad_mean + 1e-8)  # Coefficient of variation
        else:
            grad_cv = 1.0  # High uncertainty with few samples
        
        # Adaptive epoch logic
        if gradient_norm > 50:
            # Very large gradients - use fewer epochs to avoid instability
            self.ppo_epochs = 1
            reason = "large gradients (>50)"
        elif gradient_norm > 20:
            # Moderate gradients - standard epochs
            reason = "moderate gradients (20-50)"
        elif gradient_norm < 1:
            # Very small gradients - might need more epochs or learning has stagnated
            if grad_cv < 1:  # Stable small gradients
                self.ppo_epochs = self.current_adaptive_epochs + 2
                reason = "small stable gradients (<1)"
            else:
                reason = "small unstable gradients (<1)"
        else:
            # Healthy gradients (1-20)
            if grad_cv < 2:  # Stable
                self.ppo_epochs = self.current_adaptive_epochs + 1
                reason = "healthy stable gradients (1-20)"
            else:  # Unstable
                self.ppo_epochs = max(self.ppo_epochs - 1, 1)  # Reduce epochs for stability
                reason = "healthy but unstable gradients (1-20)"
        
        # Smooth transitions - don't change epochs too rapidly
        if self.verifier.enabled:
            print(f"\n=== ADAPTIVE EPOCHS ===")
            print(f"Gradient norm: {gradient_norm:.4f}")
            print(f"Gradient CV: {grad_cv:.4f}")
            print(f"Suggested epochs: {self.ppo_epochs} ({reason})")
            print(f"Actual epochs: {self.current_adaptive_epochs}")
        # Log for tensorboard
        agent.total_rewards['adaptive_epochs'] = self.current_adaptive_epochs
        agent.total_rewards['gradient_cv'] = grad_cv
        
        return self.ppo_epochs

    def update(self, buffer: RolloutBuffer, last_value, agent):            
        # Remove normalization of rewards before GAE
        rewards_tensor = buffer.get_rewards()
        device = rewards_tensor.device
        # 1. Compute returns and advantages from raw rewards and values
        advantages, returns = compute_gae(
            rewards_tensor,
            buffer.get_values(),
            last_value,
            buffer.get_dones(),
            gamma=self.gamma,
            lam=self.lam
        )
        returns = torch.clamp(returns, -75, 75)
        raw_advantages = advantages.clone()  # Store raw advantages for verification
        # 4. Normalize returns for value loss (PopArt)
        old_mean, old_std = self.popart.mean, self.popart.std
        self.popart.update(returns)
        if self.popart.mean != old_mean or self.popart.std != old_std:
            self.popart.adjust_weights(old_mean, old_std)

        # Optional verification of GAE computation
        self.verifier.verify_gae_computation(
            rewards_tensor, buffer.get_values(), last_value,
            buffer.get_dones(), raw_advantages, returns, self.gamma, self.lam
        )
        # Now normalize advantages for policy loss
        if advantages.numel() > 1 and advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.zeros_like(advantages)

        # Determine number of epochs to use (adaptive or fixed)
        initial_grad_norm = None
        starting_features_batched = buffer.get_starting_node_features()  # [batch_size, num_nodes, features]
        post_features_batched = buffer.get_post_placement_node_features()  # [batch_size, num_nodes, features]
        action_edges_batched = buffer.get_edges()  # [batch_size, num_edges, 2]
        starting_edge_features_batched = buffer.get_starting_edge_features()
        post_edge_features_batched = buffer.get_post_placement_edge_features()
        end_edge_features_batched = buffer.get_end_edge_features()
        for epoch in range(self.ppo_epochs + 2):  # + 2 to allow for adaptive adjustment

            # Get properly batched inputs - no reshaping needed!

            # Optional verification of input structure
            self.verifier.verify_structural_integrity(
                starting_features_batched, post_features_batched, action_edges_batched, epoch
            )
            
            # Run model in batch mode
            placement_logits, _, _, _ = agent.run_model(
                node_features=starting_features_batched,
                edge_features=starting_edge_features_batched,
                action_edges=action_edges_batched, 
                action=Phase.PLACE_ARMIES
            )

            _, attack_logits, army_logits, _ = agent.run_model(
                node_features=post_features_batched,
                action_edges=action_edges_batched,
                action=Phase.ATTACK_TRANSFER,
                edge_features=post_edge_features_batched,
            )

            # Optional verification of model outputs
            self.verifier.verify_model_outputs(placement_logits, attack_logits, army_logits)
            
            # Optional verification of single vs batch inference consistency (before masking)
            self.verifier.verify_single_vs_batch_inference(
                agent, starting_features_batched, post_features_batched, 
                action_edges_batched, placement_logits, attack_logits, army_logits, buffer, epoch
            )
            
            # Apply the same masking that was used during action selection
            owned_regions_list = buffer.get_owned_regions()
            placement_logits = apply_placement_masking(placement_logits, owned_regions_list)
            
            # Check for problematic all-inf samples and fix them
            all_inf_mask = torch.isinf(placement_logits).all(dim=-1)
            if all_inf_mask.any():
                placement_logits = placement_logits.clone()
                placement_logits[all_inf_mask, 0] = 0.0

            # Optional verification of buffer data integrity
            self.verifier.verify_buffer_data_integrity(
                starting_features_batched, post_features_batched, action_edges_batched, epoch
            )

            # Get individual log probabilities for each action
            new_placement_log_probs, new_attack_log_probs = compute_individual_log_probs(
                buffer.get_attacks(),
                attack_logits,
                army_logits,
                buffer.get_placements(),
                placement_logits,
                buffer.get_edges(),
            )
            
            # Optional verification of action data integrity
            self.verifier.verify_action_data(
                buffer.get_attacks(), buffer.get_placements(), buffer.get_edges(),
                new_placement_log_probs, new_attack_log_probs
            )
            
            # Get old individual log probabilities
            old_placement_log_probs = buffer.get_placement_log_probs()
            old_attack_log_probs = buffer.get_attack_log_probs()
            
            # Optional verification of old log probabilities
            self.verifier.verify_old_log_probs(old_placement_log_probs, old_attack_log_probs)

            # Ensure tensors have matching shapes by padding to the maximum size
            new_placement_log_probs, old_placement_log_probs = self._pad_log_prob_tensors_to_match(
                new_placement_log_probs, old_placement_log_probs, "placement"
            )
            new_attack_log_probs, old_attack_log_probs = self._pad_log_prob_tensors_to_match(
                new_attack_log_probs, old_attack_log_probs, "attack"
            )
            
            # Recalculate per-action ratios after padding
            placement_diff = new_placement_log_probs - old_placement_log_probs if new_placement_log_probs.numel() > 0 and old_placement_log_probs.numel() > 0 else torch.tensor([])
            attack_diff = new_attack_log_probs - old_attack_log_probs if new_attack_log_probs.numel() > 0 and old_attack_log_probs.numel() > 0 else torch.tensor([])
            
            # Optional check for extreme attack differences
            self.verifier.check_extreme_attack_differences(attack_diff)
            
            # Clamp individual action differences to prevent extreme ratios (temporarily more aggressive)
            if placement_diff.numel() > 0:
                placement_diff = torch.clamp(placement_diff, -10, 10)  # Back to less aggressive clamping
                placement_ratios = placement_diff.exp()
            else:
                placement_ratios = torch.tensor([])
                
            if attack_diff.numel() > 0:
                attack_diff = torch.clamp(attack_diff, -10, 10)  # Back to less aggressive clamping
                attack_ratios = attack_diff.exp()
            else:
                attack_ratios = torch.tensor([])
            
            # Simple approach: average ratios across all actions per episode
            eps = 1e-8
            
            # Count valid (non-zero) actions for proper averaging
            if placement_ratios.numel() > 0:
                placement_mask = (old_placement_log_probs != 0.0)
                if placement_mask.any():
                    valid_placement_ratios = placement_ratios * placement_mask.float()
                    placement_count = placement_mask.sum(dim=1).float() + eps
                    placement_avg_ratio = valid_placement_ratios.sum(dim=1) / placement_count
                else:
                    placement_avg_ratio = torch.ones(len(advantages), device=device)
            else:
                placement_avg_ratio = torch.ones(len(advantages), device=device)
            
            if attack_ratios.numel() > 0:
                # Better approach: Use old_attack_log_probs to identify episodes with real attacks
                # Real attack log probs should be negative (since they're log probabilities)
                # Padding entries are exactly 0.0
                valid_attack_mask = (old_attack_log_probs < -1e-6)  # Real log probs are negative
                episodes_with_attacks = valid_attack_mask.any(dim=1)  # [batch_size] - which episodes have any attacks
                
                if episodes_with_attacks.any():
                    # For episodes with attacks, compute average ratio
                    valid_ratios = attack_ratios * valid_attack_mask.float()
                    valid_count = valid_attack_mask.sum(dim=1).float()
                    
                    # Only compute rcheckatios for episodes that actually have attacks
                    attack_avg_ratio = torch.ones(len(advantages), device=device)
                    episodes_with_attacks_indices = episodes_with_attacks.nonzero(as_tuple=True)[0]
                    
                    if len(episodes_with_attacks_indices) > 0:
                        attack_ratios_for_episodes = valid_ratios[episodes_with_attacks_indices].sum(dim=1) / valid_count[episodes_with_attacks_indices]
                        attack_avg_ratio[episodes_with_attacks_indices] = attack_ratios_for_episodes
                else:
                    # No episodes have attacks - use neutral ratio
                    attack_avg_ratio = torch.ones(len(advantages), device=device)
            else:
                attack_avg_ratio = torch.ones(len(advantages), device=device)
            
            # Combine ratios with equal weighting
            ratio = (placement_avg_ratio + attack_avg_ratio) / 2.0
            ratio = torch.clamp(ratio, 0.1, 10.0)
            
            # For logging, compute total log prob differences
            if new_placement_log_probs.numel() > 0 and new_attack_log_probs.numel() > 0:
                total_new_log_probs = new_placement_log_probs.sum(dim=1) + new_attack_log_probs.sum(dim=1)
            elif new_placement_log_probs.numel() > 0:
                total_new_log_probs = new_placement_log_probs.sum(dim=1)
            elif new_attack_log_probs.numel() > 0:
                total_new_log_probs = new_attack_log_probs.sum(dim=1)
            else:
                total_new_log_probs = torch.zeros(len(advantages))
                
            if old_placement_log_probs.numel() > 0 and old_attack_log_probs.numel() > 0:
                total_old_log_probs = old_placement_log_probs.sum(dim=1) + old_attack_log_probs.sum(dim=1)
            elif old_placement_log_probs.numel() > 0:
                total_old_log_probs = old_placement_log_probs.sum(dim=1)
            elif old_attack_log_probs.numel() > 0:
                total_old_log_probs = old_attack_log_probs.sum(dim=1)
            else:
                total_old_log_probs = torch.zeros(len(advantages))
            
            # Optional check for extreme log probability differences
            self.verifier.check_extreme_log_prob_differences(total_new_log_probs, total_old_log_probs)
            
            # Check for NaN/inf values in ratios
            if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                ratio = torch.ones_like(ratio)
            
            # Optional verification of action distribution
            self.verifier.analyze_action_distribution(placement_logits, attack_logits, army_logits, agent)

            # Note: Model remains in training mode throughout PPO update for consistency
            # with action selection behavior (no mode switching needed)
            
            agent.total_rewards['new_log_probs_mean'] = total_new_log_probs.mean().item()
            agent.total_rewards['old_log_probs_mean'] = total_old_log_probs.mean().item()
            agent.total_rewards['log_prob_diff_mean'] = (total_new_log_probs - total_old_log_probs).mean().item()
            agent.total_rewards['log_prob_diff_std'] = (total_new_log_probs - total_old_log_probs).std().item()
            agent.total_rewards['ppo_ratio'] = ratio.mean().item()
            agent.total_rewards['advantages'] = advantages.mean().item()

            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            ).mean()
            if torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any():
                raise RuntimeError(f'policy_loss inf!: {ratio}, {advantages}')
            self.ratio_tracker.log(ratio.mean().item())
            
            # Keep model in training mode for consistent value computation
            # (no mode switching to maintain consistency throughout)
            values_pred = self.policy.get_value(buffer.get_end_features(), end_edge_features_batched, buffer.get_edges())

            # --- PopArt normalization and weight adjustment ---
            old_mean, old_std = self.popart.mean, self.popart.std
            self.popart.update(returns)
            # If stats changed, adjust value head weights/bias
            if self.popart.mean != old_mean or self.popart.std != old_std:
                self.popart.adjust_weights(old_mean, old_std)
            # Normalize returns for value loss
            normalized_returns = self.popart.normalize(returns)

            # Value loss with optional clipping for additional regularization
            if self.value_clip_range is not None:
                old_values = buffer.get_values()
                values_clipped = old_values + torch.clamp(
                    values_pred - old_values, -self.value_clip_range, self.value_clip_range
                )
                value_loss_unclipped = f.mse_loss(values_pred, normalized_returns)
                value_loss_clipped = f.mse_loss(values_clipped, normalized_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
            else:
                value_loss = f.mse_loss(values_pred, normalized_returns)

            self.value_pred_tracker.log(values_pred.mean().item())
            agent.total_rewards['values_pred'] = values_pred.mean().item()
            agent.total_rewards['normalized_returns'] = normalized_returns.mean().item()

            # Fix placement_logits for entropy computation
            placement_logits_for_entropy = placement_logits.clone()
            all_inf_mask = torch.isinf(placement_logits_for_entropy).all(dim=-1)
            if all_inf_mask.any():
                placement_logits_for_entropy = placement_logits_for_entropy.clone()
                placement_logits_for_entropy[all_inf_mask, 0] = 0.0
            
            placement_entropy, edge_entropy, army_entropy = compute_entropy(placement_logits_for_entropy, attack_logits, army_logits)

            if isinstance(placement_entropy, torch.Tensor):
                self.placement_entropy_tracker.log(placement_entropy.mean().item())
                agent.total_rewards['placement_entropy'] = placement_entropy.mean().item()

            else:
                self.placement_entropy_tracker.log(placement_entropy)

            if isinstance(edge_entropy, torch.Tensor):
                self.edge_entropy_tracker.log(edge_entropy.mean().item())
                agent.total_rewards['edge_entropy'] = edge_entropy.mean().item()

            else:
                self.edge_entropy_tracker.log(edge_entropy)

            if isinstance(army_entropy, torch.Tensor):
                self.army_entropy_tracker.log(army_entropy.mean().item())
                agent.total_rewards['army_entropy'] = army_entropy.mean().item()

            else:
                self.army_entropy_tracker.log(army_entropy)

            self.act_loss_tracker.log(policy_loss.mean().item())
            self.crit_loss_tracker.log(value_loss.mean().item())

            agent.total_rewards['policy_loss'] = policy_loss.mean().item()
            agent.total_rewards['value_loss'] = value_loss.mean().item()

            # Use configurable entropy coefficients and schedule
            entropy_factor = self.entropy_coeff_start - (agent.game_number / self.entropy_decay_episodes) * self.entropy_coeff_decay
            entropy_factor = max(entropy_factor, 0.01)  # Minimum entropy to maintain exploration
            
            # --- Dynamic entropy factor based on moving average of losses ---
            # Initialize moving averages if not present
            if not hasattr(self, 'policy_loss_ma'):
                self.policy_loss_ma = policy_loss.mean().item()
            if not hasattr(self, 'value_loss_ma'):
                self.value_loss_ma = value_loss.mean().item()
            # Update moving averages (exponential moving average)
            ma_alpha = 0.01  # Smoothing factor (can be tuned)
            self.policy_loss_ma = (1 - ma_alpha) * self.policy_loss_ma + ma_alpha * policy_loss.item()
            self.value_loss_ma = (1 - ma_alpha) * self.value_loss_ma + ma_alpha * value_loss.item()
            # Thresholds for boosting entropy (can be tuned)
            policy_loss_thresh = 2.0
            value_loss_thresh = 2.0
            min_entropy_boost = 0.2  # Minimum entropy factor if instability detected
            # Compute scheduled entropy factor
            scheduled_entropy = self.entropy_coeff_start - (agent.game_number / self.entropy_decay_episodes) * self.entropy_coeff_decay
            scheduled_entropy = max(scheduled_entropy, 0.01)
            # Dynamically boost entropy if losses are high
            if self.policy_loss_ma > policy_loss_thresh or self.value_loss_ma > value_loss_thresh:
                entropy_factor = max(scheduled_entropy, min_entropy_boost)
                if self.verbose_losses:
                    print(f"⚡ Boosting entropy factor due to high moving average losses: policy_ma={self.policy_loss_ma:.4f}, value_ma={self.value_loss_ma:.4f}")
            else:
                entropy_factor = scheduled_entropy
            
            entropy_loss = (self.placement_entropy_coeff * placement_entropy + 
                          self.edge_entropy_coeff * edge_entropy + 
                          self.army_entropy_coeff * army_entropy)
            self.entropy_loss_tracker.log(entropy_loss.mean().item())
            agent.total_rewards['entropy_loss'] = entropy_loss.mean().item()
            # Both policy_loss and value_loss should be minimized (i.e., both positive, same sign)
            loss = policy_loss + self.value_loss_coeff * value_loss - entropy_factor * entropy_loss
            self.loss_tracker.log(loss.mean().item())
            agent.total_rewards['total_loss'] = loss.mean().item()
            
            # Print detailed loss information if verbose_losses is enabled
            if self.verbose_losses:
                print(f"📊 LOSS VALUES - Game {agent.game_number}, Epoch {epoch+1}")
                print(f"   Policy Loss: {policy_loss.item():.4f}")
                print(f"   Value Loss:  {value_loss.item():.4f}")
                print(f"   Entropy Loss: {entropy_loss.item():.4f}")
                print(f"   Total Loss:  {loss.mean().item():.4f}")
                print(f"   Entropy Factor: {entropy_factor:.4f}")

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print('something went wrong with loss')
                return

            loss.backward(retain_graph=True)
            # 3. Clip gradients and step for all optimizers

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.gradient_clip_norm)

            self.placement_optimizer.step()
            self.edge_optimizer.step()
            self.army_optimizer.step()
            self.value_optimizer.step()

            # Enhanced gradient analysis
            grad_stats = self.verifier.analyze_gradients(self.policy, agent)
            current_grad_norm = grad_stats.get('total_norm', 0) if grad_stats else 0

            # On first epoch, determine adaptive epoch count
            if epoch == 0 and self.adaptive_epochs:
                adaptive_epochs = self.adjust_epochs_based_on_gradients(current_grad_norm, agent)
                if adaptive_epochs != self.ppo_epochs:
                    print(f"🔄 Adjusting epochs from {self.ppo_epochs} to {adaptive_epochs} based on gradient analysis")
                # Update the loop limit for remaining epochs

            # Weight change analysis (compare with previous weights)
            self.verifier.analyze_weight_changes(self.policy, self.prev_weights, agent)

            # Store current weights for next iteration comparison
            if self.verifier.enabled:
                self.prev_weights = {name: param.data.clone() for name, param in self.policy.named_parameters()}

            # Gradient clipping
            # Ensure all model parameters and optimizer state are on the same device before step
            # Break early if we've done enough epochs (for adaptive case)
            if self.adaptive_epochs and epoch >= self.ppo_epochs - 1:
                break

        # Save checkpoint using CheckpointManager if available
        if self.checkpoint_manager and self.checkpoint_manager.should_save_checkpoint(agent.game_number):
            self.checkpoint_manager.save_checkpoint(agent, agent.game_number)
