import datetime
import logging
import os
import torch
import torch.nn.functional as f
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.game.Phase import Phase
from src.agents.RLUtils.RLUtils import RewardNormalizer, RolloutBuffer, StatTracker, compute_entropy, compute_gae, compute_log_probs, compute_individual_log_probs, load_checkpoint
from src.agents.RLUtils.PPOVerification import PPOVerifier
from src.config.training_config import VerificationConfig


class PPOAgent:
    def __init__(
            self,
            policy,
            optimizer: torch.optim.Adam,
            gamma=0.95,
            lam=0.95,
            clip_eps=0.30,
            ppo_epochs=1,
            enable_verification=False,  # Legacy parameter for backwards compatibility
            adaptive_epochs=True,  # Enable adaptive epoch adjustment
            verification_config=None,  # New parameter for granular verification control
            gradient_clip_norm=0.1,  # Gradient clipping norm
    ):
        self.policy: WarlightPolicyNet = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.adaptive_epochs = adaptive_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.reward_normalizer = RewardNormalizer()
        self.adv_tracker = StatTracker()
        self.loss_tracker = StatTracker()
        self.act_loss_tracker = StatTracker()
        self.crit_loss_tracker = StatTracker()
        self.ratio_tracker = StatTracker()
        self.placement_entropy_tracker = StatTracker()
        self.edge_entropy_tracker = StatTracker()
        self.army_entropy_tracker = StatTracker()
        self.value_tracker = StatTracker()
        self.value_pred_tracker = StatTracker()
        self.returns_tracker = StatTracker()
        
        # Initialize verification system
        self.verifier = PPOVerifier(verification_config=verification_config)
        
        # For weight change tracking
        self.prev_weights = None
        
        # For adaptive epochs
        self.gradient_history = []
        self.current_adaptive_epochs = ppo_epochs
        
        if 1==0:
            load_checkpoint(self.policy, self.optimizer, "res/model/checkpoint.pt")
    
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
            suggested_epochs = 1
            reason = "large gradients (>50)"
        elif gradient_norm > 20:
            # Moderate gradients - standard epochs
            suggested_epochs = self.ppo_epochs
            reason = "moderate gradients (20-50)"
        elif gradient_norm < 1:
            # Very small gradients - might need more epochs or learning has stagnated
            if grad_cv < 0.1:  # Stable small gradients
                suggested_epochs = min(self.ppo_epochs + 1, 4)  # Increase epochs slightly
                reason = "small stable gradients (<1)"
            else:
                suggested_epochs = self.ppo_epochs  # Unstable small gradients
                reason = "small unstable gradients (<1)"
        else:
            # Healthy gradients (1-20)
            if grad_cv < 0.2:  # Stable
                suggested_epochs = self.ppo_epochs
                reason = "healthy stable gradients (1-20)"
            else:  # Unstable
                suggested_epochs = max(self.ppo_epochs - 1, 1)  # Reduce epochs for stability
                reason = "healthy but unstable gradients (1-20)"
        
        # Smooth transitions - don't change epochs too rapidly
        if abs(suggested_epochs - self.current_adaptive_epochs) > 1:
            if suggested_epochs > self.current_adaptive_epochs:
                self.current_adaptive_epochs += 1
            else:
                self.current_adaptive_epochs -= 1
        else:
            self.current_adaptive_epochs = suggested_epochs
        
        if self.verifier.enabled:
            print(f"\n=== ADAPTIVE EPOCHS ===")
            print(f"Gradient norm: {gradient_norm:.4f}")
            print(f"Gradient CV: {grad_cv:.4f}")
            print(f"Suggested epochs: {suggested_epochs} ({reason})")
            print(f"Actual epochs: {self.current_adaptive_epochs}")
        
        # Log for tensorboard
        agent.total_rewards['adaptive_epochs'] = self.current_adaptive_epochs
        agent.total_rewards['gradient_cv'] = grad_cv
        
        return self.current_adaptive_epochs

    def update(self, buffer: RolloutBuffer, last_value, agent):            
        rewards_tensor = buffer.get_rewards()
        self.reward_normalizer.update(rewards_tensor)
        normalized_rewards_tensor = self.reward_normalizer.normalize(rewards_tensor)

        advantages, returns = compute_gae(
            normalized_rewards_tensor,
            buffer.get_values(),
            last_value,
            buffer.get_dones(),
            gamma=self.gamma,
            lam=self.lam
        )
        self.value_tracker.log(buffer.get_values().mean().item())
        agent.total_rewards['normalized_reward'] = normalized_rewards_tensor.mean().item()

        self.adv_tracker.log(advantages.mean().item())
        if agent.game_number > 1:
            std = self.adv_tracker.std()
            advantages = (advantages - self.adv_tracker.mean()) / (std + 1e-6)
        clipped_returns = torch.clamp(returns, -75, 75)
        self.returns_tracker.log(returns.mean().item())
        returns = clipped_returns

        # Determine number of epochs to use (adaptive or fixed)
        initial_grad_norm = None
        
        for epoch in range(self.ppo_epochs):  # Initial run to get gradient norm
            # CRITICAL FIX: Ensure model is in eval mode for consistent inference
            # Training mode can cause different behavior (dropout, batch norm) leading to inconsistent logits
            was_training = agent.model.training
            agent.model.eval()
            
            # Get properly batched inputs - no reshaping needed!
            starting_features_batched = buffer.get_starting_node_features()  # [batch_size, num_nodes, features]
            post_features_batched = buffer.get_post_placement_node_features()  # [batch_size, num_nodes, features]
            action_edges_batched = buffer.get_edges()  # [batch_size, 42, 2]
            
            # Optional verification of input structure
            self.verifier.verify_structural_integrity(
                starting_features_batched, post_features_batched, action_edges_batched, epoch
            )
            
            # Run model in batch mode
            placement_logits, _, _ = agent.run_model(
                node_features=starting_features_batched, 
                action_edges=action_edges_batched, 
                action=Phase.PLACE_ARMIES
            )

            _, attack_logits, army_logits = agent.run_model(
                node_features=post_features_batched,
                action_edges=action_edges_batched,
                action=Phase.ATTACK_TRANSFER
            )

            # Optional verification of model outputs
            self.verifier.verify_model_outputs(placement_logits, attack_logits, army_logits)
            
            # Apply the same masking that was used during action selection
            from src.agents.RLUtils.RLUtils import apply_placement_masking
            owned_regions_list = buffer.get_owned_regions()
            placement_logits = apply_placement_masking(placement_logits, owned_regions_list)
            
            # Check for problematic all-inf samples and fix them
            all_inf_mask = torch.isinf(placement_logits).all(dim=-1)
            if all_inf_mask.any():
                placement_logits[all_inf_mask, 0] = 0.0

            # Optional verification of single vs batch inference consistency
            self.verifier.verify_single_vs_batch_inference(
                agent, starting_features_batched, post_features_batched, 
                action_edges_batched, placement_logits, attack_logits, army_logits, buffer
            )

            # Optional verification of buffer data integrity
            self.verifier.verify_buffer_data_integrity(
                starting_features_batched, post_features_batched, action_edges_batched
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
            
            # Calculate per-action ratios
            placement_diff = new_placement_log_probs - old_placement_log_probs if new_placement_log_probs.numel() > 0 and old_placement_log_probs.numel() > 0 else torch.tensor([])
            attack_diff = new_attack_log_probs - old_attack_log_probs if new_attack_log_probs.numel() > 0 and old_attack_log_probs.numel() > 0 else torch.tensor([])
            
            # Ensure tensors have matching shapes by padding to the maximum size
            if new_placement_log_probs.numel() > 0 and old_placement_log_probs.numel() > 0:
                max_placement_size = max(new_placement_log_probs.size(1), old_placement_log_probs.size(1))
                
                # Safety check for NaN before padding
                if torch.isnan(new_placement_log_probs).any():
                    new_placement_log_probs = torch.where(torch.isnan(new_placement_log_probs), 
                                                        torch.tensor(-1e9, device=new_placement_log_probs.device), 
                                                        new_placement_log_probs)
                
                if new_placement_log_probs.size(1) < max_placement_size:
                    padding = torch.zeros(new_placement_log_probs.size(0), 
                                        max_placement_size - new_placement_log_probs.size(1), 
                                        device=new_placement_log_probs.device)
                    new_placement_log_probs = torch.cat([new_placement_log_probs, padding], dim=1)
                elif new_placement_log_probs.size(1) > max_placement_size:
                    new_placement_log_probs = new_placement_log_probs[:, :max_placement_size]
                    
                if old_placement_log_probs.size(1) < max_placement_size:
                    padding = torch.zeros(old_placement_log_probs.size(0), 
                                        max_placement_size - old_placement_log_probs.size(1), 
                                        device=old_placement_log_probs.device)
                    old_placement_log_probs = torch.cat([old_placement_log_probs, padding], dim=1)
                elif old_placement_log_probs.size(1) > max_placement_size:
                    old_placement_log_probs = old_placement_log_probs[:, :max_placement_size]
            
            if new_attack_log_probs.numel() > 0 and old_attack_log_probs.numel() > 0:
                max_attack_size = max(new_attack_log_probs.size(1), old_attack_log_probs.size(1))
                if new_attack_log_probs.size(1) < max_attack_size:
                    padding = torch.zeros(new_attack_log_probs.size(0), 
                                        max_attack_size - new_attack_log_probs.size(1), 
                                        device=new_attack_log_probs.device)
                    new_attack_log_probs = torch.cat([new_attack_log_probs, padding], dim=1)
                if old_attack_log_probs.size(1) < max_attack_size:
                    padding = torch.zeros(old_attack_log_probs.size(0), 
                                        max_attack_size - old_attack_log_probs.size(1), 
                                        device=old_attack_log_probs.device)
                    old_attack_log_probs = torch.cat([old_attack_log_probs, padding], dim=1)
            # Calculate per-action ratios
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
                    placement_avg_ratio = torch.ones(len(advantages), device=new_placement_log_probs.device if new_placement_log_probs.numel() > 0 else 'cpu')
            else:
                placement_avg_ratio = torch.ones(len(advantages), device=new_attack_log_probs.device if new_attack_log_probs.numel() > 0 else 'cpu')
            
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
                    
                    # Only compute ratios for episodes that actually have attacks
                    attack_avg_ratio = torch.ones(len(advantages), device=attack_ratios.device)
                    episodes_with_attacks_indices = episodes_with_attacks.nonzero(as_tuple=True)[0]
                    
                    if len(episodes_with_attacks_indices) > 0:
                        attack_ratios_for_episodes = valid_ratios[episodes_with_attacks_indices].sum(dim=1) / valid_count[episodes_with_attacks_indices]
                        attack_avg_ratio[episodes_with_attacks_indices] = attack_ratios_for_episodes
                else:
                    # No episodes have attacks - use neutral ratio
                    attack_avg_ratio = torch.ones(len(advantages), device=attack_ratios.device)
            else:
                attack_avg_ratio = torch.ones(len(advantages), device=new_attack_log_probs.device if new_attack_log_probs.numel() > 0 else 'cpu')
            
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

            # Restore model training mode for gradient computation
            if was_training:
                agent.model.train()
            
            agent.total_rewards['new_log_probs_mean'] = total_new_log_probs.mean().item()
            agent.total_rewards['old_log_probs_mean'] = total_old_log_probs.mean().item()
            agent.total_rewards['log_prob_diff_mean'] = (total_new_log_probs - total_old_log_probs).mean().item()
            agent.total_rewards['log_prob_diff_std'] = (total_new_log_probs - total_old_log_probs).std().item()
            agent.total_rewards['ppo_ratio'] = ratio.mean().item()

            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            ).mean()
            if torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any():
                raise RuntimeError(f'policy_loss inf!: {ratio}, {advantages}')
            self.ratio_tracker.log(ratio.mean().item())
            values_pred = self.policy.get_value(starting_features_batched)
            value_loss = f.mse_loss(values_pred, returns)
            self.value_pred_tracker.log(values_pred.mean().item())
            
            # Fix placement_logits for entropy computation
            placement_logits_for_entropy = placement_logits.clone()
            all_inf_mask = torch.isinf(placement_logits_for_entropy).all(dim=-1)
            if all_inf_mask.any():
                placement_logits_for_entropy[all_inf_mask, 0] = 0.0
            
            placement_entropy, edge_entropy, army_entropy = compute_entropy(placement_logits_for_entropy, attack_logits, army_logits)

            entropy = placement_entropy + edge_entropy + army_entropy
            if isinstance(placement_entropy, torch.Tensor):
                self.placement_entropy_tracker.log(placement_entropy.item())
            else:
                self.placement_entropy_tracker.log(placement_entropy)

            if isinstance(edge_entropy, torch.Tensor):
                self.edge_entropy_tracker.log(edge_entropy.item())
            else:
                self.edge_entropy_tracker.log(edge_entropy)

            if isinstance(army_entropy, torch.Tensor):
                self.army_entropy_tracker.log(army_entropy.item())
            else:
                self.army_entropy_tracker.log(army_entropy)

            self.act_loss_tracker.log(policy_loss.item())
            self.crit_loss_tracker.log(value_loss.item())

            entropy_factor = 0.5 - (agent.game_number / 15000) * 0.3
            loss = policy_loss + 0.5 * value_loss \
                   - entropy_factor * (0.1 * placement_entropy + 0.1 * edge_entropy + 0.003 * army_entropy)
            self.loss_tracker.log(loss.mean().item())

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print('something went wrong with loss')
                return

            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients before clipping
            has_nan_grad = False
            for name, p in self.policy.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break
                    
            if has_nan_grad:
                return
            
            # Enhanced gradient analysis
            grad_stats = self.verifier.analyze_gradients(self.policy, agent)
            current_grad_norm = grad_stats.get('total_norm', 0) if grad_stats else 0
            
            # On first epoch, determine adaptive epoch count
            if epoch == 0 and self.adaptive_epochs:
                adaptive_epochs = self.adjust_epochs_based_on_gradients(current_grad_norm, agent)
                if adaptive_epochs != self.ppo_epochs:
                    print(f"🔄 Adjusting epochs from {self.ppo_epochs} to {adaptive_epochs} based on gradient analysis")
                # Update the loop limit for remaining epochs
                remaining_epochs = adaptive_epochs - 1  # -1 because we're already in epoch 0
            else:
                remaining_epochs = self.ppo_epochs - epoch - 1
            
            # Weight change analysis (compare with previous weights)
            self.verifier.analyze_weight_changes(self.policy, self.prev_weights, agent)
            
            # Store current weights for next iteration comparison
            if self.verifier.enabled:
                self.prev_weights = {name: param.data.clone() for name, param in self.policy.named_parameters()}

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.gradient_clip_norm)
            self.optimizer.step()
            
            # Break early if we've done enough epochs (for adaptive case)
            if self.adaptive_epochs and epoch >= self.current_adaptive_epochs - 1:
                break
            if os.path.exists("res/model/") and agent.game_number % 100 == 0:
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                torch.save(
                    {
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    f"res/model/checkpoint_{ts}_{agent.game_number}.pt",
                )
