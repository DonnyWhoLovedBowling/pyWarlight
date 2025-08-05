import datetime
import logging
import os
import torch
import torch.nn.functional as f
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.game.Phase import Phase
from src.agents.RLUtils.RLUtils import RewardNormalizer, RolloutBuffer, StatTracker, compute_entropy, compute_gae, compute_log_probs, compute_individual_log_probs, load_checkpoint


class PPOAgent:
    def __init__(
            self,
            policy,
            optimizer: torch.optim.Adam,
            gamma=0.95,
            lam=0.95,
            clip_eps=0.30,
            ppo_epochs=1,
    ):
        self.policy: WarlightPolicyNet = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
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
        if 1==0:
            load_checkpoint(self.policy, self.optimizer, "res/model/checkpoint.pt")

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

        old_log_probs = buffer.get_log_probs()

        for _ in range(self.ppo_epochs):
            # Get properly batched inputs - no reshaping needed!
            starting_features_batched = buffer.get_starting_node_features()  # [batch_size, num_nodes, features]
            post_features_batched = buffer.get_post_placement_node_features()  # [batch_size, num_nodes, features]
            action_edges_batched = buffer.get_edges()  # [batch_size, 42, 2]
            
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

            # Get individual log probabilities for each action
            new_placement_log_probs, new_attack_log_probs = compute_individual_log_probs(
                buffer.get_attacks(),
                attack_logits,
                army_logits,
                buffer.get_placements(),
                placement_logits,
                buffer.get_edges(),
            )
            
            # Get old individual log probabilities
            old_placement_log_probs = buffer.get_placement_log_probs()
            old_attack_log_probs = buffer.get_attack_log_probs()
            
            # Calculate per-action ratios
            placement_diff = new_placement_log_probs - old_placement_log_probs
            attack_diff = new_attack_log_probs - old_attack_log_probs
            
            # Clamp extreme differences to prevent numerical instability
            placement_diff = torch.clamp(placement_diff, -5, 5)
            attack_diff = torch.clamp(attack_diff, -5, 5)
            
            # Calculate per-action ratios
            placement_ratios = placement_diff.exp()
            attack_ratios = attack_diff.exp()
            
            # Compute importance-weighted average ratio across all actions for policy loss
            # Weight by the magnitude of log probs (actions with higher magnitude have more influence)
            placement_weights = torch.abs(old_placement_log_probs)
            attack_weights = torch.abs(old_attack_log_probs)
            
            # Calculate weighted average ratio per episode
            placement_ratio_weighted = (placement_ratios * placement_weights).sum(dim=1) / (placement_weights.sum(dim=1) + 1e-8)
            attack_ratio_weighted = (attack_ratios * attack_weights).sum(dim=1) / (attack_weights.sum(dim=1) + 1e-8)
            
            # Combine placement and attack ratios (geometric mean to balance both)
            ratio = torch.sqrt(placement_ratio_weighted * attack_ratio_weighted)
            ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
            
            # For logging, compute total log prob differences for comparison
            total_new_log_probs = new_placement_log_probs.sum(dim=1) + new_attack_log_probs.sum(dim=1)
            total_old_log_probs = old_placement_log_probs.sum(dim=1) + old_attack_log_probs.sum(dim=1)
            
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
            placement_entropy, edge_entropy, army_entropy = compute_entropy(placement_logits, attack_logits, army_logits)

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
            # Add gradient diagnostics
            total_norm = 0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm **= 1. / 2

            agent.total_rewards['gradient_norm'] = total_norm

            # Much more aggressive clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.1)  # Much smaller
            self.optimizer.step()
            if os.path.exists("res/model/"):
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                torch.save(
                    {
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    f"res/model/checkpoint_{ts}.pt",
                )
