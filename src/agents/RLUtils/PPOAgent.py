import datetime
import logging
import os
import torch
import torch.nn.functional as f
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.game.Phase import Phase
from src.agents.RLUtils.RLUtils import RewardNormalizer, RolloutBuffer, StatTracker, compute_entropy, compute_gae, compute_log_probs, load_checkpoint


class PPOAgent:
    def __init__(
            self,
            policy,
            optimizer: torch.optim.Adam,
            gamma=0.95,
            lam=0.95,
            clip_eps=0.30,
            ppo_epochs=6,
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

            log_probs = compute_log_probs(  # Use vectorized version
                buffer.get_attacks(),
                attack_logits,
                army_logits,
                buffer.get_placements(),
                placement_logits,
                buffer.get_edges(),
            )
            diff = torch.clamp(log_probs - old_log_probs, -10, 10)
            if type(diff) == torch.Tensor:
                ratio = diff.exp()
            else:
                ratio = torch.tensor(diff, device=log_probs.device).exp()

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

            entropy = 0.01 * placement_entropy + edge_entropy + army_entropy
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

            entropy_factor = 0.02 - (agent.game_number / 15000) * 0.01
            loss = policy_loss + 0.5 * value_loss - entropy_factor * entropy
            self.loss_tracker.log(loss.mean().item())

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print('something went wrong with loss')
                return

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
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
