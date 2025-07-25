import json
import logging
import random
import time
import sys
import os

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from io import TextIOWrapper

import numpy as np
from torch.utils.tensorboard import SummaryWriter


from src.game.FightSide import FightSide
from src.game.Phase import Phase

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override, TextIO

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv
from datetime import datetime

from src.engine.AgentBase import AgentBase

from src.game.Game import Game
from src.game.Region import Region

from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies

import faulthandler

do_hm_search = True


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

    def update(self, rewards):
        batch = torch.tensor(rewards, dtype=torch.float32)
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


class RolloutBuffer:
    def __init__(self):
        self.states: list[Game] = []
        self.edges = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, edges, action, log_prob, reward, value, done):
        self.states.append(state)
        self.edges.append(edges)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()


def load_checkpoint(policy, optimizer, path="checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✅ Loaded checkpoint from", path)
    else:
        print("ℹ️ No checkpoint found, starting fresh.")


def compute_gae(rewards, values, last_value, dones, gamma=0.95, lam=0.95):
    advantages = []
    gae = 0
    values = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages), torch.tensor(returns)


def compute_entropy(placement_logits, edge_logits, army_logits):
    if type(placement_logits) == int:
        return torch.Tensor(0), torch.Tensor(0), torch.Tensor(0)
    placement_probs = f.softmax(placement_logits, dim=-1)
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


def compute_log_probs(attacks, attack_logits, army_logits, placements, placement_logits, action_edges):
    """
    attacks: List of (src, tgt, armies)
    placements: Index or list of indices where armies were placed
    attack_logits: [num_edges]
    army_logits: [num_edges, max_army_send]
    placement_logits: [num_nodes]
    action_edges: [num_edges, 2]
    """
    # --- Placement log-prob ---
    if type(placements) == int or len(action_edges) == 0:
        return torch.tensor([0], dtype=torch.float)
    try:
        placement_log_probs = f.log_softmax(placement_logits, dim=-1)
    except AttributeError as ae:
        print(placement_logits)
        print(ae)
        raise ae

    if isinstance(placements, (list, tuple)) and len(placement_log_probs) > 0:
        placement_log_prob = torch.stack([
            placement_log_probs[p] for p in placements
        ]).sum()
    else:
        placement_log_prob = torch.tensor(0.0, device=attack_logits.device)

    # --- Attack log-probs ---
    if not attacks:
        attack_log_prob = torch.tensor(0.0, device=attack_logits.device)
    else:
        edge_log_probs = f.log_softmax(attack_logits, dim=0)
        attack_log_prob = 0.0

        for src, tgt, armies in attacks:
            # Find index of this edge in action_edges
            try:
                match = ((action_edges[:, 0] == src) & (action_edges[:, 1] == tgt)).nonzero(as_tuple=False)
            except IndexError as ie:
                print(src, tgt, armies)
                print(action_edges)
                raise ie

            if len(match) == 0:
                raise ValueError(f"Attack from {src} to {tgt} not in action_edges.")

            edge_idx = match.item()
            try:
                edge_lp = edge_log_probs[edge_idx]
            except IndexError as e:
                print(src, tgt, armies)
                print(edge_log_probs)
                print(edge_idx)
                print(action_edges)
                raise e

            try:
                army_log_probs = f.log_softmax(army_logits[edge_idx], dim=0)
            except IndexError as e:
                print(src, tgt, armies)
                print(army_logits, edge_idx)
                print(action_edges)

                raise e
            if armies > len(army_log_probs):
                raise ValueError(f"Army value {armies} exceeds army_logits range.")

            army_lp = army_log_probs[armies]
            if army_lp.sum() < -1e8:
                logging.warning(f"Army value {armies} is in masking range.")


            try:
                attack_log_prob += edge_lp + army_lp
            except TypeError as e:
                print(edge_log_probs)
                raise e

    return placement_log_prob + attack_log_prob


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
        self.policy = policy
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

        load_checkpoint(self.policy, self.optimizer, "res/model/checkpoint.pt")

    def update(self, buffer: RolloutBuffer, last_value, agent):
        self.reward_normalizer.update(buffer.rewards)
        normalized_rewards = self.reward_normalizer.normalize(buffer.rewards).tolist()

        advantages, returns = compute_gae(
            normalized_rewards,
            buffer.values,
            last_value,
            buffer.dones,
            gamma=self.gamma,
            lam=self.lam
        )
        self.value_tracker.log(buffer.values[0].item())
        agent: RLGNNAgent = agent
        agent.total_rewards['normalized_reward'] = normalized_rewards[0]

        self.adv_tracker.log(advantages[0].item())
        if agent.game_number > 1:
            std = self.adv_tracker.std()
            advantages = (advantages - self.adv_tracker.mean()) / (std + 1e-6)
        clipped_returns = torch.clamp(returns, -75, 75)
        self.returns_tracker.log(returns.item())
        returns = clipped_returns

        old_log_probs = buffer.log_probs[0]
        if buffer.states is None or len(buffer.states) == 0:
            return
        for _ in range(self.ppo_epochs):
            placement_logits, _attack_logits, _army_logits = agent.run_model(
                game=agent.starting_state, action_edges=agent.init_action_edges
            )
            _placement_logits, attack_logits, army_logits = agent.run_model(
                game=agent.post_placement_state, action_edges=agent.post_placement_edges
            )

            log_probs = compute_log_probs(
                agent.get_attacks(buffer.actions[0]),
                attack_logits,
                army_logits,
                agent.get_placements(buffer.actions[0]),
                placement_logits,
                agent.post_placement_edges,
            )
            diff = torch.clamp(log_probs - old_log_probs, -10, 10)
            if type(diff) == torch.Tensor:
                ratio = diff.exp()
            else:
                ratio = torch.Tensor(diff).exp()

            agent.total_rewards['ppo_ratio'] = ratio.item()

            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            ).mean()
            if torch.isnan(policy_loss).any() or torch.isinf(policy_loss).any():
                raise RuntimeError(f'policy_loss inf!: {ratio}, {advantages}')
            self.ratio_tracker.log(ratio.item())
            values_pred = self.policy.get_value(buffer.states[0])
            value_loss = f.mse_loss(values_pred, returns)
            self.value_pred_tracker.log(values_pred.item())
            placement_entropy, edge_entropy, army_entropy = compute_entropy(placement_logits, attack_logits,
                                                                            army_logits)
            lst = []
            lst.append(placement_entropy + edge_entropy + army_entropy)
            entropy = torch.stack(lst)
            entropy = entropy.mean()
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

            self.act_loss_tracker.log(value_loss.item())
            self.crit_loss_tracker.log(policy_loss.item())

            entropy_factor = 0.1 - (buffer.states[0].round / buffer.states[0].config.num_games) * 0.08
            loss = policy_loss + 0.5 * value_loss - entropy_factor * entropy
            self.loss_tracker.log(loss.item())

            if buffer.states[0].round % 10 == 0:
                logging.debug(f"loss: {loss}")

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print('something went wrong with loss')
                return

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            if os.path.exists("res/model/") and buffer.dones[0] and buffer.states[0].config.seed % 50 == 0:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                torch.save(
                    {
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    f"res/model/checkpoint_{ts}.pt",
                )
                # with open(f"model_weights_{ts}.txt", "w") as file:
                #     for name, param in agent.model.named_parameters():
                #         file.write(f"{name} - shape {param.shape}\n")
                #         file.write(f"{param.data}\n\n")


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
    def __init__(self, embed_dim, hidden_dim, max_army_send):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.army_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Linear(128, max_army_send)
        )
        self.max_army_send = max_army_send

    def forward(self, node_embeddings, action_edges, army_counts):
        src, tgt = action_edges[:, 0], action_edges[:, 1]
        edge_embed = torch.cat([node_embeddings[src], node_embeddings[tgt]], dim=-1)

        edge_logits = self.edge_scorer(edge_embed).squeeze(-1)  # [num_edges]
        army_logits = self.army_scorer(edge_embed)  # [num_edges, max_army_send]

        # ====== Soft discouragement for unlikely attacks ======
        src_armies = army_counts[src]
        tgt_armies = army_counts[tgt]

        bad_edges = (src_armies <= 2) | (tgt_armies >= 3 * src_armies)
        edge_logits = edge_logits - bad_edges.float() * 1.0  # subtract 1.0 as soft penalty

        invalid_self = src == tgt
        edge_logits[invalid_self] -= 100.0  # or -1e9 if you want hard mask

        # ====== Hard mask invalid army amounts per edge ======
        max_sendable = src_armies - 1
        army_mask = (
            torch.arange(self.max_army_send, device=army_logits.device)
            .unsqueeze(0)
        )  # [1, max_army_send]

        valid_mask = army_mask <= max_sendable.unsqueeze(1)  # [num_edges, max_army_send]
        army_logits[~valid_mask] = -1e9  # Mask out too-large moves

        return edge_logits, army_logits


class WarlightPolicyNet(nn.Module):
    def __init__(self, node_feat_dim, embed_dim=64, max_army_send=50):
        super().__init__()
        self.gnn1 = GCNConv(node_feat_dim, embed_dim)
        self.gnn2 = GCNConv(embed_dim, embed_dim)

        self.placement_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)  # One logit per node
        )

        self.attack_head = AttackHead(embed_dim, 64, max_army_send)

        # Value head: input is aggregated graph embedding
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.edge_tensor: torch.Tensor = None

    def get_value(self, game: Game):
        node_features = game.create_node_features()
        graph = torch.tensor(node_features, dtype=torch.float)
        edge_tensor = self.edge_tensor

        x = f.relu(self.gnn1(graph, edge_tensor))
        node_embeddings = self.gnn2(x, edge_tensor)

        # Mean pool over all nodes to get a graph-level embedding
        graph_embedding = node_embeddings.mean(dim=0)
        value = self.value_head(graph_embedding)
        return value.squeeze(-1)

    def forward(self, x, action_edges, army_counts, action='both'):
        """
        x: [num_nodes, node_feat_dim]       # node features
        edge_index: [2, num_edges]          # graph structure
        action_edges: [num_actions, 2]      # list of (src, tgt) edges for attacks
        army_counts: [num_nodes]            # current army count on each node
        """
        # GNN
        edge_index = self.edge_tensor
        x = f.relu(self.gnn1(x, edge_index))
        node_embeddings = self.gnn2(x, edge_index)
        placement_logits = torch.Tensor([])
        attack_logits = torch.Tensor([])
        army_logits = torch.Tensor([])

        if action in ['placement', 'both']:
            # Placement
            placement_logits = self.placement_head(node_embeddings).squeeze(
                -1
            )  # [num_nodes]

        if action in ['attack', 'both']:
            # Attack
            attack_logits, army_logits = self.attack_head(
                node_embeddings, action_edges, army_counts
            )

        return placement_logits, attack_logits, army_logits


@dataclass
class RLGNNAgent(AgentBase):
    in_channels = 8
    hidden_channels = 64
    model = WarlightPolicyNet(in_channels, hidden_channels)
    placement_logits = torch.tensor([])
    attack_logits = torch.tensor([])
    army_logits = torch.tensor([])
    value = torch.tensor([])
    action_edges = []
    buffer = RolloutBuffer()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    ppo_agent = PPOAgent(model, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2)
    starting_state: Game = None
    init_action_edges = None
    post_placement_state: Game = None
    post_placement_edges = None

    moves_this_turn = []
    total_rewards = defaultdict(float)
    prev_state: Game = None
    learning_stats_file: TextIOWrapper = open(f"learning_stats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "a")
    writer = SummaryWriter(log_dir="../analysis/logs/Atilla_World_extra_aggressive_better_balanced_from_scratch")  # Store data here

    game_number = 1
    num_attack_tracker = StatTracker()
    num_succes_attacks_tracker = StatTracker()
    army_per_attack_tracker = StatTracker()

    @override
    def is_rl_bot(self):
        return True

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())
        faulthandler.enable()
        self.learning_stats_file.write("clip: 0.2; gamma: 0.99; lam: 0.95; lr: 5e-5; entropy_factor: 0.01\n")

    def run_model(self, game: Game, action_edges: torch.Tensor = None, action: str = 'both'):
        node_features = game.create_node_features()
        army_counts = torch.tensor(
            [node[-1] for node in node_features], dtype=torch.float
        )
        graph = torch.tensor(node_features, dtype=torch.float)

        if game.phase == Phase.PLACE_ARMIES:
            action = 'placement'
        elif game.phase == Phase.ATTACK_TRANSFER:
            action = 'attack'
        else:
            logging.warning("still using action \"both\"")
            action = 'both'

        if len(action_edges) == 0:
            logging.debug("no regions owned")
            return torch.Tensor(0), torch.Tensor(0), torch.Tensor(0)
        else:
            return self.model(graph, action_edges, army_counts, action)

    @override
    def init_turn(self, game: Game):
        if self.model.edge_tensor is None:
            self.model.edge_tensor = torch.tensor(game.world.torch_edge_list, dtype=torch.long)
        if game.round < 3 or game.round % 20 == 0:
            logging.info(f"turn {game.round} started")
            for p in range(1, game.config.num_players + 1):
                regions = game.regions_owned_by(p)
                logging.info(
                    f"player {p} owns {len(regions)} regions and {game.number_of_armies_owned(p)} armies (" + str(
                        [f"{r}: {game.get_armies(r)}" for r in regions]) + ')'
                )
        self.init_action_edges = deepcopy(torch.tensor(game.create_action_edges(), dtype=torch.long))

        self.moves_this_turn = []
        self.starting_state = deepcopy(game)
        # with torch.no_grad():
        #     (self.placement_logits, self.attack_logits, self.army_logits) = (
        #         self.run_model(game)
        #     )

    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        chose = random.choice(choosable)
        return chose

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        self.init_turn(game)
        with torch.no_grad():
            placement_logits, attack_logits, army_logits = self.run_model(game=game,
                                                                          action_edges=self.init_action_edges,
                                                                          action='placement')
        self.placement_logits = placement_logits

        if len(self.init_action_edges) == 0:
            logging.warning("no placements")
            return []
        me = game.turn

        mine = [r.get_id() for r in game.regions_owned_by(me)]
        all_regions = set(range(len(game.world.regions)))
        not_mine = all_regions.difference(set(mine))
        available = game.armies_per_turn(me)
        self.placement_logits[list(not_mine)] = float("-inf")
        # self.writer.add_histogram("Histograms/Placement_Logits", self.placement_logits, self.game_number)

        placement_probs = self.placement_logits.softmax(dim=0)
        try:
            nodes = torch.multinomial(
                placement_probs, num_samples=available, replacement=True
            )
        except RuntimeError as re:
            print(placement_probs)
            print(self.placement_logits)
            raise re
        placement = torch.bincount(nodes, minlength=self.placement_logits.size(0))
        ret = []

        for ix, p in enumerate(placement.tolist()):
            if p > 0:
                ret.append(PlaceArmies(game.world.regions[ix], p))
        self.moves_this_turn += ret
        if len(ret) == 0:
            logging.warning("no placements")
        return ret

    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        per_node = True
        self.post_placement_state = deepcopy(game)
        self.post_placement_edges = torch.tensor(game.create_action_edges(), dtype=torch.long)
        with torch.no_grad():
            placement_logits, attack_logits, army_logits = self.run_model(game=game,
                                                                          action_edges=self.post_placement_edges)
            # self.writer.add_histogram("Histograms/Edge_Logits", attack_logits, self.game_number)
            # self.writer.add_histogram("Histograms/Army_Logits", army_logits, self.game_number)

        self.attack_logits, self.army_logits = attack_logits, army_logits
        if per_node:
            edges = self.sample_attacks_per_node()
        else:
            edges = self.sample_n_attacks(game, 5)
        return self.create_attack_transfers(game, edges)

    def terminate(self, game: Game):
        logging.debug("RLGNNAgent terminated")
        self.end_move(game)
        self.buffer.clear()
        self.init_action_edges = torch.Tensor([])
        self.post_placement_edges = torch.Tensor([])

        self.writer.add_scalar('win', game.winning_player() == self.agent_number, self.game_number)
        self.writer.add_scalar('loss_mean', self.ppo_agent.loss_tracker.mean(), self.game_number)
        self.writer.add_scalar('loss_std', self.ppo_agent.loss_tracker.std(), self.game_number)
        self.writer.add_scalar('min_loss', self.ppo_agent.loss_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_loss', self.ppo_agent.loss_tracker.get_max(), self.game_number)

        self.writer.add_scalar('act_loss_mean', self.ppo_agent.act_loss_tracker.mean(), self.game_number)
        self.writer.add_scalar('act_loss_std', self.ppo_agent.act_loss_tracker.std(), self.game_number)
        self.writer.add_scalar('min_act_loss', self.ppo_agent.act_loss_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_act_loss', self.ppo_agent.act_loss_tracker.get_max(), self.game_number)

        self.writer.add_scalar('crit_loss_mean', self.ppo_agent.crit_loss_tracker.mean(), self.game_number)
        self.writer.add_scalar('crit_loss_std', self.ppo_agent.crit_loss_tracker.std(), self.game_number)
        self.writer.add_scalar('min_crit_loss', self.ppo_agent.crit_loss_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_crit_loss', self.ppo_agent.crit_loss_tracker.get_max(), self.game_number)

        self.writer.add_scalar('edge_entropy_mean', self.ppo_agent.edge_entropy_tracker.mean(), self.game_number)
        self.writer.add_scalar('edge_entropy_std', self.ppo_agent.edge_entropy_tracker.std(), self.game_number)
        self.writer.add_scalar('min_edge_entropy', self.ppo_agent.edge_entropy_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_edge_entropy', self.ppo_agent.edge_entropy_tracker.get_max(), self.game_number)

        self.writer.add_scalar('placement_entropy_mean', self.ppo_agent.placement_entropy_tracker.mean(), self.game_number)
        self.writer.add_scalar('placement_entropy_std', self.ppo_agent.placement_entropy_tracker.std(), self.game_number)
        self.writer.add_scalar('min_placement_entropy', self.ppo_agent.placement_entropy_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_placement_entropy', self.ppo_agent.placement_entropy_tracker.get_max(), self.game_number)

        self.writer.add_scalar('army_entropy_mean', self.ppo_agent.army_entropy_tracker.mean(), self.game_number)
        self.writer.add_scalar('army_entropy_std', self.ppo_agent.army_entropy_tracker.std(), self.game_number)
        self.writer.add_scalar('min_army_entropy', self.ppo_agent.army_entropy_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_army_entropy', self.ppo_agent.army_entropy_tracker.get_max(), self.game_number)

        self.writer.add_scalar('ratio_mean', self.ppo_agent.ratio_tracker.mean(), self.game_number)
        self.writer.add_scalar('ratio_std', self.ppo_agent.ratio_tracker.std(), self.game_number)
        self.writer.add_scalar('min_ratio', self.ppo_agent.ratio_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_ratio', self.ppo_agent.ratio_tracker.get_max(), self.game_number)

        self.writer.add_scalar('advantage_mean', self.ppo_agent.adv_tracker.mean(), self.game_number)
        self.writer.add_scalar('advantage_std', self.ppo_agent.adv_tracker.std(), self.game_number)
        self.writer.add_scalar('min_advantage', self.ppo_agent.adv_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_advantage', self.ppo_agent.adv_tracker.get_max(), self.game_number)

        self.writer.add_scalar('value_mean', self.ppo_agent.value_tracker.mean(), self.game_number)
        self.writer.add_scalar('value_std', self.ppo_agent.value_tracker.std(), self.game_number)
        self.writer.add_scalar('min_value', self.ppo_agent.value_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_value', self.ppo_agent.value_tracker.get_max(), self.game_number)

        self.writer.add_scalar('value_pred_mean', self.ppo_agent.value_pred_tracker.mean(), self.game_number)
        self.writer.add_scalar('value_pred_std', self.ppo_agent.value_pred_tracker.std(), self.game_number)
        self.writer.add_scalar('min_pred_value', self.ppo_agent.value_pred_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_pred_value', self.ppo_agent.value_pred_tracker.get_max(), self.game_number)

        self.writer.add_scalar('returns_mean', self.ppo_agent.returns_tracker.mean(), self.game_number)
        self.writer.add_scalar('returns_std', self.ppo_agent.returns_tracker.std(), self.game_number)
        self.writer.add_scalar('min_returns', self.ppo_agent.returns_tracker.get_min(), self.game_number)
        self.writer.add_scalar('max_returns', self.ppo_agent.returns_tracker.get_max(), self.game_number)

        self.writer.add_scalar('attacks_per_turn', self.num_attack_tracker.mean(), self.game_number)
        self.writer.add_scalar('armies_per_attack', self.army_per_attack_tracker.mean(), self.game_number)
        self.writer.add_scalar('won_battles_per_turn', self.num_succes_attacks_tracker.mean(), self.game_number)

        self.writer.add_scalar('missed opportunities', self.total_rewards['missed opportunities'] / game.round, self.game_number)
        self.writer.add_scalar('missed transfers', self.total_rewards['missed transfers'] / game.round, self.game_number)
        self.writer.add_scalar('turn_with_attack', self.total_rewards['turn_with_attack'] / game.round, self.game_number)
        self.writer.add_scalar('turn_with_mult_attacks', self.total_rewards['turn_with_mult_attacks'] / game.round, self.game_number)
        self.writer.add_scalar('num_regions', self.total_rewards['num_regions'] / game.round, self.game_number)
        self.writer.add_scalar('army_difference', self.total_rewards['army_difference'] / game.round, self.game_number)
        for key, value in self.total_rewards.items():
            if key in ['missed opportunities', 'missed transfers', 'turn_with_attack', 'turn_with_mult_attacks', 'num_regions', 'army_difference']:
                continue
            self.writer.add_scalar(key, value, self.game_number)

        # my_regions = game.regions_owned_by(self.agent_number)
        # for r in game.world.regions:
        #     if r in my_regions:
        #         self.total_rewards[f"region_{r}_owned"] = 1
        #     else:
        #         self.total_rewards[f"region_{r}_owned"] = 0
        #
        # logging.info(f"total rewards this round: {self.total_rewards}")

        # self.learning_stats_file.write(json.dumps(self.total_rewards) + '\n')
        # self.learning_stats_file.flush()
        self.total_rewards = defaultdict(int)

        self.game_number += 1

    @override
    def end_move(self, game: Game):
        if len(self.moves_this_turn) == 0 and not game.is_done():
            logging.debug("did no moves")
            return
        value = self.model.get_value(game).detach()
        done = game.is_done()
        reward = self.compute_rewards(game)
        # Store transition in buffer
        self.buffer.add(
            self.starting_state,
            self.init_action_edges,
            self.moves_this_turn,
            compute_log_probs(self.get_attacks(), self.attack_logits, self.army_logits, self.get_placements(),
                              self.placement_logits, self.post_placement_edges),
            reward,
            value,
            done,
        )
        self.prev_state = deepcopy(game)

        with torch.no_grad():
            next_value = self.model.get_value(game) * (1 - done)
        if (game.round + 1) % 1 == 0 or done:
            self.ppo_agent.update(self.buffer, next_value, self)
            self.buffer.clear()

    def compute_rewards(self, game: Game) -> float:
        prev_state = self.prev_state
        current_state = game
        player_id = self.agent_number
        turn_number = game.round
        reward, region_reward, continent_reward, army_reward, action_reward = 0, 0, 0, 0, 0
        long_game_reward = 0
        if prev_state is not None:
            prev_regions = set([r.get_id() for r in prev_state.regions_owned_by(player_id)])
            prev_continents = prev_state.get_bonus_armies(player_id)
            prev_armies = prev_state.number_of_armies_owned(player_id)
            prev_armies_enemy = sum(
                [prev_state.number_of_armies_owned(pid) for pid in range(1, prev_state.config.num_players + 1) if
                 pid != player_id])
        else:
            prev_regions = set()
            prev_continents = 0
            prev_armies = 0

        curr_regions = set([r.get_id() for r in current_state.regions_owned_by(player_id)])
        curr_continents = current_state.get_bonus_armies(player_id)
        curr_armies = current_state.number_of_armies_owned(player_id)
        curr_armies_enemy = sum([current_state.number_of_armies_owned(pid) for pid in
                                 range(1, current_state.config.num_players + 1) if pid != player_id])

        if prev_state is not None:
            # 1️⃣ Region control
            gained_regions = len(curr_regions.difference(prev_regions))
            lost_regions = len(prev_regions.difference(curr_regions))
            region_reward = gained_regions * 0.1 - lost_regions * 0.025

            reward += region_reward

        # 2️⃣ Continent bonuses
        if prev_state is not None:
            continent_reward = (curr_continents - prev_continents) * 0.5
            reward += continent_reward

        # 3️⃣ Army dynamics
        if prev_state is not None:
            armies_destroyed = max(0, prev_armies_enemy - curr_armies_enemy)  # noqa
            armies_lost = max(0, prev_armies - curr_armies)

            diff = 0.1 * (armies_destroyed - armies_lost)  # noqa
        else:
            diff = 0
        normalized_army_delta = diff / (curr_armies + curr_armies_enemy + 1e-8)
        army_reward = 0.1 * normalized_army_delta

        reward += army_reward

        # 4️⃣ Action dynamics
        attacks = self.get_attacks(inc_transfers=False, object_data=True)
        self.num_attack_tracker.log(len(attacks))
        wins = 0
        armies_used = 0
        destroyed_armies = 0
        for a in attacks:
            armies_used += a.armies
            destroyed_armies += a.result.defenders_destroyed
            if a.result.winner == FightSide.ATTACKER:
                wins += 1
        if len(attacks) > 0:
            self.army_per_attack_tracker.log(armies_used / len(attacks))
        else:
            self.army_per_attack_tracker.log(0)
        self.num_succes_attacks_tracker.log(wins)
        if len(attacks) > 0:
            eff = destroyed_armies / armies_used
            action_reward = 0.01
            action_reward += 0.02 * eff

        reward += action_reward

        # 5️⃣ Long-game penalty
        if turn_number > 300:
            long_game_reward -= 0.02
        reward += long_game_reward

        if game.is_done():
            # Final win/loss reward
            if game.winning_player() == player_id:
                reward += 75.0 + max(0., 10. - 0.1 * game.round)
            elif game.winning_player() != -1:
                reward -= 50.0

        attacks = set([(a[0], a[1]) for a in self.get_attacks(inc_transfers=False)])
        attack_transfers = set([(a[0], a[1]) for a in self.get_attacks(inc_transfers=True)])
        transfers = attack_transfers.difference(attacks)
        my_regions = game.regions_owned_by(self.agent_number)
        passivity_reward = (curr_armies - armies_used - len(my_regions))/(curr_armies+len(my_regions)) * -0.01
        transfer_reward = 0
        for r in my_regions:
            has_enemy = False
            for n in r.get_neighbours():
                if game.get_owner(n) != self.agent_number:
                    has_enemy = True
                    if (r.get_id(), n.get_id()) not in attacks:
                        if game.get_armies(r) > (game.get_armies(n) + 1):
                            self.total_rewards['missed opportunities'] += 1.
                    break

            if not has_enemy and r.get_id() not in set([t[0] for t in transfers]):
                self.total_rewards['missed transfers'] += 1.
                transfer_reward -= 0.01
        reward += passivity_reward

        reward += transfer_reward

        placement_rewards = 0
        placements = self.get_placements(as_objects=True)
        good_placements = 0
        for p in placements:
            good_placements += 1 if any(
                [n for n in p.region.get_neighbours() if game.get_owner(n) != self.agent_number]) else 0

        placement_rewards += ((good_placements * 0.05 - (len(placements) - good_placements) * 0.025) /
                              len(game.regions_owned_by(player_id)))

        reward += placement_rewards

        if len(attacks) > 0:
            self.total_rewards['turn_with_attack'] += 1
        if len(attacks) > 1:
            self.total_rewards['turn_with_mult_attacks'] += 1

        enemy_armies = 0
        my_armies = 0
        for p in range(1, game.config.num_players + 1):
            if p == self.agent_number:
                my_armies = game.number_of_armies_owned(p)
            else:
                enemy_armies += game.number_of_armies_owned(p)

        self.total_rewards['army_difference'] += my_armies - enemy_armies
        self.total_rewards['num_regions'] += len(my_regions)
        self.total_rewards['region_reward'] += region_reward
        self.total_rewards['continent_reward'] += continent_reward
        self.total_rewards['action_reward'] += action_reward
        self.total_rewards['long_game_reward'] += long_game_reward
        self.total_rewards['army_reward'] += army_reward
        self.total_rewards['passivity_reward'] += passivity_reward
        self.total_rewards['placement_reward'] += placement_rewards
        self.total_rewards['transfer_reward'] += transfer_reward
        self.total_rewards['reward'] += reward

        return reward

    def get_attacks(self, actions=None, object_data=False, inc_transfers=True) -> list[AttackTransfer | int]:
        if actions is None:
            actions = self.moves_this_turn
        if object_data:
            ret = [a for a in actions if (isinstance(a, AttackTransfer) and (inc_transfers or a.is_attack()))]
        else:
            ret = [(a.from_region.get_id(), a.to_region.get_id(), a.armies) for a in actions if
                   (isinstance(a, AttackTransfer) and (inc_transfers or a.is_attack()))]
        return ret

    def get_placements(self, actions=None, as_objects=False) -> list[PlaceArmies | int]:
        if actions is None:
            actions = self.moves_this_turn
        if as_objects:
            return [a for a in actions if isinstance(a, PlaceArmies)]
        ret = []
        for p in [a for a in actions if isinstance(a, PlaceArmies)]:
            ret += p.armies * [p.region.get_id()]

        return ret

    def sample_n_attacks(self, game, n):
        if len(self.post_placement_edges) == 0:
            return []
        logging.debug(
            f"calculating attack/transfers, having: {len(game.regions_owned_by(self.agent_number))} regions"
        )
        T = 1.5  # >1 smooths, <1 sharpens
        smoothed_logits = self.attack_logits / T
        probs = torch.softmax(smoothed_logits, dim=0)
        k = min(n, probs.size(0))
        topk_probs, selected_idxs = torch.topk(probs, k)

        # selected_idxs = (probs > (0.7 * probs.max())).nonzero(as_tuple=True)[0]
        ret = []
        if len(selected_idxs) == 0:
            logging.debug("doing no moves")
            return ret
        for idx in selected_idxs.tolist():
            try:
                src, tgt = self.post_placement_edges[idx]
            except IndexError as ie:
                print(self.post_placement_edges.tolist())
                print(probs)
                print(idx)
                print(selected_idxs)
                raise ie
            except ValueError as ve:
                print(self.post_placement_edges.tolist())
                print(idx)
                print(selected_idxs)
                print(ve)
                raise ve

            if src != tgt:
                ret.append((src, tgt))
        return ret

    def sample_attacks_per_node(self):
        if len(self.post_placement_edges) == 0:
            return []
        src_nodes = torch.unique(self.post_placement_edges[:, 0])
        ret = []
        for src in src_nodes:
            mask = self.post_placement_edges[:, 0] == src
            candidate_edges = self.post_placement_edges[mask]
            candidate_logits = self.attack_logits[mask]

            probs = f.softmax(candidate_logits, dim=-1)
            action_index = torch.multinomial(probs, 1).item()
            tgt = candidate_edges[action_index][1].item()
            if src != tgt:
                ret.append((src.item(), tgt))
        return ret

    def create_attack_transfers(self, game: Game, edges):
        used_armies = defaultdict(int)
        ret = []
        n_attacks = 0
        n_army_attacks = 0
        for src, tgt in edges:
            mask = (self.post_placement_edges[:, 0] == src) & (self.post_placement_edges[:, 1] == tgt)

            indices = mask.nonzero(as_tuple=False)
            idx = indices.item()
            available_armies = game.armies[src] - (
                used_armies[src]
            )  # leave one behind
            if available_armies <= 0:
                continue

            # Choose how many armies to send: e.g., max logit under the cap
            try:
                T = 1.5
                army_logit = self.army_logits[idx][:available_armies]

                smoothed_army_logits = army_logit / T
                smoothed_army_logits += torch.randn_like(smoothed_army_logits) * 0.1
                army_probs = f.softmax(smoothed_army_logits, dim=-1)
                if len(army_logit) == 0:
                    continue
            except IndexError as ie:
                print(self.army_logits)
                print(idx)
                print(available_armies)
                print(ie)
                raise ie
            try:
                # k = torch.argmax(army_logit).item() + 1  # send k armies
                k = int(torch.distributions.Categorical(probs=army_probs).sample().int())
            except IndexError as ie:

                print(army_logit)
                raise ie
            if k == 0:
                continue
            if k >= available_armies:
                logging.debug("too many armies!")
                continue
            used_armies[src] += int(k)

            ret.append(
                AttackTransfer(game.world.regions[src], game.world.regions[tgt], k, None)
            )
            if game.get_owner(tgt) != self.agent_number:
                n_attacks += 1
                n_army_attacks += k

        logging.debug(f"doing {len(ret)} moves")
        self.moves_this_turn += ret
        if game.round % 20 == 1:

            logging.info(
                f"at round {game.round}, rlgnnagent does {len(ret)} attack/transfers"
            )
            if len(ret) > 0:
                logging.info(
                    f"using {sum([a.armies for a in ret]) / len(ret)} armies on avg"
                )
            logging.info(
                f"at round {game.round}, rlgnnagent does {n_attacks} attacks"
            )
            if n_attacks > 0:
                logging.info(
                    f"using {n_army_attacks / n_attacks} armies on avg"
                )

        return ret
