import logging
import random
import time
import sys
import os

from collections import defaultdict
from dataclasses import dataclass

from src.game.FightSide import FightSide

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv

from src.engine.AgentBase import AgentBase

from src.game.Game import Game
from src.game.Region import Region

from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies


class RolloutBuffer:
    def __init__(self):
        self.states = []
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
        checkpoint = torch.load(path, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("✅ Loaded checkpoint from", path)
    else:
        print("ℹ️ No checkpoint found, starting fresh.")


def compute_gae(rewards, values, last_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages), torch.tensor(returns)


def compute_log_probs(model, states, placement_actions, attack_actions):
    log_probs = []
    for state, placement_action, attack_action in zip(
        states, placement_actions, attack_actions
    ):
        placement_logits, edge_logits, army_logits = model(state)

        # Placement
        placement_log_probs = f.log_softmax(placement_logits, dim=-1)
        plog = placement_log_probs[placement_action]

        # Attack
        if attack_action is None:
            alog = torch.tensor(0.0, device=plog.device)
        else:
            edge_idx, army_amt = attack_action
            edge_log_probs = f.log_softmax(edge_logits, dim=-1)
            army_log_probs = f.log_softmax(army_logits[edge_idx], dim=-1)
            alog = edge_log_probs[edge_idx] + army_log_probs[army_amt]

        log_probs.append(plog + alog)
    return torch.stack(log_probs)


def compute_entropy(states, agent):
    entropies = []
    for state in states:
        placement_logits, edge_logits, army_logits = agent.run_model(state)
        if type(placement_logits) == int:
            return 0
        placement_probs = f.softmax(placement_logits, dim=-1)
        placement_entropy = -(
            placement_probs * f.log_softmax(placement_logits, dim=-1)
        ).sum()

        edge_probs = f.softmax(edge_logits, dim=-1)
        edge_entropy = -(edge_probs * f.log_softmax(edge_logits, dim=-1)).sum()

        army_entropy = 0
        for logits in army_logits:
            probs = f.softmax(logits, dim=-1)
            log_probs = f.log_softmax(logits, dim=-1)
            army_entropy += -(probs * log_probs).sum()

        entropies.append(placement_entropy + edge_entropy + army_entropy)
    return torch.stack(entropies)


class PPOAgent:
    def __init__(
        self,
        policy,
        optimizer: torch.optim.Adam,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ppo_epochs=4,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        load_checkpoint(self.policy, self.optimizer, "res/model/checkpoint.pt")

    def update(self, buffer: RolloutBuffer, last_value, agent):
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            last_value,
            buffer.dones,
            gamma=self.gamma,
            lam=self.lam,
        )

        old_log_probs = buffer.log_probs[0]
        if buffer.states is None or len(buffer.states) == 0:
            return
        for _ in range(self.ppo_epochs):
            placement_logits, attack_logits, army_logits = agent.run_model(
                game=buffer.states[0]
            )
            log_probs = agent.compute_action_log_prob(
                buffer.actions,
                attack_logits,
                army_logits,
                placement_logits,
                buffer.edges,
            )
            diff = log_probs - old_log_probs
            if type(diff) == torch.Tensor:
                ratio = diff.exp()
            else:
                ratio = torch.Tensor(diff).exp()

            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages,
            ).mean()

            values_pred = self.policy.get_value(buffer.states[0])
            value_loss = f.mse_loss(values_pred, returns)
            entropy = compute_entropy(buffer.states, agent)
            if type(entropy) == torch.Tensor:
                entropy = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if os.path.exists("res/model/"):
                torch.save(
                    {
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    "res/model/checkpoint.pt",
                )


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

        # Mask invalid army amounts per edge
        max_sendable = army_counts[src] - 1
        army_mask = (
            torch.arange(self.max_army_send).unsqueeze(0).to(army_logits.device)
        )  # [1, max_army_send]
        mask = army_mask < max_sendable.unsqueeze(1)  # [num_edges, max_army_send]

        army_logits[~mask] = -1e9  # Mask out too-large moves

        return edge_logits, army_logits


class WarlightPolicyNet(nn.Module):
    def __init__(self, node_feat_dim, embed_dim=64, max_army_send=100):
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

    def get_value(self, game: Game):
        node_features = game.create_army_input_list()
        graph = torch.tensor(node_features, dtype=torch.float)
        edge_list = game.world.torch_edge_list
        edge_tensor = torch.tensor(edge_list, dtype=torch.long)

        x = f.relu(self.gnn1(graph, edge_tensor))
        node_embeddings = self.gnn2(x, edge_tensor)

        # Mean pool over all nodes to get a graph-level embedding
        graph_embedding = node_embeddings.mean(dim=0)
        value = self.value_head(graph_embedding)
        return value.squeeze(-1)

    def forward(self, x, edge_index, action_edges, army_counts):
        """
        x: [num_nodes, node_feat_dim]       # node features
        edge_index: [2, num_edges]          # graph structure
        action_edges: [num_actions, 2]      # list of (src, tgt) edges for attacks
        army_counts: [num_nodes]            # current army count on each node
        """
        # GNN
        x = f.relu(self.gnn1(x, edge_index))
        node_embeddings = self.gnn2(x, edge_index)

        # Placement
        placement_logits = self.placement_head(node_embeddings).squeeze(
            -1
        )  # [num_nodes]

        # Attack
        attack_logits, army_logits = self.attack_head(
            node_embeddings, action_edges, army_counts
        )

        return placement_logits, attack_logits, army_logits


@dataclass
class RLGNNAgent(AgentBase):
    in_channels = 4
    hidden_channels = 64
    model = WarlightPolicyNet(in_channels, hidden_channels)
    placement_logits = torch.tensor([])
    attack_logits = torch.tensor([])
    army_logits = torch.tensor([])
    value = torch.tensor([])
    action_edges = []
    edge_tensor = torch.tensor([])
    buffer = RolloutBuffer()
    entropy_coef = 0.5
    value_coef = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    ppo_agent = PPOAgent(model, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2)
    starting_state: Game = None
    log_probs = 0
    moves_this_turn = []
    lost_regions = 0
    current_regions = 0

    @override
    def is_rl_bot(self):
        return True

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())

    def run_model(self, game: Game):
        node_features = game.create_army_input_list()
        army_counts = torch.tensor(
            [node[-1] for node in node_features], dtype=torch.float
        )
        graph = torch.tensor(node_features, dtype=torch.float)
        edge_list = game.world.torch_edge_list
        self.edge_tensor = torch.tensor(edge_list, dtype=torch.long)
        self.action_edges = torch.tensor(game.create_action_edges(), dtype=torch.long)
        if len(self.action_edges) == 0:
            logging.debug("no regions owned")
            return 0, 0, 0
        else:
            return self.model(graph, self.edge_tensor, self.action_edges, army_counts)

    @override
    def init_turn(self, game: Game):
        if game.round % 50 == 1:
            logging.info(f"turn {game.round} started")
            for p in range(1, game.config.num_players + 1):
                logging.info(
                    f"player {p} owns {len(game.regions_owned_by(p))} regions and {game.number_of_armies_owned(p)} armies"
                )
        new_regions = len(game.regions_owned_by(self.agent_number))
        self.lost_regions = self.current_regions - new_regions
        self.current_regions = new_regions
        self.log_probs = torch.tensor([])
        self.moves_this_turn = []
        self.starting_state = game
        with torch.no_grad():
            (self.placement_logits, self.attack_logits, self.army_logits) = (
                self.run_model(game)
            )

    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        return random.choice(choosable)

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        self.init_turn(game)
        if len(self.action_edges) == 0:
            logging.warning("no placements")
            return []

        me = game.turn

        mine = [r.get_id() for r in game.regions_owned_by(me)]
        all_regions = set(range(len(game.world.regions)))
        not_mine = all_regions.difference(set(mine))
        available = game.armies_per_turn(me)
        self.placement_logits[list(not_mine)] = float("-inf")
        placement_probs = self.placement_logits.softmax(dim=0)

        nodes = torch.multinomial(
            placement_probs, num_samples=available, replacement=True
        )
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
        if len(self.action_edges) == 0:
            return []
        logging.debug(
            f"calculating attack/transfers, having: {len(game.regions_owned_by(self.agent_number))} regions"
        )
        probs = torch.softmax(self.attack_logits, dim=0)
        selected_idxs = (probs > (0.7 * probs.max())).nonzero(as_tuple=True)[0]
        selected_attacks = []
        if len(selected_idxs) == 0:
            logging.debug("doing no moves")
            return selected_attacks
        used_armies = defaultdict(int)
        for idx in selected_idxs.tolist():
            try:
                src, tgt = self.action_edges[idx]
            except IndexError as ie:
                print(self.action_edges.tolist())
                print(probs)
                print(idx)
                print(selected_idxs)
                raise ie
            except ValueError as ve:
                print(self.action_edges.tolist())
                print(idx)
                print(selected_idxs)
                print(ve)
                raise ve

            if src == tgt:
                continue
            available_armies = game.armies[src] - (
                1 + used_armies[src.item()]
            )  # leave one behind
            if available_armies <= 0:
                continue

            # Choose how many armies to send: e.g., max logit under the cap
            try:
                army_logit = self.army_logits[idx][:available_armies]
                if len(army_logit) == 0:
                    continue
            except IndexError as ie:
                print(self.army_logits)
                print(idx)
                print(available_armies)
                print(ie)
                raise ie
            try:
                k = torch.argmax(army_logit).item() + 1  # send k armies
            except IndexError as ie:

                print(army_logit)
                raise ie
            if k >= available_armies:
                logging.debug("too many armies!")
                continue
            used_armies[src.item()] += int(k)

            selected_attacks.append(
                AttackTransfer(game.world.regions[src], game.world.regions[tgt], k)
            )
        logging.debug(f"doing {len(selected_attacks)} moves")
        self.moves_this_turn += selected_attacks
        if game.round % 50 == 1:

            logging.info(
                f"at round {game.round}, rlgnnagent does {len(selected_attacks)} attack/transfers"
            )
            if len(selected_attacks) > 0:
                logging.info(
                    f"using {sum([a.armies for a in selected_attacks])/len(selected_attacks)} armies on avg"
                )

        return selected_attacks

    def terminate(self, game: Game):
        logging.debug("RLGNNAgent terminated")
        self.end_move(game)
        # self.trajectory = dict()

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
            game,
            self.action_edges,
            self.moves_this_turn,
            self.compute_action_log_prob(),
            reward,
            value,
            done,
        )

        with torch.no_grad():
            next_value = self.model.get_value(game) * (1 - done)
        if (game.round + 1) % 1 == 0 or done:
            self.ppo_agent.update(self.buffer, next_value, self)
            self.buffer.clear()

    def compute_action_log_prob(
        self,
        actions=None,
        attack_logits=None,
        army_logits=None,
        placement_logits=None,
        action_edges=None,
    ):
        if actions is None:
            actions = self.moves_this_turn
            attack_logits = self.attack_logits
            army_logits = self.army_logits
            placement_logits = self.placement_logits
            action_edges = self.action_edges

        log_probs = []
        for placement in self.get_placements(actions):
            p = placement.region.get_id()
            placement_log_prob = torch.log_softmax(placement_logits, dim=0)[p]
            log_probs.append(placement_log_prob)

        for i, attack in enumerate(self.get_attacks(actions)):
            src = attack.from_region.get_id()
            tgt = attack.to_region.get_id()
            action_edge = action_edges[i]
            src_ix = (
                (action_edge == torch.tensor([src, tgt]))
                .all(dim=0)
                .nonzero(as_tuple=True)[0]
            )

            armies = attack.get_armies()
            attack_log_prob = torch.log_softmax(attack_logits, dim=0)[src_ix]
            try:
                army_log_prob = torch.log_softmax(army_logits[src_ix], dim=1)[
                    :, armies - 1
                ]
            except IndexError as ie:
                print(src_ix, armies)
                print(torch.log_softmax(army_logits[src_ix], dim=1))
                raise ie
            log_probs.append(army_log_prob)
            log_probs.append(attack_log_prob)
        ret = sum(log_probs)
        return ret

    def compute_rewards(self, game: Game) -> float:
        ret = self.lost_regions * -0.04
        for a in self.get_attacks():
            if a.result is not None:
                if a.result.winner == FightSide.ATTACKER:
                    ret += 0.05
                ret -= 0.01 * a.result.attackers_destroyed
                ret += 0.01 * a.result.defenders_destroyed
        if game.is_done() and game.winning_player() == self.agent_number:
            ret += 3
        elif game.is_done():
            ret -= 3
        ret += 0.01 * game.get_bonus_armies(self.agent_number)
        logging.debug(f"reward = {ret}")
        if ret > 5:
            logging.debug("reached 5")
        return ret

    def get_attacks(self, actions=None):
        if actions is None:
            actions = self.moves_this_turn
        return [a for a in actions if isinstance(a, AttackTransfer)]

    def get_placements(self, actions=None):
        if actions is None:
            actions = self.moves_this_turn
        return [p for p in actions if isinstance(p, PlaceArmies)]
