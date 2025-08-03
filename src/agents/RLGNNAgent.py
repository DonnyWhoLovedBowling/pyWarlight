from cmath import phase
from copy import copy
import json
import logging
import random
import time
import sys
import os

from collections import defaultdict
from dataclasses import dataclass
from io import TextIOWrapper

import numpy as np
from torch.utils.tensorboard import SummaryWriter


from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.game.FightSide import FightSide
from src.game.Phase import Phase

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override, TextIO

import torch
import torch.nn.functional as f

from datetime import datetime

from src.engine.AgentBase import AgentBase

from src.game.Game import Game
from src.game.Region import Region

from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies
from src.agents.RLUtils.RLUtils import RolloutBuffer, StatTracker, compute_log_probs, PrevStateBuffer
from src.agents.RLUtils.PPOAgent import PPOAgent

import faulthandler


do_hm_search = True


@dataclass
class RLGNNAgent(AgentBase):
    in_channels = 8
    hidden_channels = 64
    batch_size = 24
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = WarlightPolicyNet(in_channels, hidden_channels).to(device)
    placement_logits = torch.tensor([])
    attack_logits = torch.tensor([])
    army_logits = torch.tensor([])
    value = torch.tensor([])
    action_edges = torch.tensor([])
    buffer = RolloutBuffer()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    ppo_agent = PPOAgent(model, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2)
    starting_node_features: torch.Tensor = None
    post_placement_node_features: torch.Tensor= None

    moves_this_turn = []
    total_rewards = defaultdict(float)
    prev_state: PrevStateBuffer = None
    learning_stats_file: TextIOWrapper = open(f"learning_stats_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt", "a")
    writer = SummaryWriter(log_dir="analysis/logs/Atilla_World_balanced_entropy")  # Store data here

    game_number = 1
    num_attack_tracker = StatTracker()
    num_succes_attacks_tracker = StatTracker()
    army_per_attack_tracker = StatTracker()

    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @override
    def is_rl_bot(self):
        return True

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())
        faulthandler.enable()
        self.learning_stats_file.write("clip: 0.2; gamma: 0.99; lam: 0.95; lr: 5e-5; entropy_factor: 0.01\n")


    def run_model(self, node_features: torch.Tensor, action_edges: torch.Tensor = None, action: str = None):
        """
        Handle both single samples and batches
        
        Args:
            node_features: [num_nodes, features] for single OR [batch_size, num_nodes, features] for batch
            action_edges: [num_edges, 2] for single OR [batch_size, num_edges, 2] for batch
            action: Phase.PLACE_ARMIES or Phase.ATTACK_TRANSFER
        """
        # Detect if this is a batch
        is_batch = node_features.dim() == 3  # [batch_size, num_nodes, features]
        
        if is_batch:
            return self._run_model_batch(node_features, action_edges, action)
        else:
            # Single sample case - original logic
            army_counts = node_features[:, -1].to(dtype=torch.float, device=self.device)
            graph = node_features.to(dtype=torch.float, device=self.device)
                
            if len(action_edges) == 0:
                logging.debug("no regions owned")
                return torch.tensor(0), torch.tensor(0), torch.tensor(0)
            else:
                return self.model(graph, action_edges, army_counts, action)

    def _run_model_batch(self, node_features: torch.Tensor, action_edges: torch.Tensor, action: str = None):
        """
        Process batch by running model on each sample individually, then stack results
        
        Args:
            node_features: [batch_size, num_nodes, features] 
            action_edges: [batch_size, num_edges, 2] (padded to 42 edges)
            action: Phase.PLACE_ARMIES or Phase.ATTACK_TRANSFER
        """
        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)
        
        placement_logits_list = []
        attack_logits_list = []
        army_logits_list = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            sample_features = node_features[i]  # [num_nodes, features]
            sample_edges = action_edges[i]  # [42, 2] (padded)
            
            # Remove padding from edges (edges with -1 are padding)
            valid_edge_mask = (sample_edges[:, 0] >= 0) & (sample_edges[:, 1] >= 0)
            valid_edges = sample_edges[valid_edge_mask]  # [num_valid_edges, 2]
            
            # Run model on this sample
            if valid_edges.numel() == 0:
                logging.debug("no regions owned in batch sample")
                pl = torch.zeros(num_nodes) if action == Phase.PLACE_ARMIES or action is None else torch.tensor([])
                al = torch.zeros(0) if action == Phase.ATTACK_TRANSFER or action is None else torch.tensor([])
                arl = torch.zeros(0, 50) if action == Phase.ATTACK_TRANSFER or action is None else torch.tensor([])
            else:
                pl, al, arl = self.model(
                    sample_features.to(dtype=torch.float, device=self.device),
                    valid_edges,
                    sample_features[:, -1].to(dtype=torch.float, device=self.device),
                    action
                )
            
            placement_logits_list.append(pl)
            attack_logits_list.append(al)
            army_logits_list.append(arl)
        
        # Stack and pad results
        if action == Phase.PLACE_ARMIES or action is None:
            # Stack placement logits
            if all(pl.numel() > 0 for pl in placement_logits_list):
                placement_logits = torch.stack(placement_logits_list)  # [batch_size, num_nodes]
            else:
                placement_logits = torch.zeros(batch_size, num_nodes)
        else:
            placement_logits = torch.tensor([])
        
        if action == Phase.ATTACK_TRANSFER or action is None:
            # Pad attack logits to 42 edges
            padded_attack = []
            for al in attack_logits_list:
                if al.numel() == 0:
                    padded = torch.full((42,), -1e9)
                else:
                    padding_needed = 42 - al.size(0)
                    if padding_needed > 0:
                        padding = torch.full((padding_needed,), -1e9)
                        padded = torch.cat([al, padding])
                    else:
                        padded = al[:42]  # Truncate if too long
                padded_attack.append(padded)
            attack_logits = torch.stack(padded_attack)  # [batch_size, 42]
            
            # Pad army logits to [42, max_army_send]
            max_army_send = max(arl.size(1) if arl.numel() > 0 else 0 for arl in army_logits_list)
            if max_army_send == 0:
                max_army_send = 50  # Default
                
            padded_army = []
            for arl in army_logits_list:
                if arl.numel() == 0:
                    padded = torch.full((42, max_army_send), -1e9)
                else:
                    # Pad edges dimension to 42
                    edge_padding_needed = 42 - arl.size(0)
                    if edge_padding_needed > 0:
                        edge_padding = torch.full((edge_padding_needed, arl.size(1)), -1e9)
                        arl_edge_padded = torch.cat([arl, edge_padding], dim=0)
                    else:
                        arl_edge_padded = arl[:42]  # Truncate if too long
                    
                    # Pad army dimension to max_army_send
                    army_padding_needed = max_army_send - arl_edge_padded.size(1)
                    if army_padding_needed > 0:
                        army_padding = torch.full((42, army_padding_needed), -1e9)
                        padded = torch.cat([arl_edge_padded, army_padding], dim=1)
                    else:
                        padded = arl_edge_padded[:, :max_army_send]
                padded_army.append(padded)
            army_logits = torch.stack(padded_army)  # [batch_size, 42, max_army_send]
        else:
            attack_logits = torch.tensor([])
            army_logits = torch.tensor([])
        
        return placement_logits, attack_logits, army_logits

    @override
    def init_turn(self, game: Game):
        # logging.info(f"round {game.round}")
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
        self.action_edges = torch.tensor(game.create_action_edges(), dtype=torch.long)
        if len(self.action_edges) == 0:
            logging.debug("no action edges owned")
        self.moves_this_turn = []
        self.starting_node_features = torch.tensor(game.create_node_features(), dtype= torch.float, device=self.device)
        
    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        chose = random.choice(choosable)
        return chose

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        self.init_turn(game)
        with torch.no_grad():
            placement_logits, attack_logits, army_logits = self.run_model(self.starting_node_features,
                                                                          action_edges=self.action_edges)
        self.placement_logits = placement_logits

        if len(self.action_edges) == 0:
            logging.warning("no placements")
            return []
        me = game.turn

        mine = [r.get_id() for r in game.regions_owned_by(me)]
        all_regions = set(range(len(game.world.regions)))
        not_mine = all_regions.difference(set(mine))
        available = game.armies_per_turn(me)
        if not self.placement_logits.is_leaf:
            self.placement_logits = self.placement_logits.detach()
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
        # After placements are determined
        placement_tensor = torch.zeros(len(game.world.regions))
        placement_per_n_neigbors_tensor = torch.zeros(10)
        placements_next_to_enemy = 0
        total_placements = 0
        for ix, p in enumerate(placement.tolist()):
            placement_tensor[ix] = p
            n_neighbors = len(game.world.regions[ix].get_neighbours())
            if n_neighbors < len(placement_per_n_neigbors_tensor):
                placement_per_n_neigbors_tensor[n_neighbors] += p
            if p > 0:
                total_placements += p
            # Check if region has an enemy neighbor
            region = game.world.regions[ix]
            if any(game.get_owner(n) != self.agent_number for n in region.get_neighbours()):
                placements_next_to_enemy += p
        self.writer.add_histogram("Placements/region", placement_tensor, self.game_number)
        self.writer.add_histogram("Placements/n_neighbours", placement_per_n_neigbors_tensor, self.game_number)

        # Add percentage of placements next to enemies to total_rewards
        if total_placements > 0:
            self.total_rewards['placement_next_to_enemy_pct'] += placements_next_to_enemy / total_placements

        return ret

    @override
    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        per_node = True
        self.post_placement_node_features = torch.tensor(game.create_node_features(), dtype=torch.float)
        with torch.no_grad():
            placement_logits, attack_logits, army_logits = self.run_model(self.post_placement_node_features,
                                                                          action_edges=self.action_edges, action=Phase.ATTACK_TRANSFER)

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
        self.action_edges = torch.tensor([])

        self.writer.add_scalar('win', game.winning_player() == self.agent_number, self.game_number)
        self.writer.add_scalar('loss_mean', self.ppo_agent.loss_tracker.mean(), self.game_number)
        self.writer.add_scalar('loss_std', self.ppo_agent.loss_tracker.std(), self.game_number)

        self.writer.add_scalar('act_loss_mean', self.ppo_agent.act_loss_tracker.mean(), self.game_number)
        self.writer.add_scalar('act_loss_std', self.ppo_agent.act_loss_tracker.std(), self.game_number)

        self.writer.add_scalar('crit_loss_mean', self.ppo_agent.crit_loss_tracker.mean(), self.game_number)
        self.writer.add_scalar('crit_loss_std', self.ppo_agent.crit_loss_tracker.std(), self.game_number)

        self.writer.add_scalar('edge_entropy_mean', self.ppo_agent.edge_entropy_tracker.mean(), self.game_number)
        self.writer.add_scalar('edge_entropy_std', self.ppo_agent.edge_entropy_tracker.std(), self.game_number)

        self.writer.add_scalar('placement_entropy_mean', self.ppo_agent.placement_entropy_tracker.mean(), self.game_number)
        self.writer.add_scalar('placement_entropy_std', self.ppo_agent.placement_entropy_tracker.std(), self.game_number)

        self.writer.add_scalar('army_entropy_mean', self.ppo_agent.army_entropy_tracker.mean(), self.game_number)
        self.writer.add_scalar('army_entropy_std', self.ppo_agent.army_entropy_tracker.std(), self.game_number)

        self.writer.add_scalar('ratio_mean', self.ppo_agent.ratio_tracker.mean(), self.game_number)
        self.writer.add_scalar('ratio_std', self.ppo_agent.ratio_tracker.std(), self.game_number)

        self.writer.add_scalar('advantage_mean', self.ppo_agent.adv_tracker.mean(), self.game_number)
        self.writer.add_scalar('advantage_std', self.ppo_agent.adv_tracker.std(), self.game_number)

        self.writer.add_scalar('value_mean', self.ppo_agent.value_tracker.mean(), self.game_number)
        self.writer.add_scalar('value_std', self.ppo_agent.value_tracker.std(), self.game_number)

        self.writer.add_scalar('value_pred_mean', self.ppo_agent.value_pred_tracker.mean(), self.game_number)
        self.writer.add_scalar('value_pred_std', self.ppo_agent.value_pred_tracker.std(), self.game_number)

        self.writer.add_scalar('returns_mean', self.ppo_agent.returns_tracker.mean(), self.game_number)
        self.writer.add_scalar('returns_std', self.ppo_agent.returns_tracker.std(), self.game_number)

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

        self.total_rewards = defaultdict(int)
        self.game_number += 1

    @override
    def end_move(self, game: Game):
        if len(self.moves_this_turn) == 0 and not game.is_done():
            logging.debug("did no moves")
            return
        value = self.model.get_value(torch.tensor(self.post_placement_node_features, dtype=torch.float32)).detach()
        done = int(game.is_done())
        reward = self.compute_rewards(game)
        attacks = self.get_attacks()
        placements = self.get_placements()
        attacks_tensor = torch.tensor(attacks, dtype=torch.long)
        placements_tensor = torch.tensor(placements, dtype=torch.long)
        # Store transition in buffer
        self.buffer.add(
            self.action_edges,
            attacks,
            placements,
            compute_log_probs(attacks_tensor, self.attack_logits, self.army_logits, placements_tensor,
                              self.placement_logits, self.action_edges),
            reward,
            value,
            done,
            self.starting_node_features,
            self.post_placement_node_features
        )
        self.prev_state = PrevStateBuffer(prev_state=game, player_id=self.agent_number)

        if game.round % self.batch_size == 0 or done:
            with torch.no_grad():
                next_value = self.model.get_value(torch.tensor(self.post_placement_node_features, dtype=torch.float32)) * (1 - done)
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
            prev_regions = prev_state.regions
            prev_continents = prev_state.prev_continents
            prev_armies = prev_state.prev_armies
            prev_armies_enemy = prev_state.prev_armies_enemy
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
        for src_id, tgt_id in transfers:
            src_region = game.world.regions[src_id]
            tgt_region = game.world.regions[tgt_id]
            # Proximity before and after transfer
            prox_before = game.proximity_to_nearest_enemy(src_region)
            prox_after = game.proximity_to_nearest_enemy(tgt_region)
            if prox_after is not None and prox_before is not None:
                transfer_reward += (prox_before - prox_after) * 0.02  # Reward for moving closer to enemy
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

        # Overstacking penalty
        my_regions = game.regions_owned_by(player_id)
        overstack_reward = 0
        for region in my_regions:
            # If all neighbors are owned by the agent, it's a "safe" region
            if all(game.get_owner(n) == self.agent_number for n in region.get_neighbours()):
                overstack_reward -= 0.000001 * (game.get_armies(region) - 1)  # Tune this factor as needed

        # Scale down the overstack penalty to match other reward magnitudes

        reward += overstack_reward
        self.total_rewards['overstack_reward'] += overstack_reward

        if len(attacks) > 0:
            self.total_rewards['turn_with_attack'] += 1
        if len(attacks) > 1:
            self.total_rewards['turn_with_mult_attacks'] += 1

        # Multi-side attack reward
        attack_targets = defaultdict(set)
        for a in attacks:
            attack_targets[a[1]].add(a[0])

        multi_side_attack_reward = 0
        for tgt, srcs in attack_targets.items():
            if len(srcs) > 1:
                # Reward for each region attacked from multiple sources
                multi_side_attack_reward += 0.05 * (len(srcs) - 1)  # Tune as needed

        reward += multi_side_attack_reward


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
        self.total_rewards['multi_side_attack_reward'] += multi_side_attack_reward

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
        if len(self.action_edges) == 0:
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

            if src != tgt:
                ret.append((src.item(), tgt))
        return ret

    def sample_attacks_per_node(self):
        if len(self.action_edges) == 0:
            return []
        src_nodes = torch.unique(self.action_edges[:, 0])
        ret = []
        for src in src_nodes:
            mask = self.action_edges[:, 0] == src
            candidate_edges = self.action_edges[mask]
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
            mask = (self.action_edges[:, 0] == src) & (self.action_edges[:, 1] == tgt)

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
