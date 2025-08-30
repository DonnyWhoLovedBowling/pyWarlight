from email import policy
import random
import time
import sys
import math
import os
import traceback
from typing import Optional

from collections import defaultdict
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

from src.agents.RLUtils.CheckpointManager import CheckpointManager
from src.agents.RLUtils.WarlightModelAutoregressiveTransformer import WarlightPolicy
from src.game.FightSide import FightSide
from src.game.Phase import Phase
from src.config.training_config import TrainingConfig, ConfigFactory

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override

import torch
import torch.nn.functional as f
from src.engine.AgentBase import AgentBase

from src.game.Game import Game
from src.game.Region import Regionf

from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies
from src.agents.RLUtils.RLUtils import RolloutBuffer, compute_individual_log_probs, PrevStateBuffer
from src.agents.RLUtils.PPOAgent import PPOAgent
from src.agents.RLUtils.PPOVerification import PPOVerifier

import faulthandler


do_hm_search = True

@dataclass
class RLGNNAgent(AgentBase):
    in_channels = 7
    hidden_channels = 64
    batch_size = 24  # Restored - root cause was edge masking inconsistency, not model drift
    default_device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    
    # Use factory to create model based on config
    # Check for environment variable or use default
    config = ConfigFactory.create('transformer_edge_features')  # Use configurable training config
    
    model = WarlightPolicy(
        node_dim=config.model.in_channels,
        edge_dim=config.model.edge_feat_dim,
        hidden_dim=128,
        msg_depth=3,
        n_decoder_layers=3,
        n_heads=4,
        n_amount_bins=5,
        dropout=0.1,
        skip_residuals=False, # set True to remove residual connections
    )

    placement_logits = torch.tensor([])
    attack_logits = torch.tensor([])
    army_logits = torch.tensor([])
    value = torch.tensor([])
    action_edges = torch.tensor([])
    
    # Store actual log probabilities for actions taken
    actual_placement_log_probs = torch.tensor([])
    actual_attack_log_probs = torch.tensor([])
    
    buffer = RolloutBuffer()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ppo.learning_rate)  # Use config learning rate
    ppo_agent = PPOAgent(model, optimizer, gamma=config.ppo.gamma, lam=config.ppo.lam, clip_eps=config.ppo.clip_eps,
                         ppo_epochs=config.ppo.ppo_epochs,
                         gradient_clip_norm=config.ppo.gradient_clip_norm, value_loss_coeff=config.ppo.value_loss_coeff,
                         value_clip_range=config.ppo.value_clip_range, verbose_losses=config.logging.verbose_losses,
                         entropy_coeff_start=config.ppo.entropy_coeff_start,
                         entropy_coeff_decay=config.ppo.entropy_coeff_decay,
                         entropy_decay_episodes=config.ppo.entropy_decay_episodes,
                         placement_entropy_coeff=config.ppo.placement_entropy_coeff,
                         edge_entropy_coeff=config.ppo.edge_entropy_coeff,
                         army_entropy_coeff=config.ppo.army_entropy_coeff,
                         verification_config=config.verification)  # Enhanced PPO with entropy configuration
    starting_node_features: torch.Tensor = None
    post_placement_node_features: torch.Tensor= None
    edge_features: torch.Tensor = None

    moves_this_turn = []
    total_rewards = defaultdict(float)
    prev_state: PrevStateBuffer = None
    writer = SummaryWriter(log_dir=config.get_experiment_log_dir())  # Back to stable experiment

    game_number = 1

    # Checkpoint management
    checkpoint_manager: Optional[CheckpointManager] = None

    # histogram distributions
    placement_regions = []
    placement_neighbours = []
    
    # INPUT DATA VERIFICATION SYSTEM
    # This tracks inputs fed to run_model during single-sample inference
    # and verifies they match during batch inference
    _single_inference_data = {}  # game_round -> {phase -> {node_features, action_edges}}
    _batch_verification_enabled = False  # Disabled by default - can be enabled for debugging
    edge_tensor: Optional[torch.Tensor] = None

    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def set_config(self, config_name: str):
        """
        Set a new training configuration by name.
        
        Args:
            config_name: Name of the configuration to load (e.g., 'residual_percentage')
        """
        self.config_name = config_name
        new_config = ConfigFactory.create(config_name)
        self.config = new_config
        self.apply_training_config(new_config)
        print(f"🔄 Switched to configuration: {config_name}")

    def enable_verification(self, enable_batch_verification=True, enable_ppo_verification=True):
        """
        Enable optional verification systems for debugging.
        
        Args:
            enable_batch_verification: Enable input verification between single and batch inference
            enable_ppo_verification: Enable PPO training verification and diagnostics
        """
        self._batch_verification_enabled = enable_batch_verification
        self.ppo_agent.verifier.enabled = enable_ppo_verification
        if enable_batch_verification or enable_ppo_verification:
            print("🔍 Verification systems enabled for debugging")
    
    def apply_training_config(self, config: TrainingConfig):
        """
        Apply a comprehensive training configuration to the agent.
        
        Args:
            config: TrainingConfig containing all training parameters
        """
        # Update model if architecture changed
        if hasattr(config.model, 'model_type'):
            print(f"🔄 Switching model architecture from {self.config.model.model_type} to {config.model.model_type}")
            
            # Create new model with the specified architecture
            new_model = ModelFactory.create_model(
                config
            ).to(self.device)

            # Replace the model
            self.model = new_model
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.ppo.learning_rate)
            # Recreate PPO agent with new model
            self.ppo_agent = PPOAgent(self.model, self.optimizer, gamma=config.ppo.gamma, lam=config.ppo.lam, clip_eps=config.ppo.clip_eps,
                                     ppo_epochs=config.ppo.ppo_epochs,
                                     gradient_clip_norm=config.ppo.gradient_clip_norm, value_loss_coeff=config.ppo.value_loss_coeff,
                                     value_clip_range=config.ppo.value_clip_range, verbose_losses=config.logging.verbose_losses,
                                     entropy_coeff_start=config.ppo.entropy_coeff_start,
                                     entropy_coeff_decay=config.ppo.entropy_coeff_decay,
                                     entropy_decay_episodes=config.ppo.entropy_decay_episodes,
                                     placement_entropy_coeff=config.ppo.placement_entropy_coeff,
                                     edge_entropy_coeff=config.ppo.edge_entropy_coeff,
                                     army_entropy_coeff=config.ppo.army_entropy_coeff,
                                     verification_config=config.verification)
        else:
            # Update existing PPO configuration
            self.ppo_agent.gamma = config.ppo.gamma
            self.ppo_agent.lam = config.ppo.lam
            self.ppo_agent.clip_eps = config.ppo.clip_eps
            self.ppo_agent.ppo_epochs = config.ppo.ppo_epochs
            self.ppo_agent.gradient_clip_norm = config.ppo.gradient_clip_norm
            self.ppo_agent.value_loss_coeff = config.ppo.value_loss_coeff
            self.ppo_agent.value_clip_range = getattr(config.ppo, 'value_clip_range', None)
            
            # Update entropy configuration
            self.ppo_agent.entropy_coeff_start = config.ppo.entropy_coeff_start
            self.ppo_agent.entropy_coeff_decay = config.ppo.entropy_coeff_decay
            self.ppo_agent.entropy_decay_episodes = config.ppo.entropy_decay_episodes
            self.ppo_agent.placement_entropy_coeff = config.ppo.placement_entropy_coeff
            self.ppo_agent.edge_entropy_coeff = config.ppo.edge_entropy_coeff
            self.ppo_agent.army_entropy_coeff = config.ppo.army_entropy_coeff
            
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = config.ppo.learning_rate
        
        # Update batch size if it changed
        if hasattr(config.ppo, 'batch_size') and config.ppo.batch_size != self.batch_size:
            self.batch_size = config.ppo.batch_size
            print(f"📦 Updated batch size to {self.batch_size}")
        
        # Update verification configuration
        self._batch_verification_enabled = config.verification.batch_verification_enabled
        if config.verification.enabled:
            # Recreate verifier with new configuration
            self.ppo_agent.verifier = PPOVerifier(verification_config=config.verification)
        else:
            self.ppo_agent.verifier.enabled = False
        
        # Update logging configuration
        if hasattr(self, 'writer'):
            # Update the writer's log directory if needed
            experiment_log_dir = config.get_experiment_log_dir()
            if self.writer.log_dir != experiment_log_dir:
                self.writer.close()
                self.writer = SummaryWriter(log_dir=experiment_log_dir)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config, config.logging.experiment_name)
        self.ppo_agent.checkpoint_manager = self.checkpoint_manager
        
    
    def _load_checkpoint_if_requested(self, config):
        """Load checkpoint if requested in config"""
        if not self.checkpoint_manager:
            print("❌ Checkpoint manager not initialized")
            return
            
        checkpoint_path = None
        
        if config.logging.resume_from_checkpoint and config.logging.checkpoint_path:
            # Load specific checkpoint
            checkpoint_path = config.logging.checkpoint_path
            print(f"📂 Attempting to load specific checkpoint: {checkpoint_path}")
            
        elif config.logging.auto_resume_latest:
            # Find latest checkpoint for this experiment
            experiment_name = config.logging.resume_experiment_name or config.logging.experiment_name
            checkpoint_path = self.checkpoint_manager.find_latest_checkpoint(experiment_name)
            
            if checkpoint_path:
                print(f"📂 Auto-resuming from latest checkpoint: {os.path.basename(checkpoint_path)}")
            else:
                print(f"⚠️  No checkpoints found for experiment '{experiment_name}', starting fresh")
                return
        
        if checkpoint_path:
            # Configure what to load
            load_config = {
                "model": config.logging.load_model_state,
                "optimizer": config.logging.load_optimizer_state,
                "reward_normalizer": config.logging.load_reward_normalizer,
                "game_number": config.logging.load_game_number,
                "training_state": config.logging.load_training_state,
                "ppo_state": True
            }
            
            success = self.checkpoint_manager.load_checkpoint(self, checkpoint_path, load_config)
            if success:
                print(f"✅ Successfully resumed training from checkpoint!")
                print(f"   Continuing from game number: {self.game_number}")
            else:
                print(f"❌ Failed to load checkpoint, starting fresh training")
        
        # Store the new config
        self.config = config
        
        print(f"📝 Applied training configuration: {config.logging.experiment_name}")
        print(f"🏗️  Model architecture: {config.model.model_type}")
        print(config.summary())
    
    @override
    def is_rl_bot(self):
        return True

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())
        faulthandler.enable()
        self.apply_training_config(self.config)
        # Handle checkpoint resuming
        if self.config.logging.resume_from_checkpoint or self.config.logging.auto_resume_latest:
            self._load_checkpoint_if_requested(self.config)

        # Apply the full training configuration

    @override
    def init_turn(self, game: Game):
        if self.edge_tensor is None:
            self.edge_tensor = torch.tensor(game.world.torch_edge_list, dtype=torch.long, device=self.device)

        num_edges = self.edge_tensor.size(1)
        original_action_edges = torch.tensor(game.create_action_edges(), dtype=torch.long, device=self.device)

        # CONSISTENCY FIX: Pad/truncate action edges to 83 to match batch inference
        if original_action_edges.size(0) < num_edges:
            padding_needed = num_edges - original_action_edges.size(0)
            padding = torch.full((padding_needed, 2), -1, dtype=original_action_edges.dtype, device=original_action_edges.device)
            self.action_edges = torch.cat([original_action_edges, padding], dim=0)
            self.action_edges = torch.cat([original_action_edges, padding], dim=0)
        elif original_action_edges.size(0) > num_edges:
            self.action_edges = original_action_edges[:num_edges]  # Truncate if too long
        else:
            self.action_edges = original_action_edges
            
        # Store the original size for output truncation

        self.moves_this_turn = []
        self.starting_node_features = torch.tensor(game.create_node_features(), dtype= torch.float, device=self.device)
        self.starting_edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float, device=self.device)
        all_actions = self.model.sample_actions(
            node_feats=self.starting_node_features,
            edge_feats=self.starting_edge_features,
            edge_index=self.edge_tensor,
            legal_edge_src_ownership=self.action_edges,
            max_steps=self.config.model.max_attacks_per_turn,
            top_p=self.config.model.top_p,
        )
        self.create_attack_transfers(all_actions)
    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        chose = random.choice(choosable)
        return chose

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        self.init_turn(game)
        me = self.agent_number
        my_regions = game.regions_owned_by(me)
        self.starting_edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float, device=self.device)
        with torch.no_grad():
            placement_logits, _, _, _ = self.run_model(self.starting_node_features,
                                                        action_edges=self.action_edges,
                                                        action=Phase.PLACE_ARMIES,
                                                        edge_features=self.starting_edge_features
                                                    )
        self.placement_logits = placement_logits

        return ret

    @override
    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        per_node = self.config.model.per_node_attack_sampling
        self.post_placement_node_features = torch.tensor(game.create_node_features(), dtype=torch.float)
        self.edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float)
        with torch.no_grad():
            _, self.attack_logits, self.army_logits, _ = self.run_model(
                                                        self.post_placement_node_features,
                                                        action_edges=self.action_edges,
                                                        action=Phase.ATTACK_TRANSFER,
                                                        edge_features=self.edge_features
                                                        )

        if per_node:
            edges = self.sample_attacks_per_node(game)
        else:
            edges = self.sample_n_attacks(game)
        return self.create_attack_transfers(game, edges)

    def terminate(self, game: Game):
        self.end_move(game)
        self.action_edges = torch.tensor([])

        self.writer.add_scalar('win', game.winning_player() == self.agent_number, self.game_number)
        self.writer.add_scalar('attacks_per_turn', self.total_rewards['num_attacks'] / game.round , self.game_number)
        self.writer.add_scalar('won_battles_per_turn', self.total_rewards['won_battles'] / game.round, self.game_number)
        self.writer.add_scalar('armies_per_attack', self.total_rewards['armies_per_attack'] / game.round, self.game_number)
        self.writer.add_scalar('lost_regions', self.total_rewards['lost_regions'] / game.round, self.game_number)
        self.writer.add_scalar('gained_regions', self.total_rewards['gained_regions'] / game.round, self.game_number)
        self.writer.add_scalar('turn_with_attack', self.total_rewards['turn_with_attack'] / game.round, self.game_number)
        self.writer.add_scalar('turn_with_mult_attacks', self.total_rewards['turn_with_mult_attacks'] / game.round, self.game_number)
        self.writer.add_scalar('army_difference', self.total_rewards['army_difference'] / game.round, self.game_number)

        for key, value in self.total_rewards.items():
            if key in ['turn_with_attack', 'turn_with_mult_attacks', 'army_difference', 'num_attacks', 'lost_regions', 'armies_per_attack', 'won_battles' ]:
                continue
            self.writer.add_scalar(key, value, self.game_number)

        self.total_rewards = defaultdict(int)
        self.prev_state = None
        self.game_number += 1

    @override
    def end_move(self, game: Game):
        if len(self.moves_this_turn) == 0 and not game.is_done():
            return
        end_features = torch.tensor(game.create_node_features(), dtype=torch.float32, device=self.device)
        end_edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float32, device=self.device)
        value = self.model.get_value(end_features,end_edge_features, self.action_edges).detach()
        done = int(game.is_done())
        reward = self.compute_rewards(game)
        attacks = self.get_attacks()
        placements = self.get_placements()
        attacks_tensor = torch.tensor(attacks, dtype=torch.long)
        placements_tensor = torch.tensor(placements, dtype=torch.long)
        
        # Use the actual log probabilities captured during action selection
        if hasattr(self, 'actual_placement_log_probs') and len(self.actual_placement_log_probs) > 0:
            placement_log_probs = self.actual_placement_log_probs
        else:
            # Fallback to computing if not available
            placement_log_probs, _ = compute_individual_log_probs(
                attacks_tensor, self.attack_logits, self.army_logits, placements_tensor,
                self.placement_logits, self.action_edges
            )
            placement_log_probs = placement_log_probs.squeeze() if placement_log_probs.dim() > 1 else placement_log_probs
            
        # CONSISTENCY FIX: Always use compute_individual_log_probs for attack log probabilities
        # This ensures the same indexing and computation method as used during PPO update
        # The actual_attack_log_probs from action selection may have different indexing
        _, attack_log_probs = compute_individual_log_probs(
            attacks_tensor, self.attack_logits, self.army_logits, placements_tensor,
            self.placement_logits, self.action_edges
        )
        attack_log_probs = attack_log_probs.squeeze() if attack_log_probs.dim() > 1 else attack_log_probs
        if any((attack_log_probs < -20) & (attack_log_probs > -1e6)):
            print(f"⚠️  Warning: Extremely low attack log probabilities detected: {attack_log_probs}")
            print(f"   Attack logits: {self.attack_logits}")
            print(f"   Army logits: {self.army_logits}")
            print(f"   Attacks tensor: {attacks_tensor}")
            print(f"   Action edges: {self.action_edges}")
        # Store transition in buffer
        owned_regions = [r.get_id() for r in game.regions_owned_by(self.agent_number)] if hasattr(game, 'regions_owned_by') else None
        self.buffer.add(
            self.action_edges.clone(),  # Deep copy to prevent reference issues
            attacks,
            placements,
            placement_log_probs.clone() if isinstance(placement_log_probs, torch.Tensor) else placement_log_probs,  # Deep copy tensors
            attack_log_probs.clone() if isinstance(attack_log_probs, torch.Tensor) else attack_log_probs,  # Deep copy tensors
            reward,
            value,
            done,
            self.starting_node_features.clone(),  # Deep copy to prevent reference issues
            self.post_placement_node_features.clone(),  # Deep copy to prevent reference issues
            end_features.clone(),  # Deep copy to prevent reference issues
            owned_regions,
            self.starting_edge_features,
            self.edge_features,
            end_edge_features
        )
        self.prev_state = PrevStateBuffer(prev_state=game, player_id=self.agent_number)
        if self.buffer.size() % self.batch_size == 0:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            next_value = value * (1 - done)
            self.ppo_agent.update(self.buffer, next_value.to(device), self)
            self.buffer.clear()
            self.model.to('cpu') # Move model to CPU after update to save memory


    def compute_rewards(self, game: Game) -> float:
        prev_state = self.prev_state
        current_state = game
        player_id = self.agent_number
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
            region_reward = gained_regions * 0.05 - lost_regions * 0.025
            self.total_rewards['lost_regions'] += lost_regions
            self.total_rewards['gained_regions'] += gained_regions

            reward += region_reward

        # 2️⃣ Continent bonuses
        if prev_state is not None:
            continent_reward = (curr_continents - prev_continents) * 2
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
        self.total_rewards['num_attacks'] += len(attacks)
        wins = 0
        armies_used = 0
        destroyed_armies = 0
        for a in attacks:
            armies_used += a.armies
            destroyed_armies += a.result.defenders_destroyed
            if a.result.winner == FightSide.ATTACKER:
                wins += 1
            if self.game_number % 10 == 1:
                # Log action efficiency histogram every 10 games
                if a.is_attack():
                    print(f"attacked from {a.from_region} ({a.from_region.owner}) to {a.to_region}, ({a.to_region.owner}) with {a.armies} armies, winner: {a.result.winner}, ")

        if len(attacks) > 0:
            self.total_rewards['armies_per_attack'] += (armies_used / len(attacks))
            eff = destroyed_armies / armies_used
            action_reward = 0.005
            action_reward += 0.02 * eff

        self.total_rewards['won_battles'] += wins


        reward += action_reward

        # 5️⃣ Long-game penalty
        if game.round > 100:
            long_game_reward -= 0.01
            reward += long_game_reward

        if game.is_done():
            # Final win/loss reward
            if game.winning_player() == player_id:
                reward += 75.0 + max(0., 25. - 0.5 * game.round)
            elif game.winning_player() != -1:
                reward -= 50.0

        attacks = set(self.get_attacks(inc_transfers=False))
        attack_transfers = set(self.get_attacks(inc_transfers=True))
        transfers = attack_transfers.difference(attacks)
        my_regions = game.regions_owned_by(self.agent_number)
        if self.config.game.only_armies_used:
            reward = armies_used
        else:
            transfer_reward = 0
            for src_id, tgt_id, armies, available_armies in transfers:
                src_region = game.world.regions[src_id]
                tgt_region = game.world.regions[tgt_id]
                factor = armies/available_armies
                # Proximity before and after transfer
                prox_before = game.proximity_to_nearest_enemy(src_region)
                prox_after = game.proximity_to_nearest_enemy(tgt_region)
                if prox_after is not None and prox_before is not None:
                    transfer_reward += (prox_before - prox_after) * math.exp(
                        -0.3 * prox_before) * 0.005 * factor # Reward for moving closer to enemy
            reward += transfer_reward
            placement_rewards = 0
            placements = self.get_placements(as_objects=True)
            good_placements = 0

            if len(my_regions) > 0:
                for p in placements:
                    good_placements += 1 if any(
                        [n for n in p.region.get_neighbours() if game.get_owner(n) != self.agent_number]) else 0

                placement_rewards += ((good_placements * 0.02 - (len(placements) - good_placements) * 0.01) /
                                      len(my_regions))

            reward += placement_rewards

            # Overstacking penalty
            overstack_reward = 0
            for region in my_regions:
                # If all neighbors are owned by the agent, it's a "safe" region
                if all(game.get_owner(n) == self.agent_number for n in region.get_neighbours()):
                    overstack_reward -= 0.000005 * (game.get_armies(region) - 1)  # Tune this factor as needed

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
            self.total_rewards['num_regions'] = max(self.total_rewards['num_regions'], len(my_regions))
            self.total_rewards['region_reward'] += region_reward
            self.total_rewards['continent_reward'] += continent_reward
            self.total_rewards['action_reward'] += action_reward
            self.total_rewards['long_game_reward'] += long_game_reward
            self.total_rewards['army_reward'] += army_reward
            self.total_rewards['placement_reward'] += placement_rewards
            self.total_rewards['transfer_reward'] += transfer_reward
            self.total_rewards['reward'] += reward
            self.total_rewards['enemy_armies'] = enemy_armies

            self.total_rewards['multi_side_attack_reward'] += multi_side_attack_reward

        return reward

    def get_attacks(self, actions=None, object_data=False, inc_transfers=True) -> list[AttackTransfer | int]:
        if actions is None:
            actions = self.moves_this_turn
        if object_data:
            ret = [a for a in actions if (isinstance(a, AttackTransfer) and (inc_transfers or a.is_attack()))]
        else:
            ret = [(a.from_region.get_id(), a.to_region.get_id(), a.armies, a.available_armies) for a in actions if
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

    def create_attack_transfers(actions):
        for a in actions:
            
        used_armies = defaultdict(int)
        available_armies = {src: a - 1 for src, a in enumerate(game.armies)}
         
        ret = []
        actual_attack_log_probs = []
        
        # CONSISTENCY FIX: Only consider valid (non-padded) edges
        valid_edge_mask = (self.action_edges[:, 0] >= 0) & (self.action_edges[:, 1] >= 0)
        
        # Precompute edge log probabilities for efficiency (only for valid edges)
        edge_log_probs = f.log_softmax(self.attack_logits[valid_edge_mask], dim=0)

        for src, tgt in edges:
            # Find the index in the VALID edges
            valid_action_edges = self.action_edges[valid_edge_mask]
            mask = (valid_action_edges[:, 0] == src) & (valid_action_edges[:, 1] == tgt)

            indices = mask.nonzero(as_tuple=False)
            if len(indices) == 0:
                actual_attack_log_probs.append(0.0)  # No attack made
                continue
                
            idx = indices[0].item()  # Get first match
            
            # Find original index in the padded array
            original_indices = valid_edge_mask.nonzero(as_tuple=False).flatten()
            original_idx = original_indices[idx].item()
            
            if available_armies[src] <= 1:
                actual_attack_log_probs.append(0.0)  # No attack made
                continue

            # Choose how many armies to send using raw logits (no temperature scaling or noise)
            try:
                army_probs = f.softmax(self.army_logits[original_idx], dim=-1)
            except IndexError as ie:
                print(self.army_logits)
                print(original_idx)
                print(available_armies)
                print(ie)
                raise ie
            try:
                k = torch.multinomial(army_probs, num_samples=1).item()
            except IndexError as ie:
                raise ie
            used = round((float(k+1)/self.config.model.n_army_options)*(game.armies[src]-1))#

            if army_probs.numel() > 0 and self.game_number % 5 == 0 and game.round % 5 == 0:
                # Log placement probabilities histogram every 10 games
                for p in army_probs.cpu().detach().numpy():
                    self.writer.add_histogram('army_probs', p, self.game_number)
                # print(f"Using {used} armies for attack from {src} to {tgt} (available: {available_armies[src]}) tgt is owned by {game.get_owner(tgt)}")
                # print(f"Army logits: {self.army_logits[original_idx]}")
                # print(f"Army probs: {army_probs}")

            if used == 0 or used >= available_armies[src]:
                actual_attack_log_probs.append(0.0)  # No attack made
                continue
            available_armies[src] -= int(used)
            # Compute the actual log probability for this attack using raw logits
            edge_log_prob = edge_log_probs[idx].item()
            army_log_prob = f.log_softmax(self.army_logits[original_idx], dim=-1)[k].item()
            total_attack_log_prob = edge_log_prob + army_log_prob
            actual_attack_log_probs.append(total_attack_log_prob)

            ret.append(
                AttackTransfer(game.world.regions[src], game.world.regions[tgt], used, None, game.armies[src])
            )

        # Store the actual attack log probabilities
        self.actual_attack_log_probs = torch.tensor(actual_attack_log_probs) if actual_attack_log_probs else torch.tensor([])

        self.moves_this_turn += ret
        return ret
