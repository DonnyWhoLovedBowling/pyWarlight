import random
import time
import sys
import math
import os
from typing import Optional

from collections import defaultdict
from dataclasses import dataclass, field

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
from src.game.Region import Region

from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies
from src.agents.RLUtils.RLUtils import RolloutBuffer, PrevStateBuffer
from src.agents.RLUtils.PPOAgent import PPOAgent
from src.agents.RLUtils.PPOVerification import PPOVerifier

import faulthandler


do_hm_search = True

@dataclass
class \
        RLGNNAgent(AgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_ppo_update_time = None
        self.writer = SummaryWriter(log_dir=self.config.get_experiment_log_dir())  # Back to stable experiment

    batch_size = 24  # Restored - root cause was edge masking inconsistency, not model drift
    default_device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    
    # Use factory to create model based on config
    # Check for environment variable or use default
    config = ConfigFactory.create('transformer_edge_features')  # Use configurable training config
    
    model = WarlightPolicy(
        node_dim=config.model.in_channels,
        edge_dim=config.model.edge_feat_dim,
        num_nodes=42,
        num_edges=166,
        hidden_dim=128,
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
    ppo_agent = PPOAgent(model, optimizer, config.ppo, config.verification)  # Enhanced PPO with entropy configuration

    moves_this_turn = []
    total_rewards = defaultdict(float)
    prev_state: Optional[PrevStateBuffer] = None

    game_number = 1

    # Checkpoint management
    checkpoint_manager: Optional[CheckpointManager] = None

    # histogram distributions
    placement_regions = []
    placement_neighbours = []
    
    # INPUT DATA VERIFICATION SYSTEM
    # This tracks inputs fed to run_model during single-sample inference
    # and verifies they match during batch inference
    edge_tensor: torch.Tensor = torch.tensor([], dtype=torch.long)
    model_output: dict = field(default_factory=dict)
    ownership_mask: torch.Tensor = torch.tensor([], dtype=torch.bool)
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property
    def inference_device(self):
        """Get the device for inference operations"""
        return self.config.model.get_inference_device()

    @property
    def training_device(self):
        """Get the device for training operations"""
        return self.config.model.get_training_device()

    def _move_model_to_device(self, device: torch.device):
        """Move model to specified device"""
        self.model.to(device)

    def _prepare_tensors_for_inference(self, *tensors):
        """Move tensors to inference device with optional pinned memory"""
        device = self.inference_device
        if self.config.model.pin_memory and device.type == 'cuda':
            return tuple(t.pin_memory().to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in tensors)
        else:
            return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tensors)

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
            
            # # Create new model with the specified architecture
            # new_model = ModelFactory.create_model(
            #     config
            # ).to(self.device)

            # # Replace the model
            # self.model = new_model
            # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.ppo.learning_rate)
            # Recreate PPO agent with new model
            self.ppo_agent = PPOAgent(self.model, self.optimizer, config.ppo,
                                     verification_config=config.verification)
        else:
            # Update existing PPO configuration
            self.ppo_agent.gamma = config.ppo.gamma
            self.ppo_agent.lam = config.ppo.lam
            self.ppo_agent.clip_eps = config.ppo.clip_eps
            self.ppo_agent.ppo_epochs = config.ppo.ppo_epochs
            self.ppo_agent.gradient_clip_norm = config.ppo.gradient_clip_norm
            self.ppo_agent.value_loss_coeff = config.ppo.value_loss_coeff
            
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
        
        # Debug: Show current game_number before loading
        print(f"🔍 Current game_number before checkpoint load: {self.game_number}")
            
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
                
                # Verify game_number was loaded correctly
                if load_config.get("game_number", True) and self.game_number == 1:
                    print(f"   ⚠️  WARNING: game_number is still 1 after loading! Check checkpoint data.")
            else:
                print(f"❌ Failed to load checkpoint, starting fresh training")
        
        # Store the new config
        self.config = config
        
        print(f"📝 Applied training configuration: {config.logging.experiment_name}")
        print(f"🏗️  Model architecture: {config.model.model_type}")
        print(config.summary())
    
    @override
    def is_rl_bot(self) -> bool:
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
        
        if self.model.edge_index is None:
            self.model.edge_index = torch.tensor(game.world.torch_edge_list, dtype=torch.long, device=self.device)
            self.edge_tensor = self.model.edge_index
        num_edges = self.model.edge_index.size(1)
            
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
        self.ownership_mask = torch.tensor([1 if r.owner == self.agent_number else 0 for r in game.world.regions], dtype=torch.bool, device=self.device)
        self.buffer.add_init_vars(self.edge_tensor, self.ownership_mask)
        self.edge_tensor = self.model.edge_index
        
    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        chose = random.choice(choosable)
        return chose

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        self.init_turn(game)
        me = self.agent_number

        # Move model to inference device if needed
        if self.config.model.should_move_model_for_inference():
            self._move_model_to_device(self.inference_device)

        # Prepare tensors for inference
        starting_node_features = torch.tensor(game.create_node_features(), dtype=torch.float)
        starting_edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float)
        starting_node_features, starting_edge_features = self._prepare_tensors_for_inference(
            starting_node_features, starting_edge_features
        )

        with torch.no_grad():
            output = self.model(
                starting_node_features,
                starting_edge_features,
                self.ownership_mask,
                n_available_armies=game.armies_per_turn(me),
                armies_left=torch.tensor([]),
                action=Phase.PLACE_ARMIES
            )

        ret = []
        region_map = defaultdict(int)
        for i, region_idx in enumerate(output['placements_list']):
            region_map[region_idx] += 1
        for r, n in region_map.items():
            region = game.world.regions[r]
            ret.append(PlaceArmies(region=region, armies=n))
        self.moves_this_turn += ret

        # Store on CPU for buffer (environment is on CPU)
        self.buffer.add_placements(
            output['placements_list'],
            output['logp'].cpu() if isinstance(output['logp'], torch.Tensor) else output['logp'],
            starting_node_features.cpu(),
            starting_edge_features.cpu()
        )

        return ret

    @override
    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        # Prepare tensors for inference
        post_placement_node_features = torch.tensor(game.create_node_features(), dtype=torch.float)
        post_placement_edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float)
        armies_left_tensor = torch.tensor([game.get_armies(r) - 1 for r in game.world.regions], dtype=torch.long)

        # Move tensors to inference device
        post_placement_node_features, post_placement_edge_features, armies_left_tensor = self._prepare_tensors_for_inference(
            post_placement_node_features, post_placement_edge_features, armies_left_tensor
        )

        with torch.no_grad():
            model_output = self.model(
                post_placement_node_features,
                post_placement_edge_features,
                self.ownership_mask,
                n_available_armies=torch.tensor([]),
                armies_left=armies_left_tensor,
                action=Phase.ATTACK_TRANSFER
            )

        ret = []
        if 'attacks' in model_output and 'edge_idx' in model_output['attacks']:
            for i, e in enumerate(model_output['attacks']['edge_idx']):
                from_region = game.world.regions[self.edge_tensor[0][e].item()]
                to_region = game.world.regions[self.edge_tensor[1][e].item()]
                ret.append(AttackTransfer(from_region=from_region,
                                          to_region=to_region,
                                          armies=model_output['attacks']['army_count'][i]))
            self.actual_attack_log_probs = model_output['logp']
        else:
            print("⚠️  model_output['attacks']['edge_idx'] is not available or model_output is None.")

        self.moves_this_turn += ret

        # Store on CPU for buffer (environment is on CPU)
        self.buffer.add_attacks(
            model_output['attacks'],
            self.actual_attack_log_probs.cpu() if isinstance(self.actual_attack_log_probs, torch.Tensor) else self.actual_attack_log_probs,
            post_placement_node_features.cpu(),
            post_placement_edge_features.cpu(),
            armies_left_tensor.cpu()
        )
        return ret


    def terminate(self, game: Game):
        self.action_edges = torch.tensor([])
        self.end_move(game)
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

            # Debug: Print entropy-related values being logged to TensorBoard
            if 'entropy' in key.lower():
                print(f"[TENSORBOARD] DEBUG: Logging {key} = {value}")

            self.writer.add_scalar(key, value, self.game_number)

        self.total_rewards = defaultdict(int)
        self.prev_state = None
        self.game_number += 1

        # Save checkpoint if needed
        if self.checkpoint_manager and self.config.logging.save_checkpoints:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(self, self.game_number)
            if checkpoint_path:
                print(f"💾 Checkpoint saved at game {self.game_number}")

        self.writer.flush()
        # Do not close the writer here; only flush. Closing should be done when the agent is truly finished.

    def finalize(self):
        """Call this when the agent is truly finished (e.g., at the end of all games)."""
        self.writer.close()

    @override
    def end_move(self, game: Game):
        if game.is_done() and len(self.buffer.values) == len(self.buffer.post_placement_edge_features):
            return
        end_features = torch.tensor(game.create_node_features(), dtype=torch.float32)
        end_edge_features = torch.tensor(game.create_edge_features(), dtype=torch.float32)

        # Move tensors to inference device for value computation
        end_features, end_edge_features = self._prepare_tensors_for_inference(end_features, end_edge_features)

        value = self.model.get_value(end_features, end_edge_features).squeeze()
        done = int(game.is_done())
        reward = self.compute_rewards(game)
        
        # Store transition in buffer (on CPU for environment)
        self.buffer.add_end_vars(

            reward,
            value.cpu() if isinstance(value, torch.Tensor) else value,
            done,
            end_features.cpu(),
            end_edge_features.cpu()
        )
        
        self.prev_state = PrevStateBuffer(prev_state=game, player_id=self.agent_number)

        if self.buffer.size() % self.batch_size == 0:
            # Move model to training device before PPO update
            self._move_model_to_device(self.training_device)

            # Log wall time since last PPO update
            if self.last_ppo_update_time is not None:
                elapsed = time.time() - self.last_ppo_update_time
                print(f"[PPO] Elapsed wall time since last PPO update: {elapsed:.4f} seconds")
            else:
                print("[PPO] This is the first PPO update.")

            next_value = value.detach() * (1 - done)
            self.ppo_agent.update(self.buffer, next_value.to(self.training_device), self)
            self.last_ppo_update_time = time.time()
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
        self._consecutive_attack_wins = 0

        if prev_state is not None:
            # 1️⃣ Region control - using configurable values
            gained_regions = len(curr_regions.difference(prev_regions))
            lost_regions = len(prev_regions.difference(curr_regions))
            region_reward = (gained_regions * self.config.game.region_gain_reward -
                           lost_regions * self.config.game.region_loss_penalty)
            self.total_rewards['lost_regions'] += lost_regions
            self.total_rewards['gained_regions'] += gained_regions

            reward += region_reward

        # 2️⃣ Continent bonuses - using configurable multiplier
        if prev_state is not None:
            continent_reward = (curr_continents - prev_continents) * self.config.game.continent_bonus_multiplier
            reward += continent_reward

        # 3️⃣ Army dynamics - using configurable weight
        if prev_state is not None:
            armies_destroyed = max(0, prev_armies_enemy - curr_armies_enemy)
            armies_lost = max(0, prev_armies - curr_armies)

            diff = self.config.game.army_destruction_weight * (armies_destroyed - armies_lost)
        else:
            diff = 0

        if self.config.game.army_efficiency_normalization:
            normalized_army_delta = diff / (curr_armies + curr_armies_enemy + 1e-8)
            army_reward = normalized_army_delta
        else:
            army_reward = diff

        reward += army_reward

        # 4️⃣ Action dynamics - using configurable rewards
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

        if len(attacks) > 0:
            self.total_rewards['armies_per_attack'] += (armies_used / len(attacks))
            eff = destroyed_armies / armies_used if armies_used > 0 else 0
            action_reward = self.config.game.action_base_reward
            action_reward += self.config.game.action_efficiency_multiplier * eff

        self.total_rewards['won_battles'] += wins

        reward += action_reward

        # 5️⃣ Long-game penalty - using configurable threshold and penalty
        if game.round > self.config.game.long_game_threshold:
            long_game_reward -= self.config.game.long_game_penalty
            reward += long_game_reward

        if game.is_done():
            # Final win/loss reward - using configurable values
            if game.winning_player() == player_id:
                reward += self.config.game.win_reward + max(
                    0., self.config.game.win_speed_bonus_max -
                    self.config.game.win_speed_decay * game.round
                )
            elif game.winning_player() != -1:
                reward -= self.config.game.loss_penalty

        attacks = set(self.get_attacks(inc_transfers=False))
        attack_transfers = set(self.get_attacks(inc_transfers=True))
        transfers = attack_transfers.difference(attacks)
        my_regions = game.regions_owned_by(self.agent_number)

        if self.config.game.only_armies_used:
            reward = armies_used
        else:
            # 7️⃣ Transfer rewards - using configurable multiplier and decay
            transfer_reward = 0
            for src_id, tgt_id, armies, available_armies in transfers:
                src_region = game.world.regions[src_id]
                tgt_region = game.world.regions[tgt_id]
                factor = armies/available_armies
                prox_before = game.proximity_to_nearest_enemy(src_region)
                prox_after = game.proximity_to_nearest_enemy(tgt_region)
                if prox_after is not None and prox_before is not None:
                    transfer_reward += (prox_before - prox_after) * math.exp(
                        -self.config.game.transfer_proximity_decay * prox_before
                    ) * self.config.game.transfer_proximity_multiplier * factor
            reward += transfer_reward

            # 8️⃣ Placement rewards - using configurable bonuses and penalties
            placement_rewards = 0
            placements = self.get_placements(as_objects=True)
            good_placements = 0

            if len(my_regions) > 0:
                for p in placements:
                    good_placements += 1 if any(
                        [n for n in p.region.get_neighbours() if game.get_owner(n) != self.agent_number]) else 0

                if self.config.game.placement_normalization_factor:
                    placement_rewards += (
                        (good_placements * self.config.game.placement_next_to_enemy_bonus -
                         (len(placements) - good_placements) * self.config.game.placement_safe_penalty) /
                        len(my_regions)
                    )
                else:
                    placement_rewards += (
                        good_placements * self.config.game.placement_next_to_enemy_bonus -
                        (len(placements) - good_placements) * self.config.game.placement_safe_penalty
                    )

            reward += placement_rewards

            # 9️⃣ Overstacking penalty - using configurable rate
            overstack_reward = 0
            for region in my_regions:
                if all(game.get_owner(n) == self.agent_number for n in region.get_neighbours()):
                    overstack_reward -= self.config.game.overstack_penalty_rate * (game.get_armies(region) - 1)

            reward += overstack_reward
            self.total_rewards['overstack_reward'] += overstack_reward

            if len(attacks) > 0:
                self.total_rewards['turn_with_attack'] += 1
            if len(attacks) > 1:
                self.total_rewards['turn_with_mult_attacks'] += 1

            # 🔟 Multi-side attack reward - using configurable bonus
            attack_targets = defaultdict(set)
            for a in attacks:
                attack_targets[a[1]].add(a[0])

            multi_side_attack_reward = 0
            for tgt, srcs in attack_targets.items():
                if len(srcs) > 1:
                    multi_side_attack_reward += self.config.game.multi_side_attack_bonus * (len(srcs) - 1)

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
