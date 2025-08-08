import logging
import random
import time
import sys
import math
import hashlib
import json

from collections import defaultdict
from dataclasses import dataclass
from io import TextIOWrapper

from torch.utils.tensorboard import SummaryWriter


from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.agents.RLUtils.ModelFactory import ModelFactory
from src.game.FightSide import FightSide
from src.game.Phase import Phase
from src.config.training_config import TrainingConfig, ConfigFactory

if sys.version_info[1] < 11:
    from typing_extensions import override, Literal
else:
    from typing import override

import torch
import torch.nn.functional as f
from src.engine.AgentBase import AgentBase

from src.game.Game import Game
from src.game.Region import Region

from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies
from src.agents.RLUtils.RLUtils import RolloutBuffer, StatTracker, compute_individual_log_probs, PrevStateBuffer
from src.agents.RLUtils.PPOAgent import PPOAgent
from src.agents.RLUtils.PPOVerification import PPOVerifier

import faulthandler


do_hm_search = True

@dataclass
class RLGNNAgent(AgentBase):
    in_channels = 8
    hidden_channels = 64
    batch_size = 24  # Restored - root cause was edge masking inconsistency, not model drift
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    
    # Use factory to create model based on config
    config = ConfigFactory.create('sage_model')  # Use reduced army send configuration
    model = ModelFactory.create_model(
        model_type=config.model.model_type,
        node_feat_dim=config.model.in_channels,
        embed_dim=config.model.embed_dim,
        max_army_send=config.model.max_army_send
    ).to(device)

    placement_logits = torch.tensor([])
    attack_logits = torch.tensor([])
    army_logits = torch.tensor([])
    value = torch.tensor([])
    action_edges = torch.tensor([])
    
    # Store actual log probabilities for actions taken
    actual_placement_log_probs = torch.tensor([])
    actual_attack_log_probs = torch.tensor([])
    
    buffer = RolloutBuffer()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate)  # Use config learning rate
    ppo_agent = PPOAgent(model, optimizer, gamma=config.ppo.gamma, lam=config.ppo.lam, clip_eps=config.ppo.clip_eps,
                         ppo_epochs=config.ppo.ppo_epochs, adaptive_epochs=config.ppo.adaptive_epochs,
                         gradient_clip_norm=config.ppo.gradient_clip_norm, value_loss_coeff=config.ppo.value_loss_coeff,
                         value_clip_range=config.ppo.value_clip_range,
                         entropy_coeff_start=config.ppo.entropy_coeff_start,
                         entropy_coeff_decay=config.ppo.entropy_coeff_decay,
                         entropy_decay_episodes=config.ppo.entropy_decay_episodes,
                         placement_entropy_coeff=config.ppo.placement_entropy_coeff,
                         edge_entropy_coeff=config.ppo.edge_entropy_coeff,
                         army_entropy_coeff=config.ppo.army_entropy_coeff,
                         verification_config=config.verification)  # Enhanced PPO with entropy configuration
    starting_node_features: torch.Tensor = None
    post_placement_node_features: torch.Tensor= None

    moves_this_turn = []
    total_rewards = defaultdict(float)
    prev_state: PrevStateBuffer = None
    writer = SummaryWriter(log_dir=f"analysis/logs/{config.get_experiment_log_dir()}")  # Back to stable experiment

    game_number = 1
    num_attack_tracker = StatTracker()
    num_succes_attacks_tracker = StatTracker()
    army_per_attack_tracker = StatTracker()

    # histogram distributions
    placement_regions = []
    placement_neighbours = []
    
    # INPUT DATA VERIFICATION SYSTEM
    # This tracks inputs fed to run_model during single-sample inference
    # and verifies they match during batch inference
    _single_inference_data = {}  # game_round -> {phase -> {node_features, action_edges}}
    _batch_verification_enabled = False  # Disabled by default - can be enabled for debugging

    @property
    def device(self):
        return next(self.model.parameters()).device
    
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
        if hasattr(config.model, 'model_type') and config.model.model_type != self.config.model.model_type:
            print(f"🔄 Switching model architecture from {self.config.model.model_type} to {config.model.model_type}")
            
            # Create new model with the specified architecture
            new_model = ModelFactory.create_model(
                model_type=config.model.model_type,
                node_feat_dim=config.model.in_channels,
                embed_dim=config.model.embed_dim,
                max_army_send=config.model.max_army_send
            ).to(self.device)
            
            # Replace the model
            self.model = new_model
            
            # Recreate optimizer for new model parameters
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.ppo.learning_rate)
            
            # Recreate PPO agent with new model
            self.ppo_agent = PPOAgent(self.model, self.optimizer, gamma=config.ppo.gamma, lam=config.ppo.lam, clip_eps=config.ppo.clip_eps,
                                     ppo_epochs=config.ppo.ppo_epochs, adaptive_epochs=config.ppo.adaptive_epochs,
                                     gradient_clip_norm=config.ppo.gradient_clip_norm, value_loss_coeff=config.ppo.value_loss_coeff,
                                     value_clip_range=config.ppo.value_clip_range,
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
            self.ppo_agent.adaptive_epochs = config.ppo.adaptive_epochs
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
        
        # Apply the full training configuration
        self.apply_training_config(self.config)


    def run_model(self, node_features: torch.Tensor, action_edges: torch.Tensor = None, action = None):
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
            # BATCH INFERENCE VERIFICATION: Check if inputs match stored single-sample data
            if self._batch_verification_enabled and hasattr(self, '_single_inference_data'):
                self._verify_batch_inputs_match_single_samples(node_features, action_edges, action)
            return self._run_model_batch(node_features, action_edges, action)
        else:
            # SINGLE SAMPLE INFERENCE: Store input data for later verification
            if self._batch_verification_enabled:
                self._store_single_inference_data(node_features, action_edges, action)
            
            # Single sample case - original logic
            army_counts = node_features[:, -1].to(dtype=torch.float, device=self.device)
            graph = node_features.to(dtype=torch.float, device=self.device)

            if action_edges.numel() == 0 and action == Phase.ATTACK_TRANSFER:
                return torch.tensor(0), torch.tensor(0), torch.tensor(0)
            else:
                # CONSISTENCY FIX: Action edges are already padded/truncated in init_turn
                # Just create edge mask for valid (non-padded) edges
                edge_mask = (action_edges[:, 0] >= 0) & (action_edges[:, 1] >= 0)
                
                # Run model with padded edges - keep outputs at full size for consistency
                return self.model(graph, action_edges, army_counts, action, edge_mask)

    def _store_single_inference_data(self, node_features: torch.Tensor, action_edges: torch.Tensor, action: str):
        """Store input data from single-sample inference for later verification against batch inputs"""
        # Create a unique key for this inference call
        # Use phase as primary key and round as secondary key since batches are separated by phase
        phase_key = str(action)
        # Count how many times this phase has been called to determine the round number
        if phase_key not in self._single_inference_data:
            self._single_inference_data[phase_key] = {}
        round_key = len(self._single_inference_data[phase_key])  # Round number for this phase
        
        # Store deep copies of the input tensors
        self._single_inference_data[phase_key][round_key] = {
            'node_features': node_features.clone().detach(),
            'action_edges': action_edges.clone().detach() if action_edges is not None else None,
            'action': action
        }
        
        print(f"📝 STORED single inference data - Phase {phase_key}, Round {round_key}")
        print(f"   Node features shape: {node_features.shape}")
        print(f"   Action edges shape: {action_edges.shape if action_edges is not None else 'None'}")
        print(f"   Node features sample: {node_features[0, :3].detach().cpu().numpy()}")
        if action_edges is not None and action_edges.numel() > 0:
            print(f"   Action edges sample: {action_edges[:3].detach().cpu().numpy()}")

    def _verify_batch_inputs_match_single_samples(self, node_features: torch.Tensor, action_edges: torch.Tensor, action: str):
        """Verify that each sample in the batch matches previously stored single-sample data"""
        batch_size = node_features.size(0)
        phase_key = str(action)
        
        print(f"\n🔍 VERIFYING batch inputs match single samples")
        print(f"   Batch size: {batch_size}, Phase: {phase_key}")
        if phase_key in self._single_inference_data:
            print(f"   Available rounds for {phase_key}: {list(self._single_inference_data[phase_key].keys())}")
        else:
            print(f"   No data found for phase {phase_key}")
        
        mismatches_found = False
        
        for batch_idx in range(batch_size):
            # Each batch index should correspond to a round for this phase
            round_key = batch_idx
            
            if phase_key not in self._single_inference_data:
                print(f"   ❌ ERROR: No single inference data found for phase {phase_key}")
                mismatches_found = True
                continue
                
            if round_key not in self._single_inference_data[phase_key]:
                print(f"   ❌ ERROR: No single inference data found for phase {phase_key}, round {round_key}")
                mismatches_found = True
                continue
                
            stored_data = self._single_inference_data[phase_key][round_key]
            stored_node_features = stored_data['node_features']
            stored_action_edges = stored_data['action_edges']
            
            # Extract current batch sample
            batch_node_features = node_features[batch_idx]
            batch_action_edges = action_edges[batch_idx] if action_edges is not None else None
            
            # Ensure tensors are on the same device for comparison
            device = batch_node_features.device
            stored_node_features = stored_node_features.to(device)
            if stored_action_edges is not None:
                stored_action_edges = stored_action_edges.to(device)
            
            # Verify node features match
            if not torch.allclose(stored_node_features, batch_node_features, rtol=1e-5, atol=1e-6):
                print(f"   ❌ MISMATCH in episode {batch_idx} node features!")
                print(f"      Stored sample:  {stored_node_features[0, :3].detach().cpu().numpy()}")
                print(f"      Batch sample:   {batch_node_features[0, :3].detach().cpu().numpy()}")
                print(f"      Max difference: {(stored_node_features - batch_node_features).abs().max():.6f}")
                mismatches_found = True
            
            # Verify action edges match (if both exist)
            if stored_action_edges is not None and batch_action_edges is not None:
                if not torch.equal(stored_action_edges, batch_action_edges):
                    print(f"   ❌ MISMATCH in episode {batch_idx} action edges!")
                    print(f"      Stored shape: {stored_action_edges.shape}, Batch shape: {batch_action_edges.shape}")
                    print(f"      ISSUE: Single inference uses unpadded edges ({stored_action_edges.shape[0]} edges)")
                    print(f"             Batch inference uses padded edges ({batch_action_edges.shape[0]} edges)")
                    print(f"      This causes different model inputs and explains log prob differences!")
                    mismatches_found = True
            elif stored_action_edges is None and batch_action_edges is not None:
                print(f"   ❌ MISMATCH in episode {batch_idx}: stored edges were None but batch has edges")
                mismatches_found = True
            elif stored_action_edges is not None and batch_action_edges is None:
                print(f"   ❌ MISMATCH in episode {batch_idx}: stored edges exist but batch edges are None")
                mismatches_found = True
            
            if batch_idx < 3:  # Only print details for first few episodes
                print(f"   ✓ Episode {batch_idx} inputs match stored single inference data")
        
        if not mismatches_found:
            print(f"   ✅ ALL batch inputs match stored single inference data perfectly!")
        else:
            print(f"   💥 INPUT MISMATCHES DETECTED - This explains log probability differences!")
            
        # Clear stored data for this specific phase after verification to avoid memory buildup
        # Clear all rounds for this phase since they've been verified
        if phase_key in self._single_inference_data:
            del self._single_inference_data[phase_key]

    def _run_model_batch(self, node_features: torch.Tensor, action_edges: torch.Tensor, action: str = None):
        """
        Process batch by running model on each sample individually, then stack results

        Args:
            node_features: [batch_size, num_nodes, features]
            action_edges: [batch_size, num_edges, 2] (padded to 42 edges)
            action: Phase.PLACE_ARMIES or Phase.ATTACK_TRANSFER
        """
        device = node_features.device
        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)

        placement_logits_list = []
        attack_logits_list = []
        army_logits_list = []


        # Remove padding from all samples in batch
        valid_edge_masks = (action_edges[:, :, 0] >= 0) & (action_edges[:, :, 1] >= 0)  # [batch_size, 42]
        self.model.to(device)
        placement_logits, attack_logits, army_logits = self.model(
            node_features.to(dtype=torch.float, device=device),  # [batch_size, num_nodes, features]
            action_edges,  # [batch_size, 42, 2] (with padding)
            node_features[:, :, -1].to(dtype=torch.float, device=device),  # [batch_size, num_nodes]
            action,
            valid_edge_masks
            # Pass mask to model to ignore padded edges
        )
        if action == Phase.PLACE_ARMIES:
            placement_logits_list = [placement_logits[i] for i in range(batch_size)]
        else:
            attack_logits_list = [attack_logits[i] for i in range(batch_size)]
            army_logits_list = [army_logits[i] for i in range(batch_size)]

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
                max_army_send = self.config.model.max_army_send  # Use config value instead of hardcoded 50

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
        if self.model.edge_tensor is None:
            self.model.edge_tensor = torch.tensor(game.world.torch_edge_list, dtype=torch.long, device=self.device)
        
        original_action_edges = torch.tensor(game.create_action_edges(), dtype=torch.long, device=self.device)
        
        # CONSISTENCY FIX: Pad/truncate action edges to 42 to match batch inference
        if original_action_edges.size(0) < 42:
            padding_needed = 42 - original_action_edges.size(0)
            padding = torch.full((padding_needed, 2), -1, dtype=original_action_edges.dtype, device=original_action_edges.device)
            self.action_edges = torch.cat([original_action_edges, padding], dim=0)
        elif original_action_edges.size(0) > 42:
            self.action_edges = original_action_edges[:42]  # Truncate if too long
        else:
            self.action_edges = original_action_edges
            
        # Store the original size for output truncation
        self.original_num_edges = min(original_action_edges.size(0), 42)
        
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
        me = self.agent_number
        my_regions = game.regions_owned_by(me)
        with torch.no_grad():
            placement_logits, attack_logits, army_logits = self.run_model(self.starting_node_features,
                                                                          action_edges=self.action_edges,
                                                                          action=Phase.PLACE_ARMIES)
        self.placement_logits = placement_logits

        if len(my_regions) == 0:
            return []

        # Store the original placement logits BEFORE masking for later use in PPO
        self.original_placement_logits = placement_logits.clone() if hasattr(placement_logits, 'clone') else placement_logits
        
        # Now apply masking for action selection
        mine = [r.get_id() for r in my_regions]
        all_regions = set(range(len(game.world.regions)))
        not_mine = all_regions.difference(set(mine))
        available = game.armies_per_turn(me)
        if not self.placement_logits.is_leaf:
            self.placement_logits = self.placement_logits.detach()
        self.placement_logits[list(not_mine)] = float('-inf')  # Mask out regions not owned by the agent

        placement_probs = self.placement_logits.softmax(dim=0)
        try:
            nodes = torch.multinomial(
                placement_probs, num_samples=available, replacement=True
            )
        except RuntimeError as re:
            print(placement_probs)
            print(self.placement_logits)
            raise re
            
        # Compute actual log probabilities for the selected placements using log_softmax for numerical stability
        placement_log_probs_full = f.log_softmax(self.placement_logits, dim=0)
        # Store log probabilities in the same order as get_placements() will return them
        actual_placement_log_probs = []
        placement = torch.bincount(nodes, minlength=self.placement_logits.size(0))
        ret = []
        
        for ix, p in enumerate(placement.tolist()):
            if p > 0:
                # For each army placed in this region, add the log probability
                for _ in range(p):
                    actual_placement_log_probs.append(placement_log_probs_full[ix].item())
        self.actual_placement_log_probs = torch.tensor(actual_placement_log_probs)
        
        for ix, p in enumerate(placement.tolist()):
            if p > 0:
                ret.append(PlaceArmies(game.world.regions[ix], p))
        self.moves_this_turn += ret
        # After placements are determined
        placements_next_to_enemy = 0
        total_placements = 0
        for ix, p in enumerate(placement.tolist()):
            if p > 0:
                total_placements += p
            # Check if region has an enemy neighbor
            region = game.world.regions[ix]
            if any(game.get_owner(n) != self.agent_number for n in region.get_neighbours()):
                placements_next_to_enemy += p
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
                                                                          action_edges=self.action_edges,
                                                                          action=Phase.ATTACK_TRANSFER)

        self.attack_logits, self.army_logits = attack_logits, army_logits
        if per_node:
            edges = self.sample_attacks_per_node()
        else:
            edges = self.sample_n_attacks(game, 5)
        return self.create_attack_transfers(game, edges)

    def terminate(self, game: Game):
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
        # self.writer.add_histogram("Placements/region",
        #                           torch.tensor(self.placement_regions, dtype=torch.float16),
        #                           self.game_number)
        # self.writer.add_histogram("Placements/n_neighbours",
        #                           torch.tensor(self.placement_neighbours, dtype=torch.float16),
        #                           self.game_number)

        for key, value in self.total_rewards.items():
            if key in ['missed opportunities', 'missed transfers', 'turn_with_attack', 'turn_with_mult_attacks', 'num_regions', 'army_difference']:
                continue
            self.writer.add_scalar(key, value, self.game_number)

        self.total_rewards = defaultdict(int)
        self.game_number += 1

    @override
    def end_move(self, game: Game):
        if len(self.moves_this_turn) == 0 and not game.is_done():
            return
        end_features = torch.tensor(game.create_node_features(), dtype=torch.float32, device=self.device)
        value = self.model.get_value(end_features).detach()
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
            owned_regions
        )
        self.prev_state = PrevStateBuffer(prev_state=game, player_id=self.agent_number)

        if game.round % self.batch_size == 0 or done:
            next_value = value * (1 - done)
            self.ppo_agent.update(self.buffer, next_value, self)
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
            region_reward = gained_regions * 0.025 - lost_regions * 0.0125

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
            action_reward = 0.005
            action_reward += 0.01 * eff

        reward += action_reward

        # 5️⃣ Long-game penalty
        if game.round > 100:
            long_game_reward -= 0.01
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
        passivity_reward = 0
        if len(my_regions) > 0:
            passivity_reward = (curr_armies - armies_used - len(my_regions))/(curr_armies+len(my_regions)) * -0.003

        transfer_reward = 0
        for src_id, tgt_id in transfers:
            src_region = game.world.regions[src_id]
            tgt_region = game.world.regions[tgt_id]
            # Proximity before and after transfer
            prox_before = game.proximity_to_nearest_enemy(src_region)
            prox_after = game.proximity_to_nearest_enemy(tgt_region)
            if prox_after is not None and prox_before is not None:
                transfer_reward += (prox_before - prox_after) *  math.exp(-0.3 * prox_before) * 0.015  # Reward for moving closer to enemy
        reward += passivity_reward
        reward += transfer_reward
        placement_rewards = 0
        placements = self.get_placements(as_objects=True)
        good_placements = 0

        if len(my_regions) > 0:
            for p in placements:
                good_placements += 1 if any(
                    [n for n in p.region.get_neighbours() if game.get_owner(n) != self.agent_number]) else 0

            placement_rewards += ((good_placements * 0.05 - (len(placements) - good_placements) * 0.025) /
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
                multi_side_attack_reward += 0.1 * (len(srcs) - 1)  # Tune as needed

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
        # Use raw logits directly without temperature scaling
        probs = torch.softmax(self.attack_logits, dim=0)
        k = min(n, probs.size(0))
        topk_probs, selected_idxs = torch.topk(probs, k)

        # selected_idxs = (probs > (0.7 * probs.max())).nonzero(as_tuple=True)[0]
        ret = []
        if len(selected_idxs) == 0:
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
        
        # CONSISTENCY FIX: Only consider valid (non-padded) edges
        valid_edge_mask = (self.action_edges[:, 0] >= 0) & (self.action_edges[:, 1] >= 0)
        valid_action_edges = self.action_edges[valid_edge_mask]
        valid_attack_logits = self.attack_logits[valid_edge_mask]
        
        if len(valid_action_edges) == 0:
            return []
            
        src_nodes = torch.unique(valid_action_edges[:, 0])
        ret = []
        edge_selection_log_probs = []
        
        for src in src_nodes:
            mask = valid_action_edges[:, 0] == src
            candidate_edges = valid_action_edges[mask]
            candidate_logits = valid_attack_logits[mask]

            probs = f.softmax(candidate_logits, dim=-1)
            action_index = torch.multinomial(probs, 1).item()
            tgt = candidate_edges[action_index][1].item()
            
            # Store the log probability of this edge selection using log_softmax for numerical stability
            edge_log_prob = f.log_softmax(candidate_logits, dim=-1)[action_index].item()
            edge_selection_log_probs.append(edge_log_prob)
            
            if src != tgt:
                ret.append((src.item(), tgt))
        
        # Store edge selection log probabilities for later use in create_attack_transfers
        self.edge_selection_log_probs = torch.tensor(edge_selection_log_probs) if edge_selection_log_probs else torch.tensor([])
        return ret

    def create_attack_transfers(self, game: Game, edges):
        used_armies = defaultdict(int)
        ret = []
        n_attacks = 0
        n_army_attacks = 0
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
            
            available_armies = game.armies[src] - (
                used_armies[src]
            )  # leave one behind
            if available_armies <= 0:
                actual_attack_log_probs.append(0.0)  # No attack made
                continue

            # Choose how many armies to send using raw logits (no temperature scaling or noise)
            try:
                army_logit = self.army_logits[original_idx][:available_armies]
                army_probs = f.softmax(army_logit, dim=-1)
                if len(army_logit) == 0:
                    actual_attack_log_probs.append(0.0)  # No attack made
                    continue
            except IndexError as ie:
                print(self.army_logits)
                print(original_idx)
                print(available_armies)
                print(ie)
                raise ie
            try:
                k = int(torch.distributions.Categorical(probs=army_probs).sample().int())
            except IndexError as ie:
                print(army_logit)
                raise ie
            if k == 0:
                actual_attack_log_probs.append(0.0)  # No attack made
                continue
            used_armies[src] += int(k)
            if k >= available_armies:
                actual_attack_log_probs.append(0.0)  # No attack made
                continue

            # Compute the actual log probability for this attack using raw logits
            edge_log_prob = edge_log_probs[idx].item()
            army_log_prob = f.log_softmax(army_logit, dim=-1)[k].item()
            total_attack_log_prob = edge_log_prob + army_log_prob
            actual_attack_log_probs.append(total_attack_log_prob)

            ret.append(
                AttackTransfer(game.world.regions[src], game.world.regions[tgt], k, None)
            )
            if game.get_owner(tgt) != self.agent_number:
                n_attacks += 1
                n_army_attacks += k

        # Store the actual attack log probabilities
        self.actual_attack_log_probs = torch.tensor(actual_attack_log_probs) if actual_attack_log_probs else torch.tensor([])

        self.moves_this_turn += ret
        return ret
