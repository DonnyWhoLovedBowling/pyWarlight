"""
PPO Training Verification Module

This module contains optional verification utilities for PPO training.
It can be enabled/disabled via configuration flags to keep the main PPO update loop clean.
"""

import torch
import torch.nn.functional as f
from src.game.Phase import Phase
from src.agents.RLUtils.RLUtils import apply_placement_masking


class PPOVerifier:
    """
    Optional verification system for PPO training to detect input mismatches,
    NaN/Inf values, and consistency issues between inference and training phases.
    """
    
    def __init__(self, verification_config=None):
        """
        Initialize verifier with configuration
        
        Args:
            verification_config: VerificationConfig object, or None to disable all verification
        """
        if verification_config is None:
            # Create a minimal disabled config
            self.enabled = False
            self.tolerance = 1e-5
            self.config = None
        else:
            self.enabled = verification_config.enabled
            self.tolerance = verification_config.tolerance
            self.config = verification_config
    
    def _should_run(self, check_name: str) -> bool:
        """Check if a specific verification should run"""
        if not self.enabled or self.config is None:
            return False
        return getattr(self.config, check_name, False)
    
    def _safe_max_diff(self, tensor1, tensor2):
        """Safely compute max difference handling NaN/Inf values"""
        if tensor1.shape != tensor2.shape:
            return float('inf')  # Shape mismatch
        
        diff = (tensor1 - tensor2).abs()
        
        # Check for NaN/Inf in the difference
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            # Return NaN if either tensor contains NaN/Inf
            if torch.isnan(tensor1).any() or torch.isnan(tensor2).any():
                return float('nan')
            if torch.isinf(tensor1).any() or torch.isinf(tensor2).any():
                return float('inf')
        
        # Filter out NaN/Inf values for max calculation
        valid_diff = diff[~(torch.isnan(diff) | torch.isinf(diff))]
        if valid_diff.numel() == 0:
            return float('nan')  # All values were NaN/Inf
        
        return valid_diff.max()
    
    def verify_structural_integrity(self, starting_features_batched, post_features_batched, action_edges_batched, epoch):
        """Verify the structural integrity of batched inputs"""
        if not self._should_run('verify_structural_integrity'):
            return
        
        if self.config.detailed_logging:
            print(f"\n=== STRUCTURAL VERIFICATION (Epoch {epoch}) ===")
            print(f"Starting features shape: {starting_features_batched.shape}")
            print(f"Post features shape: {post_features_batched.shape}")
            print(f"Action edges shape: {action_edges_batched.shape}")
        
        # Verify that each episode's data matches what was stored during action selection
        num_to_check = min(self.config.print_first_n_episodes, starting_features_batched.size(0))
        for i in range(num_to_check):
            if self.config.detailed_logging:
                print(f"\nEpisode {i} verification:")
                print(f"  Starting features sample: {starting_features_batched[i, 0, :3].cpu().numpy()}")
                print(f"  Post features sample: {post_features_batched[i, 0, :3].cpu().numpy()}")
                print(f"  First 5 action edges: {action_edges_batched[i, :5].cpu().numpy()}")
            
            # Check for any NaN or inf values that could cause issues
            if torch.isnan(starting_features_batched[i]).any():
                print(f"  WARNING: NaN detected in starting features for episode {i}")
            if torch.isinf(starting_features_batched[i]).any():
                print(f"  WARNING: Inf detected in starting features for episode {i}")
            if torch.isnan(post_features_batched[i]).any():
                print(f"  WARNING: NaN detected in post features for episode {i}")
            if torch.isinf(post_features_batched[i]).any():
                print(f"  WARNING: Inf detected in post features for episode {i}")
        
        # Assert that inputs have expected properties
        assert starting_features_batched.dtype == torch.float32, f"Expected float32, got {starting_features_batched.dtype}"
        assert post_features_batched.dtype == torch.float32, f"Expected float32, got {post_features_batched.dtype}"
        assert action_edges_batched.dtype == torch.long, f"Expected long, got {action_edges_batched.dtype}"
        assert action_edges_batched.size(1) == 42, f"Expected 42 edges, got {action_edges_batched.size(1)}"
        assert not torch.isnan(starting_features_batched).any(), "NaN found in starting features"
        assert not torch.isnan(post_features_batched).any(), "NaN found in post features"
        
        if self.config.print_verification_summary and self.config.detailed_logging:
            print("✓ All structural verifications passed")
    
    def verify_model_outputs(self, placement_logits, attack_logits, army_logits):
        """Verify model outputs for NaN/Inf values and basic sanity checks"""
        if not self._should_run('verify_model_outputs'):
            return
            
        # VERIFICATION: Check placement logits output
        if self.config.detailed_logging:
            print(f"Placement logits shape: {placement_logits.shape}")
            print(f"Placement logits sample: {placement_logits[0, :3].detach().cpu().numpy()}")
        assert not torch.isnan(placement_logits).any(), "NaN found in placement logits after model run"
        
        # VERIFICATION: Check attack logits output
        if self.config.detailed_logging:
            print(f"Attack logits shape: {attack_logits.shape}")
            print(f"Attack logits sample: {attack_logits[0, :3].detach().cpu().numpy()}")
            print(f"Army logits shape: {army_logits.shape}")
        assert not torch.isnan(attack_logits).any(), "NaN found in attack logits after model run"
        assert not torch.isnan(army_logits).any(), "NaN found in army logits after model run"
    
    def verify_single_vs_batch_inference(self, agent, starting_features_batched, post_features_batched, 
                                       action_edges_batched, placement_logits, attack_logits, army_logits, buffer, epoch):
        """
        Critical test: Simulate single-episode inference to verify identical outputs.
        This tests that run_model produces identical results when called with the same inputs
        as during the original action selection phase.
        
        Only runs in the first epoch since model parameters change after training.
        """
        if not self._should_run('verify_single_vs_batch') or starting_features_batched.size(0) == 0:
            return
        
        # Only verify in the first epoch since model parameters change after training
        if epoch > 0:
            return
            
        if self.config.detailed_logging:
            print(f"\n=== SINGLE-EPISODE INFERENCE SIMULATION ===")
        test_episode_idx = 0  # Test the first episode
        
        # Extract single episode data
        single_start_features = starting_features_batched[test_episode_idx]
        single_post_features = post_features_batched[test_episode_idx]
        single_action_edges = action_edges_batched[test_episode_idx]
        
        if self.config.detailed_logging:
            print(f"Testing episode {test_episode_idx}:")
            print(f"  Input shapes - start: {single_start_features.shape}, post: {single_post_features.shape}, edges: {single_action_edges.shape}")
        
        # Simulate what happens during place_armies() inference (get raw logits)
        with torch.no_grad():
            single_placement_logits_raw, _, _ = agent.run_model(
                node_features=single_start_features.unsqueeze(0),  # Add batch dimension
                action_edges=single_action_edges.unsqueeze(0),     # Add batch dimension  
                action=Phase.PLACE_ARMIES
            )
            single_placement_logits_raw = single_placement_logits_raw.squeeze(0)  # Remove batch dimension
        
        # Simulate what happens during attack_transfer() inference
        with torch.no_grad():
            _, single_attack_logits, single_army_logits = agent.run_model(
                node_features=single_post_features.unsqueeze(0),   # Add batch dimension
                action_edges=single_action_edges.unsqueeze(0),     # Add batch dimension
                action=Phase.ATTACK_TRANSFER
            )
            single_attack_logits = single_attack_logits.squeeze(0)  # Remove batch dimension
            single_army_logits = single_army_logits.squeeze(0)     # Remove batch dimension
        
        # Compare with batched outputs for the same episode
        batch_placement_logits = placement_logits[test_episode_idx]
        batch_attack_logits = attack_logits[test_episode_idx]
        batch_army_logits = army_logits[test_episode_idx]
        
        if self.config.detailed_logging:
            print(f"  Single inference shapes - placement: {single_placement_logits_raw.shape}, attack: {single_attack_logits.shape}, army: {single_army_logits.shape}")
            print(f"  Batch inference shapes - placement: {batch_placement_logits.shape}, attack: {batch_attack_logits.shape}, army: {batch_army_logits.shape}")
        
        # Verify identical results with NaN handling
        placement_diff = self._safe_max_diff(single_placement_logits_raw, batch_placement_logits)
        attack_diff = self._safe_max_diff(single_attack_logits, batch_attack_logits)
        army_diff = self._safe_max_diff(single_army_logits, batch_army_logits)
        
        if self.config.detailed_logging:
            print(f"  Max differences - placement: {placement_diff:.6f}, attack: {attack_diff:.6f}, army: {army_diff:.6f}")
        
        # Check for NaN/Inf values in placement logits
        single_has_nan = torch.isnan(single_placement_logits_raw).any() or torch.isinf(single_placement_logits_raw).any()
        batch_has_nan = torch.isnan(batch_placement_logits).any() or torch.isinf(batch_placement_logits).any()
        
        if single_has_nan or batch_has_nan:
            print(f"  🚨 NaN/Inf detected in placement logits - Single: {single_has_nan}, Batch: {batch_has_nan}")
            if self.config.detailed_logging:
                print(f"    Single NaN count: {torch.isnan(single_placement_logits_raw).sum()}")
                print(f"    Single Inf count: {torch.isinf(single_placement_logits_raw).sum()}")
                print(f"    Batch NaN count: {torch.isnan(batch_placement_logits).sum()}")
                print(f"    Batch Inf count: {torch.isinf(batch_placement_logits).sum()}")
        
        # These should be identical (or very close due to floating point precision)
        if placement_diff > self.tolerance and not torch.isnan(torch.tensor(placement_diff)):
            print(f"  ERROR: Placement logits differ by {placement_diff:.6f} > {self.tolerance}")
            if self.config.detailed_logging:
                print(f"    Single sample: {single_placement_logits_raw[:5].detach().cpu().numpy()}")
                print(f"    Batch sample: {batch_placement_logits[:5].detach().cpu().numpy()}")
        
        if attack_diff > self.tolerance and not torch.isnan(torch.tensor(attack_diff)):
            print(f"  ERROR: Attack logits differ by {attack_diff:.6f} > {self.tolerance}")
            if self.config.detailed_logging:
                print(f"    Single sample: {single_attack_logits[:5].detach().cpu().numpy()}")
                print(f"    Batch sample: {batch_attack_logits[:5].detach().cpu().numpy()}")
        
        if army_diff > self.tolerance and not torch.isnan(torch.tensor(army_diff)):
            print(f"  ERROR: Army logits differ by {army_diff:.6f} > {self.tolerance}")
            if army_logits.dim() == 2 and batch_army_logits.dim() == 2 and self.config.detailed_logging:
                print(f"    Single sample: {single_army_logits[:3, :3].detach().cpu().numpy()}")
                print(f"    Batch sample: {batch_army_logits[:3, :3].detach().cpu().numpy()}")
        
        all_differences_ok = (
            (placement_diff <= self.tolerance or torch.isnan(torch.tensor(placement_diff))) and
            (attack_diff <= self.tolerance or torch.isnan(torch.tensor(attack_diff))) and  
            (army_diff <= self.tolerance or torch.isnan(torch.tensor(army_diff)))
        )
        
        if all_differences_ok and not (single_has_nan or batch_has_nan):
            if self.config.detailed_logging:
                print(f"  ✓ Single episode inference matches batch inference perfectly!")
        else:
            if self.config.detailed_logging:
                print(f"  ✗ Differences detected between single and batch inference")
                print(f"    This indicates the inputs or model state differ between inference and PPO update")
        with torch.no_grad():
            _, single_attack_logits, single_army_logits = agent.run_model(
                node_features=single_post_features.unsqueeze(0),   # Add batch dimension
                action_edges=single_action_edges.unsqueeze(0),     # Add batch dimension
                action=Phase.ATTACK_TRANSFER
            )
            single_attack_logits = single_attack_logits.squeeze(0)  # Remove batch dimension
            single_army_logits = single_army_logits.squeeze(0)     # Remove batch dimension
        
        # Compare with batched outputs for the same episode
        batch_placement_logits = placement_logits[test_episode_idx]
        batch_attack_logits = attack_logits[test_episode_idx]
        batch_army_logits = army_logits[test_episode_idx]
        
        print(f"  Single inference shapes - placement: {single_placement_logits_raw.shape}, attack: {single_attack_logits.shape}, army: {single_army_logits.shape}")
        print(f"  Batch inference shapes - placement: {batch_placement_logits.shape}, attack: {batch_attack_logits.shape}, army: {batch_army_logits.shape}")
        
        # Verify identical results with NaN handling
        placement_diff = self._safe_max_diff(single_placement_logits_raw, batch_placement_logits)
        attack_diff = self._safe_max_diff(single_attack_logits, batch_attack_logits)
        army_diff = self._safe_max_diff(single_army_logits, batch_army_logits)
        
        print(f"  Max differences - placement: {placement_diff:.6f}, attack: {attack_diff:.6f}, army: {army_diff:.6f}")
        
        # Check for NaN/Inf values in placement logits
        single_has_nan = torch.isnan(single_placement_logits_raw).any() or torch.isinf(single_placement_logits_raw).any()
        batch_has_nan = torch.isnan(batch_placement_logits).any() or torch.isinf(batch_placement_logits).any()
        
        if single_has_nan or batch_has_nan:
            print(f"  🚨 NaN/Inf detected in placement logits - Single: {single_has_nan}, Batch: {batch_has_nan}")
        
        # These should be identical (or very close due to floating point precision)
        if placement_diff > self.tolerance and not torch.isnan(torch.tensor(placement_diff)):
            print(f"  ERROR: Placement logits differ by {placement_diff:.6f} > {self.tolerance}")
            print(f"    Single sample: {single_placement_logits_raw[:5].detach().cpu().numpy()}")
            print(f"    Batch sample: {batch_placement_logits[:5].detach().cpu().numpy()}")
        
        if attack_diff > self.tolerance and not torch.isnan(torch.tensor(attack_diff)):
            print(f"  ERROR: Attack logits differ by {attack_diff:.6f} > {self.tolerance}")
            print(f"    Single sample: {single_attack_logits[:5].detach().cpu().numpy()}")
            print(f"    Batch sample: {batch_attack_logits[:5].detach().cpu().numpy()}")
        
        if army_diff > self.tolerance and not torch.isnan(torch.tensor(army_diff)):
            print(f"  ERROR: Army logits differ by {army_diff:.6f} > {self.tolerance}")
            if single_army_logits.dim() == 2 and batch_army_logits.dim() == 2:
                print(f"    Single sample: {single_army_logits[:3, :3].detach().cpu().numpy()}")
                print(f"    Batch sample: {batch_army_logits[:3, :3].detach().cpu().numpy()}")
        
        all_differences_ok = (
            (placement_diff <= self.tolerance or torch.isnan(torch.tensor(placement_diff))) and
            (attack_diff <= self.tolerance or torch.isnan(torch.tensor(attack_diff))) and  
            (army_diff <= self.tolerance or torch.isnan(torch.tensor(army_diff)))
        )
        
        if all_differences_ok and not (single_has_nan or batch_has_nan):
            print(f"  ✓ Single episode inference matches batch inference perfectly!")
        else:
            print(f"  ✗ Differences detected between single and batch inference")
            print(f"    This indicates the inputs or model state differ between inference and PPO update")
    
    def verify_buffer_data_integrity(self, starting_features_batched, post_features_batched, action_edges_batched, epoch):
        """
        Critical verification: Check if batched inputs match the original single-sample inputs.
        This verifies that the data stored in buffer during action selection is identical
        to what we're using now during PPO update.
        
        Only runs in the first epoch since model parameters change after training.
        """
        if not self._should_run('verify_buffer_integrity'):
            return
        
        # Only verify in the first epoch since model parameters change after training
        if epoch > 0:
            return
            
        if self.config.detailed_logging:
            print(f"\n=== INFERENCE vs PPO UPDATE INPUT VERIFICATION ===")
        
        # Test: Compare each episode's batched data with what would have been fed during inference
        for episode_idx in range(min(3, starting_features_batched.size(0))):
            if self.config.detailed_logging:
                print(f"\nEpisode {episode_idx} - Comparing inference vs PPO inputs:")
            
            # Extract single episode data (what was stored from inference)
            single_start_features = starting_features_batched[episode_idx]  # [num_nodes, features]
            single_post_features = post_features_batched[episode_idx]      # [num_nodes, features]
            single_action_edges = action_edges_batched[episode_idx]        # [42, 2]
            
            if self.config.detailed_logging:
                print(f"  Single episode starting features shape: {single_start_features.shape}")
                print(f"  Single episode post features shape: {single_post_features.shape}")
                print(f"  Single episode action edges shape: {single_action_edges.shape}")
                
                # Verify data integrity - these should match exactly what was fed during inference
                print(f"  Starting features sample: {single_start_features[0, :3].cpu().numpy()}")
                print(f"  Post features sample: {single_post_features[0, :3].cpu().numpy()}")
                print(f"  Action edges sample: {single_action_edges[:3].cpu().numpy()}")
            
            # Check for data corruption
            if torch.isnan(single_start_features).any():
                print(f"  ERROR: NaN found in episode {episode_idx} starting features!")
            if torch.isinf(single_start_features).any():
                print(f"  ERROR: Inf found in episode {episode_idx} starting features!")
            if torch.isnan(single_post_features).any():
                print(f"  ERROR: NaN found in episode {episode_idx} post features!")
            if torch.isinf(single_post_features).any():
                print(f"  ERROR: Inf found in episode {episode_idx} post features!")
            
            # Verify action edges are valid indices
            if (single_action_edges < 0).any() or (single_action_edges >= single_start_features.size(0)).any():
                print(f"  ERROR: Invalid action edge indices in episode {episode_idx}!")
                if self.config.detailed_logging:
                    print(f"    Edge range: [{single_action_edges.min()}, {single_action_edges.max()}]")
                    print(f"    Valid range: [0, {single_start_features.size(0) - 1}]")
            
            if self.config.detailed_logging:
                print(f"  ✓ Episode {episode_idx} data integrity verified")
    
    def verify_action_data(self, attacks_data, placements_data, edges_data, 
                          new_placement_log_probs, new_attack_log_probs):
        """Verify actions and edges data integrity"""
        if not self._should_run('verify_action_data'):
            return
            
        if self.config.detailed_logging:
            print(f"\n=== ACTION DATA VERIFICATION ===")
            print(f"Attacks shape: {attacks_data.shape}")
            print(f"Placements shape: {placements_data.shape}")
            print(f"Edges shape: {edges_data.shape}")
        
        # Verify that attacks and edges are properly aligned
        assert attacks_data.dtype == torch.long, f"Expected attacks dtype long, got {attacks_data.dtype}"
        assert placements_data.dtype == torch.long, f"Expected placements dtype long, got {placements_data.dtype}"
        assert edges_data.dtype == torch.long, f"Expected edges dtype long, got {edges_data.dtype}"
        
        if self.config.detailed_logging:
            print(f"New placement log probs shape: {new_placement_log_probs.shape}")
            print(f"New attack log probs shape: {new_attack_log_probs.shape}")
            print("✓ Action data verification passed")
    
    def verify_old_log_probs(self, old_placement_log_probs, old_attack_log_probs):
        """Verify old log probabilities from buffer"""
        if not self._should_run('verify_old_log_probs'):
            return
            
        if self.config.detailed_logging:
            print(f"\n=== OLD LOG PROBS VERIFICATION ===")
            print(f"Old placement log probs shape: {old_placement_log_probs.shape}")
            print(f"Old attack log probs shape: {old_attack_log_probs.shape}")
            
            # Check samples of old log probs
            if old_placement_log_probs.numel() > 0:
                print(f"Old placement log probs sample: {old_placement_log_probs[0, :3].cpu().numpy()}")
            if old_attack_log_probs.numel() > 0:
                print(f"Old attack log probs sample: {old_attack_log_probs[0, :3].cpu().numpy()}")
            print("✓ Old log probs verification passed")
    
    def check_extreme_log_prob_differences(self, total_new_log_probs, total_old_log_probs):
        """Check for extreme log probability differences that might indicate issues"""
        if not self._should_run('check_extreme_log_probs'):
            return
            
        max_diff = (total_new_log_probs - total_old_log_probs).abs().max()
        if max_diff > 50:
            print(f"WARNING: Large log prob difference detected: {max_diff:.2f}")
    
    def check_extreme_attack_differences(self, attack_diff):
        """Debug: Check for extreme attack differences"""
        if not self._should_run('check_extreme_attack_diffs') or attack_diff.numel() == 0:
            return
            
        max_attack_diff = attack_diff.abs().max()
        if max_attack_diff > 5:
            print(f"WARNING: Extreme attack log prob difference detected: {max_attack_diff:.2f}")
            print(f"This indicates a mismatch in how attack log probabilities are indexed/computed")
            print(f"Between action selection (stored in buffer) and PPO update (recomputed)")
            print(f"The issue is that old and new attack log probs have different indexing schemes")
            # Note: Removed exit() call as this should be handled gracefully
    
    def analyze_gradients(self, model, agent):
        """
        Comprehensive gradient analysis to monitor training health
        """
        if not self._should_run('analyze_gradients'):
            return {}
            
        grad_stats = {}
        layer_grads = {}
        
        total_norm = 0
        total_params = 0
        nan_params = 0
        zero_grad_params = 0
        
        # Analyze gradients by layer type
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                param_norm = grad.norm().item()
                total_norm += param_norm ** 2
                total_params += param.numel()
                
                # Check for problematic gradients
                if torch.isnan(grad).any():
                    nan_params += 1
                if grad.abs().max().item() < 1e-8:
                    zero_grad_params += 1
                
                # Store per-layer statistics
                layer_type = name.split('.')[0]  # e.g., 'gnn1', 'attack_head', etc.
                if layer_type not in layer_grads:
                    layer_grads[layer_type] = []
                layer_grads[layer_type].append(param_norm)
        
        total_norm = total_norm ** 0.5
        
        # Compute layer-wise gradient norms
        layer_stats = {}
        for layer_type, norms in layer_grads.items():
            layer_stats[layer_type] = {
                'mean_norm': sum(norms) / len(norms),
                'max_norm': max(norms),
                'min_norm': min(norms)
            }
        
        grad_stats = {
            'total_norm': total_norm,
            'nan_params': nan_params,
            'zero_grad_params': zero_grad_params,
            'total_params': total_params,
            'layer_stats': layer_stats
        }
        
        # Print analysis if enabled
        if self.config.detailed_logging:
            print(f"\n=== GRADIENT ANALYSIS ===")
            print(f"Total gradient norm: {total_norm:.6f}")
            print(f"Parameters with NaN gradients: {nan_params}")
            print(f"Parameters with zero gradients: {zero_grad_params}")
            print(f"Layer-wise gradient norms:")
            for layer, stats in layer_stats.items():
                print(f"  {layer}: mean={stats['mean_norm']:.6f}, max={stats['max_norm']:.6f}, min={stats['min_norm']:.6f}")
        
        # Detect potential issues
        if total_norm > 100:
            print(f"⚠️  WARNING: Very large gradient norm ({total_norm:.2f}) - potential exploding gradients")
        elif total_norm < 1e-6:
            print(f"⚠️  WARNING: Very small gradient norm ({total_norm:.2e}) - potential vanishing gradients")
        elif 1 <= total_norm <= 10:
            if self.config.detailed_logging:
                print(f"✅ Healthy gradient norm range")
        
        if nan_params > 0:
            print(f"🚨 ERROR: {nan_params} parameters have NaN gradients!")
        
        # Store in agent rewards for tensorboard logging
        agent.total_rewards['gradient_norm'] = total_norm
        agent.total_rewards['nan_gradient_params'] = nan_params
        agent.total_rewards['zero_gradient_params'] = zero_grad_params
        
        return grad_stats
        
        grad_stats = {
            'total_norm': total_norm,
            'nan_params': nan_params,
            'zero_grad_params': zero_grad_params,
            'total_params': total_params,
            'layer_stats': layer_stats
        }
        
        # Print analysis if enabled
        print(f"\n=== GRADIENT ANALYSIS ===")
        print(f"Total gradient norm: {total_norm:.6f}")
        print(f"Parameters with NaN gradients: {nan_params}")
        print(f"Parameters with zero gradients: {zero_grad_params}")
        print(f"Layer-wise gradient norms:")
        for layer, stats in layer_stats.items():
            print(f"  {layer}: mean={stats['mean_norm']:.6f}, max={stats['max_norm']:.6f}, min={stats['min_norm']:.6f}")
        
        # Detect potential issues
        if total_norm > 100:
            print(f"⚠️  WARNING: Very large gradient norm ({total_norm:.2f}) - potential exploding gradients")
        elif total_norm < 1e-6:
            print(f"⚠️  WARNING: Very small gradient norm ({total_norm:.2e}) - potential vanishing gradients")
        elif 1 <= total_norm <= 10:
            print(f"✅ Healthy gradient norm range")
        
        if nan_params > 0:
            print(f"🚨 ERROR: {nan_params} parameters have NaN gradients!")
        
        # Store in agent rewards for tensorboard logging
        agent.total_rewards['gradient_norm'] = total_norm
        agent.total_rewards['nan_gradient_params'] = nan_params
        agent.total_rewards['zero_gradient_params'] = zero_grad_params
        
        return grad_stats
    
    def analyze_weight_changes(self, model, prev_weights, agent):
        """
        Track how much model weights are changing between updates
        """
        if not self._should_run('analyze_weight_changes') or prev_weights is None:
            return {}
        
        weight_changes = {}
        total_change = 0
        total_weight_norm = 0
        
        if self.config.detailed_logging:
            print(f"\n=== WEIGHT CHANGE ANALYSIS ===")
        
        for name, param in model.named_parameters():
            if name in prev_weights:
                current_weight = param.data
                prev_weight = prev_weights[name]
                
                change = (current_weight - prev_weight).norm().item()
                weight_norm = current_weight.norm().item()
                
                relative_change = change / (weight_norm + 1e-8)
                weight_changes[name] = {
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'weight_norm': weight_norm
                }
                
                total_change += change ** 2
                total_weight_norm += weight_norm ** 2
                
                # Print significant changes
                if self.config.detailed_logging:
                    if relative_change > 0.1:  # More than 10% change
                        print(f"  {name}: {relative_change:.4f} relative change (large)")
                    elif relative_change < 1e-6:  # Very small change
                        print(f"  {name}: {relative_change:.2e} relative change (minimal)")
        
        total_change = total_change ** 0.5
        total_weight_norm = total_weight_norm ** 0.5
        overall_relative_change = total_change / (total_weight_norm + 1e-8)
        
        if self.config.detailed_logging:
            print(f"Overall relative weight change: {overall_relative_change:.6f}")
        
        if overall_relative_change > 0.05:
            print(f"⚠️  Large weight changes - model learning rapidly")
        elif overall_relative_change < 1e-5:
            print(f"⚠️  Very small weight changes - learning may have stagnated")
        else:
            if self.config.detailed_logging:
                print(f"✅ Moderate weight changes - healthy learning")
        
        # Store for logging
        agent.total_rewards['weight_change_norm'] = total_change
        agent.total_rewards['relative_weight_change'] = overall_relative_change
        
        return weight_changes
    
    def analyze_action_distribution(self, placement_logits, attack_logits, army_logits, agent):
        """
        Analyze the distribution of actions to detect policy collapse or over-concentration
        """
        if not self._should_run('analyze_action_distributions'):  # Fixed: use plural to match config
            return
            
        if self.config.detailed_logging:
            print(f"\n=== ACTION DISTRIBUTION ANALYSIS ===")
        
        # Placement entropy
        if placement_logits.numel() > 0:
            placement_probs = torch.softmax(placement_logits, dim=-1)
            placement_entropy = -(placement_probs * torch.log(placement_probs + 1e-8)).sum(dim=-1).mean()
            max_placement_entropy = torch.log(torch.tensor(float(placement_logits.size(-1))))
            normalized_placement_entropy = placement_entropy / max_placement_entropy
            
            if self.config.detailed_logging:
                print(f"Placement entropy: {placement_entropy:.4f} (normalized: {normalized_placement_entropy:.4f})")
            
            if normalized_placement_entropy < 0.1:
                print(f"⚠️  Very low placement entropy - policy may be over-concentrated")
            elif normalized_placement_entropy > 0.9:
                print(f"⚠️  Very high placement entropy - policy may be too random")
            else:
                if self.config.detailed_logging:
                    print(f"✅ Healthy placement entropy")
        
        # Attack entropy  
        if attack_logits.numel() > 0:
            attack_probs = torch.softmax(attack_logits, dim=-1)
            attack_entropy = -(attack_probs * torch.log(attack_probs + 1e-8)).sum(dim=-1).mean()
            max_attack_entropy = torch.log(torch.tensor(float(attack_logits.size(-1))))
            normalized_attack_entropy = attack_entropy / max_attack_entropy
            
            if self.config.detailed_logging:
                print(f"Attack entropy: {attack_entropy:.4f} (normalized: {normalized_attack_entropy:.4f})")
            
            if normalized_attack_entropy < 0.1:
                print(f"⚠️  Very low attack entropy - policy may be over-concentrated") 
            elif normalized_attack_entropy > 0.9:
                print(f"⚠️  Very high attack entropy - policy may be too random")
            else:
                if self.config.detailed_logging:
                    print(f"✅ Healthy attack entropy")
        
        # Army selection entropy
        if army_logits.numel() > 0:
            army_probs = torch.softmax(army_logits, dim=-1) 
            army_entropy = -(army_probs * torch.log(army_probs + 1e-8)).sum(dim=-1).mean()
            max_army_entropy = torch.log(torch.tensor(float(army_logits.size(-1))))
            normalized_army_entropy = army_entropy / max_army_entropy
            
            if self.config.detailed_logging:
                print(f"Army entropy: {army_entropy:.4f} (normalized: {normalized_army_entropy:.4f})")
            
            if normalized_army_entropy < 0.1:
                print(f"⚠️  Very low army entropy - policy may be over-concentrated")
            elif normalized_army_entropy > 0.9: 
                print(f"⚠️  Very high army entropy - policy may be too random")
            else:
                if self.config.detailed_logging:
                    print(f"✅ Healthy army entropy")
    
    def verify_value_computation(self, agent, features_batched, buffer, values_pred, returns, advantages):
        """
        Verify value computation consistency and identify potential issues
        """
        if not self._should_run('verify_value_computation'):
            return
        
        if self.config.detailed_logging:
            print(f"\n=== VALUE COMPUTATION VERIFICATION ===")
        
        # 1. Check value prediction shape consistency
        old_values = buffer.get_values()
        assert values_pred.shape == old_values.shape, f"Value shape mismatch: pred={values_pred.shape}, old={old_values.shape}"
        
        # 2. Check returns computation (returns should equal advantages + old_values)
        expected_returns = advantages + old_values
        returns_diff = self._safe_max_diff(returns, expected_returns)
        if torch.isnan(returns_diff):
            print(f"🚨 ERROR: NaN in returns computation (returns: {torch.isnan(returns).sum()}, expected: {torch.isnan(expected_returns).sum()})")
        elif returns_diff > 0.1:  # Adjusted threshold - large differences indicate real problems
            print(f"🚨 ERROR: Large returns computation mismatch. Max diff: {returns_diff:.6f}")
            if self.config.detailed_logging:
                print(f"    Returns range: [{returns.min():.4f}, {returns.max():.4f}]")
                print(f"    Expected range: [{expected_returns.min():.4f}, {expected_returns.max():.4f}]")
                print(f"    This suggests GAE computation or buffer inconsistency issues")
        elif returns_diff > 1e-5:
            print(f"⚠️  WARNING: Moderate returns computation mismatch. Max diff: {returns_diff:.6f}")
        elif self.config.detailed_logging:
            print(f"✅ Returns computation consistent (diff: {returns_diff:.8f})")
        
        # 3. Verify value prediction ranges and statistics
        if self.config.detailed_logging:
            print(f"Value prediction range: [{values_pred.min():.4f}, {values_pred.max():.4f}]")
            print(f"Old values range: [{old_values.min():.4f}, {old_values.max():.4f}]")
            print(f"Returns range: [{returns.min():.4f}, {returns.max():.4f}]")
            print(f"Advantages range: [{advantages.min():.4f}, {advantages.max():.4f}]")
        
        # 4. Check for extreme value differences (potential training instability)
        value_diff = (values_pred - old_values).abs()
        
        # Handle NaN values in differences
        if torch.isnan(value_diff).any():
            print(f"🚨 ERROR: NaN values in value differences!")
            max_value_diff = float('nan')
            mean_value_diff = float('nan')
        else:
            max_value_diff = value_diff.max()
            mean_value_diff = value_diff.mean()
        
        if torch.isnan(max_value_diff):
            print(f"🚨 ERROR: NaN max value difference!")
        elif max_value_diff > 10.0:
            print(f"⚠️  WARNING: Large max value prediction difference: {max_value_diff:.4f}")
        
        if torch.isnan(mean_value_diff):
            print(f"🚨 ERROR: NaN mean value difference!")
        elif mean_value_diff > 2.0:
            print(f"⚠️  WARNING: Large mean value prediction difference: {mean_value_diff:.4f}")
        elif self.config.detailed_logging:
            print(f"✅ Value differences within reasonable range (max: {max_value_diff:.4f}, mean: {mean_value_diff:.4f})")
        
        # 5. Verify value head consistency by recomputing
        model_training_mode = agent.model.training
        if model_training_mode:
            print(f"⚠️  WARNING: Model is in training mode during verification - this can cause recomputation differences!")
            agent.model.eval()  # Temporarily set to eval mode for consistent results
        
        with torch.no_grad():
            values_recomputed = agent.model.get_value(features_batched)
            recompute_diff = self._safe_max_diff(values_pred, values_recomputed)
            if torch.isnan(recompute_diff):
                print(f"🚨 ERROR: NaN in value recomputation (pred: {torch.isnan(values_pred).sum()}, recomputed: {torch.isnan(values_recomputed).sum()})")
            elif recompute_diff > 0.1:  # Adjusted threshold - large differences indicate model state issues
                print(f"🚨 ERROR: Large value recomputation mismatch: {recompute_diff:.6f}")
                print(f"    This indicates serious model state inconsistency or feature processing issues!")
                print(f"    Model was in {'training' if model_training_mode else 'eval'} mode")
                if self.config.detailed_logging:
                    print(f"    Original values range: [{values_pred.min():.4f}, {values_pred.max():.4f}]")
                    print(f"    Recomputed range: [{values_recomputed.min():.4f}, {values_recomputed.max():.4f}]")
            elif recompute_diff > 1e-6:
                print(f"⚠️  WARNING: Moderate value recomputation mismatch: {recompute_diff:.6f}")
                print(f"    This may indicate model state inconsistency")
            elif self.config.detailed_logging:
                print(f"✅ Value recomputation consistent (diff: {recompute_diff:.8f})")
        
        # Restore original training mode
        if model_training_mode:
            agent.model.train()
        
        # 6. Check for NaN or Inf values
        if torch.isnan(values_pred).any():
            print(f"🚨 ERROR: NaN values in predicted values!")
        if torch.isinf(values_pred).any():
            print(f"🚨 ERROR: Inf values in predicted values!")
        if torch.isnan(returns).any():
            print(f"🚨 ERROR: NaN values in returns!")
        if torch.isinf(returns).any():
            print(f"🚨 ERROR: Inf values in returns!")
        
        # 7. Check value loss implications
        if hasattr(agent, 'total_rewards'):
            agent.total_rewards['max_value_change'] = max_value_diff.item()
            agent.total_rewards['mean_value_change'] = mean_value_diff.item()
            agent.total_rewards['value_pred_mean'] = values_pred.mean().item()
            agent.total_rewards['value_pred_std'] = values_pred.std().item()
            agent.total_rewards['returns_mean'] = returns.mean().item()
            agent.total_rewards['returns_std'] = returns.std().item()
    
    def verify_gae_computation(self, rewards, old_values, last_value, dones, advantages, returns, gamma, lam):
        """
        Verify GAE (Generalized Advantage Estimation) computation step by step
        """
        if not self._should_run('verify_gae_computation'):
            return
            
        if self.config.detailed_logging:
            print(f"\n=== GAE COMPUTATION VERIFICATION ===")
            print(f"Rewards shape: {rewards.shape}, range: [{rewards.min():.4f}, {rewards.max():.4f}]")
            print(f"Values shape: {old_values.shape}, range: [{old_values.min():.4f}, {old_values.max():.4f}]")
            print(f"Last value: {last_value:.4f}")
            print(f"Gamma: {gamma}, Lambda: {lam}")
        
        # Manual GAE computation for verification
        device = old_values.device
        T = len(rewards)
        
        # Extend values with last_value
        extended_values = torch.cat([old_values, torch.tensor([last_value], device=device)])
        
        # Compute deltas
        deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]
        
        # Compute advantages manually
        advantages_manual = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
            advantages_manual[t] = gae
        
        # Compute returns manually
        returns_manual = advantages_manual + old_values
        
        # Compare with provided values
        adv_diff = (advantages - advantages_manual).abs().max()
        ret_diff = (returns - returns_manual).abs().max()
        
        if adv_diff > 0.1:
            print(f"🚨 ERROR: Large GAE advantages mismatch. Max diff: {adv_diff:.6f}")
            print(f"    This indicates serious issues in GAE computation or reward scaling")
        elif adv_diff > 1e-5:
            print(f"⚠️  WARNING: Moderate GAE advantages mismatch. Max diff: {adv_diff:.6f}")
        elif self.config.detailed_logging:
            print(f"✅ GAE advantages computation correct (diff: {adv_diff:.8f})")
        
        if ret_diff > 0.1:
            print(f"🚨 ERROR: Large GAE returns mismatch. Max diff: {ret_diff:.6f}")
            print(f"    This indicates serious issues in GAE computation or value estimation")
        elif ret_diff > 1e-5:
            print(f"⚠️  WARNING: Moderate GAE returns mismatch. Max diff: {ret_diff:.6f}")
        elif self.config.detailed_logging:
            print(f"✅ GAE returns computation correct (diff: {ret_diff:.8f})")
        
        # Check for numerical issues
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print(f"🚨 ERROR: NaN/Inf in advantages!")
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            print(f"🚨 ERROR: NaN/Inf in returns!")
            
        return advantages_manual, returns_manual
