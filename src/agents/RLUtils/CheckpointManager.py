"""
Enhanced checkpoint management system for RL training
Saves and loads: model state, optimizer state, reward normalizer, game number, and other training state
"""

import os
import torch
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import asdict
import logging

class CheckpointManager:
    """Manages comprehensive checkpointing for RL training"""
    
    def __init__(self, config, experiment_name: str):
        """Initialize checkpoint manager"""
        self.config = config
        self.experiment_name = experiment_name
        
        # Create checkpoint directory structure
        self.base_checkpoint_dir = os.path.join(config.logging.log_dir, "checkpoints")
        self.checkpoint_dir = os.path.join(self.base_checkpoint_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.is_saved = False
        # Configuration
        self.save_frequency = config.logging.checkpoint_every_n_episodes
        self.keep_last_n = getattr(config.logging, 'keep_last_n_checkpoints', 5)
        
        print(f"📁 Checkpoint manager initialized for experiment: {experiment_name}")
        print(f"   Directory: {self.checkpoint_dir}")
        print(f"   Save frequency: every {self.save_frequency} games")
    
    def should_save_checkpoint(self, game_number: int) -> bool:
        """Check if we should save a checkpoint at this game number"""
        if game_number % self.save_frequency == 0:
            return not self.is_saved
        else:
            self.is_saved = False
            return False
    
    def find_latest_checkpoint(self, experiment_name: Optional[str] = None) -> Optional[str]:
        """Find the latest checkpoint for an experiment"""
        if experiment_name is None:
            experiment_name = self.experiment_name
            
        exp_checkpoint_dir = os.path.join(self.base_checkpoint_dir, experiment_name)
        
        if not os.path.exists(exp_checkpoint_dir):
            return None
            
        # Find all checkpoint files
        checkpoint_files = [f for f in os.listdir(exp_checkpoint_dir) if f.endswith('.pth')]
        
        if not checkpoint_files:
            return None
            
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(exp_checkpoint_dir, x)), reverse=True)
        
        latest_checkpoint = os.path.join(exp_checkpoint_dir, checkpoint_files[0])
        return latest_checkpoint
    
    def save_checkpoint(self, agent, game_number: int, force: bool = False) -> Optional[str]:
        """Save comprehensive checkpoint"""
        if not force and not self.should_save_checkpoint(game_number):
            return None
        self.is_saved = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"checkpoint_game_{game_number}_{timestamp}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
        
        # Collect all state information
        checkpoint_data = {
            "game_number": game_number,
            "timestamp": timestamp,
            "experiment_name": self.experiment_name,
            "config": self._serialize_config(self.config),
            
            # Model and optimizer
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            
            # PPO agent state
            "ppo_agent_state": self._get_ppo_agent_state(agent.ppo_agent),
            

            # Agent-specific state
            "agent_state": {
                "total_rewards": getattr(agent, 'total_rewards', {}),
                "game_number": getattr(agent, 'game_number', game_number),
                "turns_count": getattr(agent, 'turns_count', 0),
                "placement_count": getattr(agent, 'placement_count', 0),
                "attack_count": getattr(agent, 'attack_count', 0)
            },
            
            # Statistics and trackers
            "stat_trackers": self._get_stat_tracker_states(agent),
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"💾 Saved checkpoint: {os.path.basename(checkpoint_path)}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, agent, checkpoint_path: str, load_config: Dict[str, bool]) -> bool:
        """Load checkpoint with selective loading based on config"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            # Get device from the agent (which properly handles model device detection)
            device = getattr(agent, 'device', 'cpu')
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            print(f"📂 Loading checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"   Saved at game: {checkpoint_data.get('game_number', 'unknown')}")
            print(f"   Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
            
            # Load model state
            if load_config.get("model", True):
                agent.model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
                print("   ✅ Model state loaded")
            
            # Load optimizer state
            if load_config.get("optimizer", True):
                agent.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                print("   ✅ Optimizer state loaded")

            # Load game number
            if load_config.get("game_number", True):
                agent.game_number = checkpoint_data.get("game_number", 0)
                print(f"   ✅ Game number loaded: {agent.game_number}")
            
            # Load PPO agent state
            if load_config.get("ppo_state", True):
                self._restore_ppo_agent_state(
                    agent.ppo_agent, 
                    checkpoint_data.get("ppo_agent_state", {})
                )
                print("   ✅ PPO agent state loaded")
            
            # Load stat trackers
            if load_config.get("stat_trackers", True):
                self._restore_stat_tracker_states(agent, checkpoint_data.get("stat_trackers", {}))
                print("   ✅ Stat trackers loaded")
            
            # Load agent-specific state
            if load_config.get("training_state", True):
                agent_state = checkpoint_data.get("agent_state", {})
                if hasattr(agent, 'total_rewards'):
                    agent.total_rewards.update(agent_state.get("total_rewards", {}))
                agent.turns_count = agent_state.get("turns_count", 0)
                agent.placement_count = agent_state.get("placement_count", 0)
                agent.attack_count = agent_state.get("attack_count", 0)
                print("   ✅ Training state loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_reward_normalizer_state(self, normalizer) -> Dict[str, Any]:
        """Extract reward normalizer state"""
        if normalizer is None:
            return {}
            
        return {
            "mean": float(normalizer.mean) if hasattr(normalizer, 'mean') else 0.0,
            "var": float(normalizer.var) if hasattr(normalizer, 'var') else 1.0,
            "count": float(normalizer.count) if hasattr(normalizer, 'count') else 1e-8
        }
    
    def _restore_reward_normalizer_state(self, normalizer, state: Dict[str, Any]):
        """Restore reward normalizer state"""
        if not state or normalizer is None:
            return
            
        if hasattr(normalizer, 'mean'):
            normalizer.mean = state.get("mean", 0.0)
        if hasattr(normalizer, 'var'):
            normalizer.var = state.get("var", 1.0)
        if hasattr(normalizer, 'count'):
            normalizer.count = state.get("count", 1e-8)
    
    def _get_ppo_agent_state(self, ppo_agent) -> Dict[str, Any]:
        """Extract PPO agent state (trackers, etc.)"""
        state = {}
        
        # Save advantage tracker state
        if hasattr(ppo_agent, 'advantage_tracker') and ppo_agent.advantage_tracker:
            tracker = ppo_agent.advantage_tracker
            state["advantage_tracker"] = {
                "num_samples": getattr(tracker, 'num_samples', 0),
                "sum_advantages": getattr(tracker, 'sum_advantages', 0.0),
                "sum_squared_advantages": getattr(tracker, 'sum_squared_advantages', 0.0),
                "baseline_mean": getattr(tracker, 'baseline_mean', 0.0),
                "baseline_std": getattr(tracker, 'baseline_std', 1.0)
            }
        
        # Save other PPO-specific state
        state["entropy_coeff"] = getattr(ppo_agent, 'entropy_coeff', 0.01)
        state["clipfrac_tracker"] = getattr(ppo_agent, 'clipfrac_tracker', [])
        
        return state
    
    def _restore_ppo_agent_state(self, ppo_agent, state: Dict[str, Any]):
        """Restore PPO agent state"""
        if not state:
            return
            
        # Restore advantage tracker
        if "advantage_tracker" in state and hasattr(ppo_agent, 'advantage_tracker'):
            tracker_state = state["advantage_tracker"]
            tracker = ppo_agent.advantage_tracker
            if tracker:
                tracker.num_samples = tracker_state.get("num_samples", 0)
                tracker.sum_advantages = tracker_state.get("sum_advantages", 0.0)
                tracker.sum_squared_advantages = tracker_state.get("sum_squared_advantages", 0.0)
                tracker.baseline_mean = tracker_state.get("baseline_mean", 0.0)
                tracker.baseline_std = tracker_state.get("baseline_std", 1.0)
        
        # Restore other PPO state
        if hasattr(ppo_agent, 'entropy_coeff'):
            ppo_agent.entropy_coeff = state.get("entropy_coeff", 0.01)
        if hasattr(ppo_agent, 'clipfrac_tracker'):
            ppo_agent.clipfrac_tracker = state.get("clipfrac_tracker", [])
    
    def _get_stat_tracker_states(self, agent) -> Dict[str, Any]:
        """Extract stat tracker states"""
        trackers = {}
        
        # Get common tracking attributes
        for attr in ['total_rewards', 'game_number', 'turns_count', 'placement_count', 'attack_count']:
            if hasattr(agent, attr):
                trackers[attr] = getattr(agent, attr)
        
        return trackers
    
    def _restore_stat_tracker_states(self, agent, trackers: Dict[str, Any]):
        """Restore stat tracker states"""
        for attr, value in trackers.items():
            if hasattr(agent, attr):
                setattr(agent, attr, value)
    
    def _serialize_config(self, config) -> Dict[str, Any]:
        """Serialize config for saving (convert to dict)"""
        try:
            # Try to convert dataclass to dict
            return asdict(config)
        except:
            # Fallback: just save the string representation
            return {"config_str": str(config)}
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the keep limit"""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        
        if len(checkpoint_files) <= self.keep_last_n:
            return
            
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
        
        # Remove oldest files
        files_to_remove = checkpoint_files[:-self.keep_last_n]
        for filename in files_to_remove:
            filepath = os.path.join(self.checkpoint_dir, filename)
            try:
                os.remove(filepath)
                print(f"🗑️  Removed old checkpoint: {filename}")
            except Exception as e:
                print(f"⚠️  Failed to remove old checkpoint {filename}: {e}")
