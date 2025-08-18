"""
Training Configuration System

This module provides a comprehensive configuration system for PPO training,
including all hyperparameters, verification settings, and logging options.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import os


@dataclass 
class ModelConfig:
    """Configuration for the neural network model"""
    in_channels: int = 17
    hidden_channels: int = 64
    embed_dim: int = 64
    max_army_send: int = 50  # Deprecated: kept for backward compatibility
    n_army_options: int = 4  # Number of army percentage options (25%, 50%, 75%, 100%)
    device: str = 'cpu'  # 'cpu', 'cuda', or 'auto'
    
    # Model architecture selection
    model_type: str = 'standard'  # 'standard', 'residual', 'sage', 'transformer'
    edge_feat_dim: int = 5  # Number of edge features (default 0 for backward compatibility)
    
    def get_device(self) -> torch.device:
        """Get the actual torch device"""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    # Core PPO parameters
    gamma: float = 0.99                    # Discount factor
    lam: float = 0.95                      # GAE lambda
    clip_eps: float = 0.2                  # PPO clipping epsilon
    ppo_epochs: int = 1                    # Number of PPO epochs per update
    learning_rate: float = 5e-6            # Adam optimizer learning rate
    batch_size: int = 24                   # Episodes per batch
    
    # Advanced PPO settings
    adaptive_epochs: bool = True           # Enable adaptive epoch adjustment
    max_adaptive_epochs: int = 4           # Maximum epochs when using adaptive
    min_adaptive_epochs: int = 1           # Minimum epochs when using adaptive
    gradient_clip_norm: float = 0.1        # Gradient clipping norm
    
    # Value function and entropy
    value_loss_coeff: float = 0.5          # Value loss coefficient
    value_clip_range: float = None         # Value clipping range (None = no clipping)
    value_clip_eps: float = 0.2            # Value clipping epsilon (if value_clip_range enabled)
    entropy_coeff_start: float = 0.5       # Initial entropy coefficient
    entropy_coeff_decay: float = 0.3       # Entropy decay over training
    entropy_decay_episodes: int = 15000    # Episodes over which to decay entropy
    placement_entropy_coeff: float = 0.9   # Placement entropy coefficient
    edge_entropy_coeff: float = 0.1        # Edge entropy coefficient  
    army_entropy_coeff: float = 0.003      # Army entropy coefficient
    
    # Reward processing
    reward_clamp_min: float = -75.0        # Minimum reward value
    reward_clamp_max: float = 75.0         # Maximum reward value
    normalize_rewards: bool = True         # Enable reward normalization
    normalize_advantages: bool = True      # Enable advantage normalization


@dataclass
class VerificationConfig:
    """Configuration for verification and debugging systems"""
    # Master verification switches
    enabled: bool = False                   # Master switch for all verification
    
    # Input verification
    verify_structural_integrity: bool = True      # Check input tensor shapes and types
    verify_model_outputs: bool = True             # Check for NaN/Inf in model outputs
    verify_single_vs_batch: bool = True           # Compare single vs batch inference
    verify_buffer_integrity: bool = True          # Check buffer data consistency
    verify_action_data: bool = True               # Verify action tensors
    verify_old_log_probs: bool = True             # Check old log probabilities
    
    # Advanced verification
    check_extreme_log_prob_diffs: bool = True     # Check for extreme log prob differences
    check_extreme_attack_diffs: bool = True       # Check for extreme attack differences
    
    # Training analysis
    analyze_gradients: bool = True                 # Comprehensive gradient analysis
    analyze_weight_changes: bool = True            # Track weight changes over time
    analyze_action_distributions: bool = True      # Monitor action entropy and distributions
    verify_value_computation: bool = True          # Verify value prediction consistency
    verify_gae_computation: bool = True            # Verify GAE computation correctness
    
    # Batch inference verification (from RLGNNAgent)
    batch_verification_enabled: bool = True       # Enable input verification between phases
    
    # Verification thresholds
    tolerance: float = 1e-5                       # Numerical tolerance for comparisons
    extreme_log_prob_threshold: float = 50.0      # Threshold for extreme log prob differences
    extreme_attack_threshold: float = 5.0         # Threshold for extreme attack differences
    
    # Output control
    print_first_n_episodes: int = 3              # Print details for first N episodes
    print_verification_summary: bool = True      # Print verification summaries
    detailed_logging: bool = False                # Enable detailed verification logging


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""
    # TensorBoard logging
    log_dir: str = "analysis/logs"                # Base directory for logs
    experiment_name: str = "Attila"
    flush_every_n_episodes: int = 100              # Flush logs every N episodes
    
    # Model checkpointing
    save_checkpoints: bool = True                 # Enable model checkpointing
    checkpoint_dir: str = "res/model"             # Directory for saving checkpoints
    checkpoint_every_n_episodes: int = 500       # Save checkpoint every N episodes
    keep_last_n_checkpoints: int = 25             # Number of checkpoints to keep
    
    # Checkpoint resuming
    resume_from_checkpoint: bool = False          # Whether to resume from a checkpoint
    checkpoint_path: str = ""                    # Specific checkpoint path to resume from
    auto_resume_latest: bool = False             # Automatically resume from latest checkpoint
    resume_experiment_name: str = ""             # Resume from latest checkpoint of this experiment
    
    # What to load from checkpoint
    load_model_state: bool = True                # Load model weights
    load_optimizer_state: bool = True            # Load optimizer state
    load_reward_normalizer: bool = True          # Load reward normalizer state
    load_game_number: bool = True                # Resume game numbering
    load_stat_trackers: bool = True              # Load tracking statistics
    load_training_state: bool = True             # Load training state (entropy coeffs, etc.)
    
    # Metrics to log
    log_gradient_metrics: bool = True             # Log gradient norms and statistics
    log_weight_change_metrics: bool = True       # Log weight change statistics
    log_entropy_metrics: bool = True             # Log action entropy metrics
    log_reward_breakdown: bool = True             # Log detailed reward components
    log_game_statistics: bool = True             # Log game-specific statistics
    
    # Console output
    print_every_n_episodes: int = 50              # Print progress every N episodes
    verbose_losses: bool = False                  # Print detailed loss information
    verbose_rewards: bool = False                 # Print detailed reward information


@dataclass
class GameConfig:
    """Configuration for game-specific settings"""
    # Reward system weights (from compute_rewards)
    region_gain_reward: float = 0.5               # Reward per region gained
    region_loss_penalty: float = 0.0125           # Penalty per region lost
    continent_bonus_multiplier: float = 2.0       # Multiplier for continent bonuses
    army_efficiency_reward: float = 0.1           # Reward for efficient army usage
    action_base_reward: float = 0.005             # Base reward for taking actions
    action_efficiency_multiplier: float = 0.01    # Multiplier for action efficiency
    long_game_penalty: float = 0.02               # Per-turn penalty for long games
    win_reward: float = 75.0                      # Reward for winning
    win_speed_bonus: float = 10.0                 # Bonus for fast wins (max)
    win_speed_decay: float = 0.1                  # Decay rate for speed bonus
    loss_penalty: float = 50.0                    # Penalty for losing
    only_armies_used = False

    # Placement rewards
    placement_next_to_enemy_bonus: float = 0.1    # Bonus for placing next to enemies
    placement_safe_penalty: float = 0.05          # Penalty for placing in safe regions
    
    # Transfer and positioning rewards
    transfer_proximity_multiplier: float = 0.02   # Reward for moving closer to enemies
    transfer_proximity_decay: float = 0.3         # Exponential decay for proximity
    passivity_penalty_rate: float = 0.01          # Penalty rate for being passive
    
    # Attack rewards
    multi_side_attack_bonus: float = 0.05         # Bonus for multi-directional attacks
    overstack_penalty_rate: float = 0.000005     # Penalty rate for overstacking
    
    # Action sampling
    use_temperature_scaling: bool = False          # Enable temperature scaling
    temperature: float = 1.0                      # Temperature for action sampling

@dataclass
class TrainingConfig:
    """Master configuration containing all sub-configurations"""
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    game: GameConfig = field(default_factory=GameConfig)
    
    # Meta settings
    max_episodes: int = 50000                     # Maximum training episodes
    random_seed: Optional[int] = None             # Random seed for reproducibility
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure directories exist
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        
        # Set random seed if specified
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
    
    def get_experiment_log_dir(self) -> str:
        """Get the full path for experiment logs"""
        return os.path.join(self.logging.log_dir, f"{self.logging.experiment_name}")
    
    def summary(self) -> str:
        """Generate a summary of the configuration"""
        summary = []
        summary.append("=== TRAINING CONFIGURATION SUMMARY ===")
        summary.append(f"Model: {self.model.hidden_channels}D hidden, device={self.model.get_device()}")
        summary.append(f"PPO: lr={self.ppo.learning_rate}, batch_size={self.ppo.batch_size}, epochs={self.ppo.ppo_epochs}")
        summary.append(f"Verification: {'ENABLED' if self.verification.enabled else 'DISABLED'}")
        summary.append(f"Adaptive epochs: {'YES' if self.ppo.adaptive_epochs else 'NO'}")
        summary.append(f"Logging: {self.logging.experiment_name}")
        summary.append(f"Max episodes: {self.max_episodes}")
        return "\n".join(summary)


# Pre-defined configurations for common use cases

def get_debug_config() -> TrainingConfig:
    """Configuration with full verification enabled for debugging"""
    config = TrainingConfig()
    config.verification.enabled = True
    config.verification.batch_verification_enabled = True
    config.logging.verbose_losses = True
    config.logging.verbose_rewards = True
    config.logging.print_every_n_episodes = 1
    config.logging.experiment_name = "debug_full_verification"
    return config


def get_production_config() -> TrainingConfig:
    """Optimized configuration for production training"""
    config = TrainingConfig()
    config.verification.enabled = False
    config.verification.batch_verification_enabled = False
    config.logging.verbose_losses = False
    config.logging.verbose_rewards = False
    config.logging.print_every_n_episodes = 10
    config.logging.experiment_name = "production_training"
    config.ppo.adaptive_epochs = True
    return config


def get_analysis_config() -> TrainingConfig:
    """Configuration with gradient and weight analysis enabled"""
    config = TrainingConfig()
    config.verification.enabled = True
    config.verification.verify_structural_integrity = False  # Skip basic checks
    config.verification.verify_model_outputs = False
    config.verification.verify_single_vs_batch = False
    config.verification.analyze_gradients = True            # Keep analysis
    config.verification.analyze_weight_changes = True
    config.verification.analyze_action_distributions = True
    config.verification.batch_verification_enabled = False
    config.logging.experiment_name = "gradient_analysis"
    return config


def get_fast_debug_config() -> TrainingConfig:
    """Minimal verification for quick debugging"""
    config = TrainingConfig()
    config.verification.enabled = True
    config.verification.verify_structural_integrity = True
    config.verification.verify_model_outputs = True
    config.verification.verify_single_vs_batch = False      # Skip slow checks
    config.verification.verify_buffer_integrity = False
    config.verification.analyze_gradients = False
    config.verification.analyze_weight_changes = False
    config.verification.batch_verification_enabled = False
    config.logging.experiment_name = "fast_debug"
    config.ppo.batch_size = 8  # Smaller batches for faster iteration
    return config


def get_stable_learning_config() -> TrainingConfig:
    """Configuration designed to fix value function instability"""
    config = TrainingConfig()
    
    # AGGRESSIVE gradient stability settings
    config.ppo.learning_rate = 3e-5           # Much more conservative
    config.ppo.clip_eps = 0.05                # Very tight clipping
    config.ppo.gradient_clip_norm = 0.25      # Aggressive gradient clipping
    config.ppo.value_loss_coeff = 0.1         # Much lower value loss weight
    config.ppo.ppo_epochs = 1                 # Single epoch to avoid overtraining
    config.ppo.adaptive_epochs = False        # Disable adaptive
    
    # Normalize advantages to prevent explosion
    config.ppo.normalize_advantages = True
    config.ppo.normalize_rewards = True
    
    # Enable key verification to monitor stability
    config.verification.enabled = True
    config.verification.verify_structural_integrity = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    config.verification.analyze_action_distributions = False
    config.verification.batch_verification_enabled = False
    
    config.logging.experiment_name = "gradient_stability_fix"
    return config


def get_optimized_stable_config() -> TrainingConfig:
    """Optimized configuration building on gradient stability success"""
    config = TrainingConfig()
    
    # Gradually increase learning while maintaining stability
    config.ppo.learning_rate = 5e-5           # Modest increase from 3e-5
    config.ppo.clip_eps = 0.08                # Slightly looser clipping
    config.ppo.gradient_clip_norm = 0.5       # Less aggressive clipping
    config.ppo.value_loss_coeff = 0.15        # Slightly higher value learning
    config.ppo.ppo_epochs = 2                 # Add second epoch for better learning
    config.ppo.adaptive_epochs = False        # Keep disabled for now
    
    # Maintain normalization
    config.ppo.normalize_advantages = True
    config.ppo.normalize_rewards = True
    
    # Enable verification to monitor continued stability
    config.verification.enabled = True
    config.verification.verify_structural_integrity = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    config.verification.analyze_action_distributions = True  # Enable for policy analysis
    config.verification.batch_verification_enabled = False
    
    config.logging.experiment_name = "optimized_stable_learning"
    return config


def get_conservative_multi_epoch_config() -> TrainingConfig:
    """Conservative configuration testing ONLY multiple epochs, keeping all other parameters identical to stable_learning"""
    config = TrainingConfig()
    
    # IDENTICAL to stable_learning except for epochs
    config.ppo.learning_rate = 3e-5           # SAME as stable_learning
    config.ppo.clip_eps = 0.05                # SAME as stable_learning  
    config.ppo.gradient_clip_norm = 0.25      # SAME as stable_learning
    config.ppo.value_loss_coeff = 0.1         # SAME as stable_learning
    config.ppo.ppo_epochs = 2                 # ONLY CHANGE: 1 -> 2 epochs
    config.ppo.adaptive_epochs = False        # SAME as stable_learning
    
    # IDENTICAL normalization settings
    config.ppo.normalize_advantages = True
    config.ppo.normalize_rewards = True
    
    # IDENTICAL verification settings  
    config.verification.enabled = True
    config.verification.verify_structural_integrity = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    config.verification.analyze_action_distributions = True  # Keep enabled for monitoring
    config.verification.batch_verification_enabled = False
    
    config.logging.experiment_name = "conservative_multi_epoch_test"
    return config


def get_value_regularized_multi_epoch_config() -> TrainingConfig:
    """Multi-epoch configuration with enhanced value function regularization"""
    config = TrainingConfig()
    
    # Multi-epoch with value regularization
    config.ppo.learning_rate = 3e-5           # Keep conservative learning rate
    config.ppo.clip_eps = 0.05                # Keep tight clipping
    config.ppo.gradient_clip_norm = 0.25      # Keep aggressive gradient clipping
    config.ppo.value_loss_coeff = 0.05        # MUCH lower value loss weight (0.1 -> 0.05)
    config.ppo.value_clip_range = 0.2         # Enable value clipping 
    config.ppo.ppo_epochs = 2                 # Multiple epochs
    config.ppo.batch_size = 32                # Larger batch for more stable gradients
    config.ppo.adaptive_epochs = False        # Keep disabled
    
    # Enhanced normalization
    config.ppo.normalize_advantages = True
    config.ppo.normalize_rewards = True
    
    # Enhanced verification for value tracking
    config.verification.enabled = True
    config.verification.verify_structural_integrity = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    config.verification.analyze_action_distributions = True
    config.verification.batch_verification_enabled = False
    
    config.logging.experiment_name = "value_regularized_multi_epoch"
    return config


def get_larger_batch_multi_epoch_config() -> TrainingConfig:
    """Multi-epoch configuration with larger batch size for gradient stability"""
    config = TrainingConfig()
    
    # Conservative multi-epoch with larger batches
    config.ppo.learning_rate = 3e-5           # Keep conservative learning rate
    config.ppo.clip_eps = 0.05                # Keep tight clipping
    config.ppo.gradient_clip_norm = 0.25      # Keep aggressive gradient clipping
    config.ppo.value_loss_coeff = 0.08        # Slightly lower value loss weight
    config.ppo.ppo_epochs = 2                 # Multiple epochs
    config.ppo.batch_size = 48                # Much larger batch (24 -> 48)
    config.ppo.adaptive_epochs = False        # Keep disabled
    
    # Enhanced normalization
    config.ppo.normalize_advantages = True
    config.ppo.normalize_rewards = True
    
    # Verification
    config.verification.enabled = True
    config.verification.verify_structural_integrity = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    config.verification.analyze_action_distributions = True
    config.verification.batch_verification_enabled = False
    
    config.logging.experiment_name = "larger_batch_multi_epoch"
    return config


def get_residual_percentage_fixed_gradients_config() -> TrainingConfig:
    """Configuration for residual model with 4 army percentage options and fixed gradient clipping"""
    config = TrainingConfig()
    
    # Model settings with percentage-based army selection (same as residual_percentage)
    config.model.model_type = "residual"  # Use stable residual model
    config.model.embed_dim = 64
    config.model.n_army_options = 4  # 4 percentage options: 25%, 50%, 75%, 100%
    config.model.max_army_send = 50  # Kept for backward compatibility
    
    # PPO settings - mostly same as residual_percentage but with FIXED gradient clipping
    config.ppo.learning_rate = 1e-4  # Same as working period
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # 🎯 KEY FIX: Gradient clipping set to allow healthy gradients
    config.ppo.gradient_clip_norm = 15.0  # Based on analysis: mean ~15 during winning period
    
    # Entropy coefficients tuned for 4 army options (same as original)
    config.ppo.entropy_coeff_start = 0.05
    config.ppo.entropy_coeff_decay = 0.04
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1
    config.ppo.edge_entropy_coeff = 1
    config.ppo.army_entropy_coeff = 1  # Balanced for 4 options
    
    # Value and advantage settings (same as original)
    config.ppo.value_loss_coeff = 0.5  # Higher due to residual stability
    config.ppo.adaptive_epochs = True
    
    # Enhanced verification to monitor gradient health
    config.verification.enabled = True
    config.verification.detailed_logging = False  # Enable for gradient monitoring
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True  # IMPORTANT: Monitor gradient flow
    config.verification.analyze_weight_changes = False
    config.verification.verify_gae_computation = False
    config.verification.verify_model_outputs = False

    # Logging with descriptive name
    config.logging.experiment_name = "residual_percentage_fixed_gradients"
    config.logging.print_every_n_episodes = 50
    config.logging.log_gradient_metrics = True  # Enable gradient logging
    
    return config


def get_residual_percentage_boosted_learning_config() -> TrainingConfig:
    """Configuration for residual model with fixed gradients AND boosted learning rate"""
    config = TrainingConfig()
    
    # Model settings with percentage-based army selection (same as residual_percentage)
    config.model.model_type = "residual"  # Use stable residual model
    config.model.embed_dim = 64
    config.model.n_army_options = 4  # 4 percentage options: 25%, 50%, 75%, 100%
    config.model.max_army_send = 50  # Kept for backward compatibility
    
    # PPO settings - BOOSTED learning rate based on analysis
    config.ppo.learning_rate = 2e-4  # DOUBLED from 1e-4 based on analysis
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # 🎯 KEY FIX: Higher gradient clipping threshold for stronger learning signal
    config.ppo.gradient_clip_norm = 25.0  # Increased from 15.0 to allow even stronger gradients
    
    # Entropy coefficients tuned for 4 army options (same as original)
    config.ppo.entropy_coeff_start = 0.05
    config.ppo.entropy_coeff_decay = 0.04
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1
    config.ppo.edge_entropy_coeff = 1
    config.ppo.army_entropy_coeff = 1  # Balanced for 4 options
    
    # Value and advantage settings (same as original)
    config.ppo.value_loss_coeff = 0.5  # Higher due to residual stability
    config.ppo.adaptive_epochs = True
    
    # Enhanced verification to monitor gradient health
    config.verification.enabled = True
    config.verification.detailed_logging = False  # Enable for gradient monitoring
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True  # IMPORTANT: Monitor gradient flow
    config.verification.analyze_weight_changes = False
    config.verification.verify_gae_computation = False
    config.verification.verify_model_outputs = False

    # Logging with descriptive name
    config.logging.experiment_name = "residual_percentage_boosted_learning"
    config.logging.print_every_n_episodes = 50
    config.logging.log_gradient_metrics = True  # Enable gradient logging
    
    return config


# Configuration factory
class ConfigFactory:
    """Factory for creating and managing configurations"""
    
    @staticmethod
    def create(config_name: str = "production") -> TrainingConfig:
        """Create a configuration by name"""
        configs = {
            "production": get_production_config,
            "debug": get_debug_config,
            "analysis": get_analysis_config,
            "fast_debug": get_fast_debug_config,
            "stable_learning": get_stable_learning_config,
            "optimized_stable": get_optimized_stable_config,
            "conservative_multi_epoch": get_conservative_multi_epoch_config,
            "value_regularized_multi_epoch": get_value_regularized_multi_epoch_config,
            "larger_batch_multi_epoch": get_larger_batch_multi_epoch_config,
            "residual_model": get_residual_model_config,
            "sage_model": get_sage_model_config,
            "sage_model_decisive": get_sage_model_decisive_config,
            "transformer_model": get_transformer_model_config,
            "transformer_model_decisive": get_transformer_model_decisive_config,
            "transformer_large_high_entropy": get_transformer_larger_balanced_entropy_config,
            "transformer_edge_features": get_transformer_edge_features_config,

            "residual_low_entropy": get_residual_low_entropy_config,
            "residual_percentage_fixed_gradients": get_residual_percentage_fixed_gradients_config,
            "residual_percentage_boosted_learning": get_residual_percentage_boosted_learning_config,
        }
        
        if config_name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        
        return configs[config_name]()
    
    @staticmethod
    def create_custom(**kwargs) -> TrainingConfig:
        """Create a configuration with custom overrides"""
        config = TrainingConfig()
        
        # Apply nested overrides
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested attributes like 'ppo.learning_rate'
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, key, value)
        
        return config
    
    @staticmethod
    def list_available() -> list[str]:
        """List all available pre-defined configurations"""
        return [
            "production", "debug", "analysis", "fast_debug", "stable_learning", 
            "optimized_stable", "conservative_multi_epoch", "value_regularized_multi_epoch", 
            "larger_batch_multi_epoch", "residual_model", "sage_model", "sage_model_decisive", 
            "transformer_model", "transformer_model_decisive", "transformer_high_lr",
            "transformer_larger_model", "transformer_fixed_multi_epoch",
            "residual_low_entropy", "reduced_army_send", "residual_percentage", 
            "residual_percentage_fixed_gradients", "residual_percentage_boosted_learning", 
            "residual_percentage_unclipped_gradients"
        ]


def get_residual_model_config() -> TrainingConfig:
    """Configuration optimized for ResGCN model architecture"""
    config = TrainingConfig()
    
    # Model settings
    config.model.model_type = "residual"
    config.model.embed_dim = 64
    
    # PPO settings optimized for residual networks
    config.ppo.learning_rate = 1e-4  # Higher LR due to stability
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 1.0  # Can handle higher gradients
    config.ppo.value_loss_coeff = 0.2  # Higher value loss coefficient
    config.ppo.value_clip_range = 0.3  # Value clipping enabled
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # Aggressive entropy schedule for decisiveness
    config.ppo.entropy_coeff_start = 1.0  # Start higher for initial exploration
    config.ppo.entropy_coeff_decay = 0.9  # Decay more aggressively (90% reduction)
    config.ppo.entropy_decay_episodes = 5000  # Faster decay (5000 vs 15000)
    config.ppo.placement_entropy_coeff = 0.2  # Double placement entropy weight
    config.ppo.edge_entropy_coeff = 0.3  # Triple edge entropy weight
    config.ppo.army_entropy_coeff = 0.01  # Increase army entropy weight
    
    # Verification
    config.verification.enabled = True
    config.verification.gradient_norm_threshold = 1000.0  # Higher threshold
    config.verification.detailed_logging = False  # Quiet logging to reduce verbosity
    config.verification.batch_verification_enabled = False
    
    # Logging
    config.logging.experiment_name = "residual_model_decisive"
    config.logging.log_frequency = 50
    
    return config


def get_residual_low_entropy_config() -> TrainingConfig:
    """Configuration for residual model with very aggressive entropy reduction"""
    config = TrainingConfig()
    
    # Model settings
    config.model.model_type = "residual"
    config.model.embed_dim = 64
    
    # PPO settings optimized for residual networks
    config.ppo.learning_rate = 1e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 1.0
    config.ppo.value_loss_coeff = 0.5
    config.ppo.value_clip_range = 0.3
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # Very aggressive entropy schedule for maximum decisiveness
    config.ppo.entropy_coeff_start = 0.2  # Start very high
    config.ppo.entropy_coeff_decay = 0.1  # Decay to near zero (90% reduction)
    config.ppo.entropy_decay_episodes = 1000  # Very fast decay (2000 episodes)
    config.ppo.placement_entropy_coeff = .1  # Very high placement entropy weight
    config.ppo.edge_entropy_coeff = .1  # Very high edge entropy weight
    config.ppo.army_entropy_coeff = .1  # High army entropy weight
    
    # Verification
    config.verification.enabled = False
    config.verification.gradient_norm_threshold = 1000.0
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    
    # Logging
    config.logging.experiment_name = "residual_model_ultra_decisive"
    config.logging.log_frequency = 50
    
    return config

def get_residual_low_entropy_config() -> TrainingConfig:
    """Configuration for residual model with very aggressive entropy reduction"""
    config = TrainingConfig()
    
    # Model settings
    config.model.model_type = "residual"
    config.model.embed_dim = 64
    
    # PPO settings optimized for residual networks
    config.ppo.learning_rate = 1e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 1.0
    config.ppo.value_loss_coeff = 0.5
    config.ppo.value_clip_range = 0.3
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # Very aggressive entropy schedule for maximum decisiveness
    config.ppo.entropy_coeff_start = 0.2  # Start very high
    config.ppo.entropy_coeff_decay = 0.1  # Decay to near zero (90% reduction)
    config.ppo.entropy_decay_episodes = 1000  # Very fast decay (2000 episodes)
    config.ppo.placement_entropy_coeff = .1  # Very high placement entropy weight
    config.ppo.edge_entropy_coeff = .1  # Very high edge entropy weight
    config.ppo.army_entropy_coeff = .1  # High army entropy weight
    
    # Verification
    config.verification.enabled = False
    config.verification.gradient_norm_threshold = 1000.0
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    
    # Logging
    config.logging.experiment_name = "residual_model_ultra_decisive"
    config.logging.log_frequency = 50
    
    return config

def get_sage_model_config() -> TrainingConfig:
    """Configuration optimized for GraphSAGE model architecture"""
    config = TrainingConfig()
    
    # Model settings
    config.model.model_type = "sage"
    config.model.embed_dim = 64

    # PPO settings optimized for SAGE
    # Entropy coefficients tuned for 4 army options
    config.ppo.entropy_coeff_start = 0.05
    config.ppo.entropy_coeff_decay = 0.04
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1
    config.ppo.edge_entropy_coeff = 1
    config.ppo.army_entropy_coeff = 1  # Balanced for 4 options

    # Value and advantage settings
    config.ppo.value_loss_coeff = 0.5  # Higher due to residual stability
    config.ppo.max_grad_norm = 1.0
    config.ppo.adaptive_epochs = True
    config.ppo.kl_threshold = 0.02

    # Early stopping
    config.ppo.early_stopping_enabled = True
    config.ppo.patience = 3
    config.ppo.min_improvement = 0.01

    # Verification
    config.verification.enabled = True
    config.verification.gradient_norm_threshold = 1000.0
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    config.verification.verify_gae_computation = False
    config.verification.verify_model_outputs = False

    # Logging
    config.logging.experiment_name = "sage_model_stable"
    config.logging.log_frequency = 50

    return config


def get_sage_model_decisive_config() -> TrainingConfig:
    """SAGE model configuration with reduced army entropy for decisive learning"""
    config = TrainingConfig()
    
    # Model settings
    config.model.model_type = "sage"
    config.model.embed_dim = 64

    # PPO settings optimized for SAGE with proper army entropy
    config.ppo.learning_rate = 1e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 5.0  # Slightly higher for SAGE
    config.ppo.value_loss_coeff = 0.5
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # Fixed entropy coefficients - properly scaled for 4 army options!
    config.ppo.entropy_coeff_start = 0.05
    config.ppo.entropy_coeff_decay = 0.04
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 0.2  # Moderate placement exploration
    config.ppo.edge_entropy_coeff = 0.3      # Moderate edge exploration  
    config.ppo.army_entropy_coeff = 0.05     # Scaled properly: 1.0 × (2.0/5.64) × base_factor

    # Adaptive training
    config.ppo.adaptive_epochs = True

    # Verification
    config.verification.enabled = True
    config.verification.detailed_logging = True
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True

    # Logging
    config.logging.experiment_name = "sage_model_decisive"
    config.logging.verbose_losses = False
    config.logging.verbose_rewards = False
    config.logging.print_every_n_episodes = 50
    
    return config


def get_transformer_model_config() -> TrainingConfig:
    """Configuration optimized for Transformer model architecture"""
    config = TrainingConfig()
    
    # Model settings
    config.model.model_type = "transformer"
    config.model.embed_dim = 64
    
    # PPO settings optimized for Transformers
    config.ppo.learning_rate = 1e-4  # Transformers typically need higher LR
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 16  # Smaller batch due to attention memory usage
    config.ppo.gradient_clip_norm = 1.0
    config.ppo.value_loss_coeff = 0.1  # Conservative for attention weights
    config.ppo.value_clip_range = 0.2
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # Verification
    config.verification.enabled = True
    config.verification.gradient_norm_threshold = 800.0
    config.verification.detailed_logging = True
    config.verification.batch_verification_enabled = False
    
    # Logging
    config.logging.experiment_name = "transformer_model_stable"
    config.logging.log_frequency = 50
    
    return config


def get_transformer_model_decisive_config() -> TrainingConfig:
    """Transformer model configuration based on sage_model_decisive with enhanced checkpointing"""
    config = TrainingConfig()
    
    # Model settings - Transformer architecture
    config.model.model_type = "transformer"
    config.model.embed_dim = 64

    # PPO settings - based on sage_model_decisive but adapted for Transformer
    config.ppo.learning_rate = 1e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32  # Use same batch size as other working configs
    config.ppo.gradient_clip_norm = 5.0  # Same as sage_model_decisive
    config.ppo.value_loss_coeff = 0.5
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    
    # Fixed entropy coefficients - same as sage_model_decisive
    config.ppo.entropy_coeff_start = 0.05
    config.ppo.entropy_coeff_decay = 0.04
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1  # Moderate placement exploration
    config.ppo.edge_entropy_coeff = 1      # Moderate edge exploration  
    config.ppo.army_entropy_coeff = 1     # Properly scaled for 4 army options

    # Adaptive training
    config.ppo.adaptive_epochs = True

    # Enhanced checkpointing configuration
    config.logging.save_checkpoints = True
    config.logging.checkpoint_every_n_episodes = 100  # Save every 100 episodes as requested
    config.logging.keep_last_n_checkpoints = 10       # Keep more checkpoints for analysis
    
    # Automatic resume capabilities
    config.logging.auto_resume_latest = True
    config.logging.resume_experiment_name = "transformer_decisive_experiment"
    
    # Comprehensive checkpoint loading
    config.logging.load_model_state = True
    config.logging.load_optimizer_state = True
    config.logging.load_reward_normalizer = True  # Critical for continued training
    config.logging.load_game_number = True
    config.logging.load_stat_trackers = True
    config.logging.load_training_state = True

    # Verification - same as sage_model_decisive
    config.verification.enabled = True
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = False

    # Logging with specific experiment name for transformer
    config.logging.experiment_name = "transformer_decisive_experiment"
    config.logging.verbose_losses = False
    config.logging.verbose_rewards = False
    config.logging.print_every_n_episodes = 50
    
    return config


def get_transformer_larger_high_entropy_config() -> TrainingConfig:
    """Transformer config with larger model and high entropy coefficients for stability."""
    config = TrainingConfig()
    # Larger transformer model
    config.model.model_type = "transformer"
    config.model.embed_dim = 128
    config.model.hidden_channels = 128  # If used in your architecture
    # PPO settings
    config.ppo.learning_rate = 1e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 5.0
    config.ppo.value_loss_coeff = 2.0  # Increased to keep value loss competitive
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    # High entropy coefficients
    config.ppo.entropy_coeff_start = 1.0
    config.ppo.entropy_coeff_decay = 0.8
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1.0
    config.ppo.edge_entropy_coeff = 1.0
    config.ppo.army_entropy_coeff = 1.0
    # Adaptive epochs enabled
    config.ppo.adaptive_epochs = True
    
    # Logging and checkpointing
    config.logging.save_checkpoints = True
    config.logging.checkpoint_every_n_episodes = 100
    config.logging.keep_last_n_checkpoints = 10
    config.logging.auto_resume_latest = True
    config.logging.experiment_name = "transformer_larger_high_entropy_experiment"
    config.logging.verbose_losses = True
    config.logging.verbose_rewards = False
    config.logging.print_every_n_episodes = 25
    # Verification
    config.verification.enabled = True
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    return config


def get_transformer_larger_balanced_entropy_config() -> TrainingConfig:
    """Transformer config with larger model and balanced entropy coefficients (0.3 → 0.1)."""
    config = TrainingConfig()
    # Larger transformer model
    config.model.model_type = "transformer"
    config.model.embed_dim = 128
    config.model.hidden_channels = 128  # If used in your architecture
    # PPO settings
    config.ppo.learning_rate = 2e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 15.0
    config.ppo.value_loss_coeff = 0.5
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    # Balanced entropy coefficients
    config.ppo.entropy_coeff_start = 0.03
    config.ppo.entropy_coeff_decay = 0.02  # Decays from 0.03 to 0.01 over 10000 episodes
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1
    config.ppo.edge_entropy_coeff = 1
    config.ppo.army_entropy_coeff = 1
    # Adaptive epochs enabled
    config.ppo.adaptive_epochs = True
    # Logging and checkpointing
    config.logging.save_checkpoints = True
    config.logging.checkpoint_every_n_episodes = 500
    config.logging.keep_last_n_checkpoints = 10
    config.logging.auto_resume_latest = True
    config.logging.experiment_name = "transformer_larger_balanced_entropy_experiment"
    config.logging.verbose_losses = False
    config.logging.verbose_rewards = False
    config.logging.print_every_n_episodes = 25
    # Verification
    config.verification.enabled = True
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    return config

def get_transformer_edge_features_config() -> TrainingConfig:
    """Transformer config with larger model and balanced entropy coefficients (0.3 → 0.1)."""
    config = TrainingConfig()
    # Larger transformer model
    config.model.model_type = "transformer"
    config.model.embed_dim = 128
    config.model.hidden_channels = 128  # If used in your architecture
    # PPO settings
    config.ppo.learning_rate = 2e-4
    config.ppo.ppo_epochs = 2
    config.ppo.batch_size = 32
    config.ppo.gradient_clip_norm = 15.0
    config.ppo.value_loss_coeff = 0.25
    config.ppo.clip_eps = 0.2
    config.ppo.gamma = 0.99
    config.ppo.lam = 0.95
    # Balanced entropy coefficients
    config.ppo.entropy_coeff_start = 0.01
    config.ppo.entropy_coeff_decay = 0.008  # Decays from 0.03 to 0.01 over 10000 episodes
    config.ppo.entropy_decay_episodes = 10000
    config.ppo.placement_entropy_coeff = 1
    config.ppo.edge_entropy_coeff = 1
    config.ppo.army_entropy_coeff = 1
    # Adaptive epochs enabled
    config.ppo.adaptive_epochs = True
    # Logging and checkpointing
    config.logging.save_checkpoints = True
    config.logging.checkpoint_every_n_episodes = 500
    config.logging.keep_last_n_checkpoints = 10
    config.logging.auto_resume_latest = True
    config.logging.experiment_name = "transformer_edge_features_experiment"
    config.logging.verbose_losses = False
    config.logging.verbose_rewards = False
    config.logging.print_every_n_episodes = 25
    # Verification
    config.verification.enabled = True
    config.verification.detailed_logging = False
    config.verification.batch_verification_enabled = False
    config.verification.analyze_gradients = True
    config.verification.analyze_weight_changes = True
    return config

