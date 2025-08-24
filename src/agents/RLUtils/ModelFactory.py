from typing import Literal

from src.agents.RLUtils.TransformerActorCritic import TransformerActorCritic
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.agents.RLUtils.WarlightModelResidual import WarlightPolicyNetResidual
from src.agents.RLUtils.WarlightModelSAGE import WarlightPolicyNetSAGE
from src.agents.RLUtils.WarlightModelTransformer import WarlightPolicyNetTransformer

ModelType = Literal['standard', 'residual', 'sage', 'transformer']

class ModelFactory:
    """Factory for creating different model architectures optimized for gradient stability"""
    
    @staticmethod
    def create_model(config):
        """residual_low_entropy
        Create a model instance based on the specified type.
        
        Args:
            config: holds all necessary parameters:
            model_type: Type of model architecture,
            node_feat_dim: Number of input features per node,
            embed_dim: Hidden embedding dimension,
            n_army_options: Number of army percentage options (default 4: 25%, 50%, 75%, 100%)
            
        Returns:
            Model instance
        """
        model_type=config.model.model_type
        node_feat_dim=config.model.in_channels
        embed_dim=config.model.embed_dim
        n_army_options=getattr(config.model, 'n_army_options', 4)  # Default to 4 if not specified
        edge_feat_dim = getattr(config.model, 'edge_feat_dim', 0)

        if model_type == 'standard':
            return WarlightPolicyNetTransformer(node_feat_dim, embed_dim, n_army_options=n_army_options, edge_feat_dim=edge_feat_dim)
        
        elif model_type == 'residual':
            return WarlightPolicyNetResidual(node_feat_dim, embed_dim, n_army_options)
        
        elif model_type == 'sage':
            return WarlightPolicyNetSAGE(node_feat_dim, embed_dim, n_army_options)
        
        elif model_type == 'transformer':
            return WarlightPolicyNetTransformer(node_feat_dim, embed_dim, n_army_options=n_army_options, edge_feat_dim=edge_feat_dim)
        elif model_type == 'transformer_actor_critic':
            return TransformerActorCritic()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_recommended_config(model_type: ModelType) -> dict:
        """
        Get recommended training configuration for each model type.
        
        Returns:
            Dictionary with recommended hyperparameters
        """
        configs = {
            'standard': {
                'learning_rate': 3e-5,
                'gradient_clip_norm': 0.5,
                'value_loss_coeff': 0.1,
                'ppo_epochs': 1,
                'batch_size': 24
            },
            'residual': {
                'learning_rate': 1e-4,  # Can use higher LR due to stability
                'gradient_clip_norm': 1.0,
                'value_loss_coeff': 0.2,  # Can handle higher value loss
                'ppo_epochs': 2,
                'batch_size': 32
            },
            'sage': {
                'learning_rate': 5e-5,
                'gradient_clip_norm': 1.0,
                'value_loss_coeff': 0.15,
                'ppo_epochs': 2,
                'batch_size': 32
            },
            'transformer': {
                'learning_rate': 1e-4,  # Transformers typically need higher LR
                'gradient_clip_norm': 1.0,
                'value_loss_coeff': 0.1,  # Conservative for attention weights
                'ppo_epochs': 2,
                'batch_size': 16  # Attention is memory-intensive
            }
        }
        return configs[model_type]
    
    @staticmethod
    def get_stability_analysis(model_type: ModelType) -> dict:
        """
        Get stability characteristics and expected gradient behavior.
        
        Returns:
            Dictionary with stability information
        """
        analyses = {
            'standard': {
                'gradient_stability': 'Low',
                'expected_grad_norm': '100-1000',
                'explosion_risk': 'High',
                'benefits': 'Simple, fast, baseline',
                'drawbacks': 'Gradient explosion, vanishing gradients in deep networks'
            },
            'residual': {
                'gradient_stability': 'High',
                'expected_grad_norm': '10-100',
                'explosion_risk': 'Low',
                'benefits': 'Residual connections prevent vanishing gradients, LayerNorm adds stability',
                'drawbacks': 'Slightly more complex, more parameters'
            },
            'sage': {
                'gradient_stability': 'Medium-High',
                'expected_grad_norm': '20-200',
                'explosion_risk': 'Low-Medium',
                'benefits': 'Sampling-based aggregation is more stable, good for large graphs',
                'drawbacks': 'May lose some fine-grained spatial information'
            },
            'transformer': {
                'gradient_stability': 'Medium-High',
                'expected_grad_norm': '30-300',
                'explosion_risk': 'Medium',
                'benefits': 'Attention mechanism captures long-range dependencies, very stable with proper norm',
                'drawbacks': 'More memory intensive, attention weights can explode without clipping'
            }
        }
        return analyses[model_type]

def print_model_comparison():
    """Print a comparison of all available model architectures"""
    print("🏗️  MODEL ARCHITECTURE COMPARISON FOR WARLIGHT")
    print("=" * 60)
    
    for model_type in ['standard', 'residual', 'sage', 'transformer']:
        config = ModelFactory.get_recommended_config(model_type)
        analysis = ModelFactory.get_stability_analysis(model_type)
        
        print(f"\n📋 {model_type.upper()} MODEL")
        print(f"   Gradient Stability: {analysis['gradient_stability']}")
        print(f"   Expected Grad Norm: {analysis['expected_grad_norm']}")
        print(f"   Explosion Risk: {analysis['explosion_risk']}")
        print(f"   Recommended LR: {config['learning_rate']}")
        print(f"   Recommended Epochs: {config['ppo_epochs']}")
        print(f"   Recommended Batch: {config['batch_size']}")
        print(f"   Benefits: {analysis['benefits']}")
        print(f"   Drawbacks: {analysis['drawbacks']}")

if __name__ == "__main__":
    print_model_comparison()
