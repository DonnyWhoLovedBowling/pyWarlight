from typing import Literal
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.agents.RLUtils.WarlightModelResidual import WarlightPolicyNetResidual
from src.agents.RLUtils.WarlightModelSAGE import WarlightPolicyNetSAGE
from src.agents.RLUtils.WarlightModelTransformer import WarlightPolicyNetTransformer

ModelType = Literal['standard', 'residual', 'sage', 'transformer']

class ModelFactory:
    """Factory for creating different model architectures optimized for gradient stability"""
    
    @staticmethod
    def create_model(
        model_type: ModelType,
        node_feat_dim: int = 8,
        embed_dim: int = 64,
        max_army_send: int = 50
    ):
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model architecture
            node_feat_dim: Number of input features per node
            embed_dim: Hidden embedding dimension
            max_army_send: Maximum armies that can be sent in one attack
            
        Returns:
            Model instance
        """
        if model_type == 'standard':
            return WarlightPolicyNet(node_feat_dim, embed_dim, max_army_send)
        
        elif model_type == 'residual':
            return WarlightPolicyNetResidual(node_feat_dim, embed_dim, max_army_send)
        
        elif model_type == 'sage':
            return WarlightPolicyNetSAGE(node_feat_dim, embed_dim, max_army_send)
        
        elif model_type == 'transformer':
            return WarlightPolicyNetTransformer(node_feat_dim, embed_dim, max_army_send=max_army_send)
        
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
