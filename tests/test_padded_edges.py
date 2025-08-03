import torch
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet

def test_padded_edges():
    """Test if model can handle padded edges"""
    # Initialize model
    model = WarlightPolicyNet(
        node_feat_dim=8,
        embed_dim=64,
        max_army_send=42
    )
    model.eval()
    
    # Create test data with padding
    node_features = torch.randn(42, 8)  # 42 nodes, 8 features
    
    # Valid edges + padding
    valid_edges = torch.tensor([[0, 1], [1, 2], [2, 3]])  # 3 valid edges
    padding = torch.full((39, 2), -1)  # 39 padded edges with -1
    padded_edges = torch.cat([valid_edges, padding], dim=0)  # Total 42 edges
    
    army_counts = node_features[:, -1]
    
    try:
        # Test if model handles padded edges
        placement_logits, attack_logits, army_logits = model(
            node_features, padded_edges, army_counts
        )
        print("✅ Model can handle padded edges")
        print(f"Placement logits shape: {placement_logits.shape}")
        print(f"Attack logits shape: {attack_logits.shape}")
        print(f"Army logits shape: {army_logits.shape}")
        return True
    except Exception as e:
        print(f"❌ Model cannot handle padded edges: {e}")
        return False


if __name__ == "__main__":
    test_padded_edges()
