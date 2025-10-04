import torch
import unittest
from src.game.Phase import Phase

# Use your actual model here
from src.agents.RLUtils.WarlightModelAutoregressiveTransformer import WarlightPolicy

class TestRecomputeTurnLogprobsBatch(unittest.TestCase):
    def test_batch_mode_episode(self):
        """
        Test recompute_turn_logprobs in batch mode for multiple episode steps
        """
        batch_size = 4
        node_feats_batch = torch.stack([torch.randn(self.N, self.H) for _ in range(batch_size)])
        edge_feats_batch = torch.stack([torch.randn(self.E, self.H) for _ in range(batch_size)])
        ownership_batch = torch.stack([self.ownership_mask.clone() for _ in range(batch_size)])
        armies_batch = torch.stack([self.armies_left_before.clone() for _ in range(batch_size)])

        results = []
        for b in range(batch_size):
            # Only place on owned nodes
            owned_nodes = torch.where(ownership_batch[b])[0].tolist()
            if len(owned_nodes) == 0:
                placements = []
            else:
                placements = [owned_nodes[i % len(owned_nodes)] for i in range(3)]

            # Only attack from owned nodes with >1 army
            src_nodes = self.edge_index[0]
            valid_edge_indices = []
            for i, src in enumerate(src_nodes):
                if ownership_batch[b][src] and armies_batch[b][src] > 1:
                    valid_edge_indices.append(i)
            if len(valid_edge_indices) < 2:
                attacks = {'edge_idx': [], 'frac_idx': [], 'army_count': []}
            else:
                attacks = {
                    'edge_idx': valid_edge_indices[:2],
                    'frac_idx': [1, 2],
                    'army_count': [1, 1]
                }

            result = self.model.recompute_turn_logprobs(
                node_feats=node_feats_batch[b],
                edge_feats=edge_feats_batch[b],
                ownership_mask=ownership_batch[b],
                placements_list=placements,
                attacks_recorded=attacks,
                armies_left_before=armies_batch[b],
                action=None
            )
            results.append(result)

        for result in results:
            self.assertTrue(torch.isfinite(result["joint_logp_new"]))
            self.assertTrue(torch.isfinite(result["joint_ent_new"]))
            self.assertTrue(torch.isfinite(result["value"]))
            
    def setUp(self):
        torch.manual_seed(42)

        # Graph parameters
        self.N = 5   # nodes
        self.E = 7   # edges
        self.H = 16  # node/edge feature dim
        self.hidden_dim = 32  # model hidden dim

        # Initialize policy model
        self.model = WarlightPolicy(
            node_dim=self.H,
            edge_dim=self.H,
            num_nodes=self.N,
            num_edges=self.E,
            hidden_dim=self.hidden_dim,
            skip_residuals=False
        )

        # Edge index (source, target)
        self.edge_index = torch.randint(0, self.N, (2, self.E))
        self.model.edge_index = self.edge_index.clone()
        # Create a test state
        self.node_feats = torch.randn(self.N, self.H)
        self.edge_feats = torch.randn(self.E, self.H)
        
        # Ownership mask and armies_left per node (initial)
        self.ownership_mask = torch.tensor([1,1,1,0,0], dtype=torch.bool)
        self.armies_left_before = torch.tensor([5,3,4,2,1], dtype=torch.int)

        # Create realistic 
        # actions (place armies on owned nodes)
        self.placements_list = [0, 0, 1, 2, 1]  # place 2 armies on node 0, 2 on node 1, 1 on node 2
        
        # Find valid edges for attacks (edges from owned nodes with sufficient armies)
        src_nodes = self.edge_index[0]
        valid_edge_indices = []
        for i, src in enumerate(src_nodes):
            if self.ownership_mask[src] and self.armies_left_before[src] > 1:  # need armies to move
                valid_edge_indices.append(i)
        
        # If no valid edges, create some by modifying the state
        if len(valid_edge_indices) == 0:
            # Make sure edge 0 is from an owned node
            self.edge_index[0, 0] = 0  # set source of first edge to node 0 (owned)
            self.edge_index[1, 0] = 3  # set target to node 3 (not owned)
            valid_edge_indices = [0]
        
        # Take first few valid edges
        valid_edges = valid_edge_indices[:3] if len(valid_edge_indices) >= 3 else valid_edge_indices
        
        # Create realistic attack actions
        self.attacks_recorded = {
            'edge_idx': valid_edges,                    # valid edge indices for attacks
            'frac_idx': [1, 2, 0][:len(valid_edges)],  # fraction bin choices
            'army_count': [1, 1, 1][:len(valid_edges)] # conservative army counts
        }

    def test_recompute_turn_logprobs_both(self):
        """
        Test recompute_turn_logprobs for both placements and attacks (default)
        """
        result = self.model.recompute_turn_logprobs(
            node_feats=self.node_feats,
            edge_feats=self.edge_feats,
            ownership_mask=self.ownership_mask,
            placements_list=self.placements_list,
            attacks_recorded=self.attacks_recorded,
            armies_left_before=self.armies_left_before,
            action=None
        )
        expected_keys = [
            "place_logp_steps", "place_ent_steps",
            "attack_logp_steps", "attack_ent_steps",
            "placement_logp_new", "attack_logp_new",
            "joint_logp_new", "joint_ent_new", "value"
        ]
        for key in expected_keys:
            self.assertIn(key, result)
        self.assertTrue(torch.isfinite(result["joint_logp_new"]))
        self.assertTrue(torch.isfinite(result["joint_ent_new"]))
        self.assertTrue(torch.isfinite(result["value"]))
        self.assertEqual(len(result["place_logp_steps"]), len(self.placements_list))
        self.assertEqual(len(result["attack_logp_steps"]), len(self.attacks_recorded['edge_idx']))

    def test_empty_actions(self):
        """
        Test with empty actions (both)
        """
        result = self.model.recompute_turn_logprobs(
            node_feats=self.node_feats,
            edge_feats=self.edge_feats,
            ownership_mask=self.ownership_mask,
            placements_list=[],
            attacks_recorded={'edge_idx': [], 'frac_idx': [], 'army_count': []},
            armies_left_before=self.armies_left_before,
            action=None
        )
        self.assertTrue(torch.isfinite(result["joint_logp_new"]))
        self.assertTrue(torch.isfinite(result["joint_ent_new"]))
        self.assertEqual(result["placement_logp_new"].item(), 0.0)
        self.assertEqual(result["attack_logp_new"].item(), 0.0)

    def test_only_placement_actions(self):
        """
        Test with only placement actions, no attacks
        """
        result = self.model.recompute_turn_logprobs(
            node_feats=self.node_feats,
            edge_feats=self.edge_feats,
            ownership_mask=self.ownership_mask,
            placements_list=self.placements_list,
            attacks_recorded={'edge_idx': [], 'frac_idx': [], 'army_count': []},
            armies_left_before=self.armies_left_before,
            action=Phase.PLACE_ARMIES
        )
        # Should only return placement keys
        expected_keys = ["place_logp_steps", "place_ent_steps", "placement_logp_new", "joint_logp_new", "joint_ent_new", "value"]
        for key in expected_keys:
            self.assertIn(key, result)
        self.assertTrue(torch.isfinite(result["joint_logp_new"]))
        self.assertTrue(result["placement_logp_new"].item() != 0.0)
        self.assertEqual(result["joint_logp_new"].item(), result["placement_logp_new"].item())
        self.assertNotIn("attack_logp_new", result)
    def test_only_attack_actions(self):
        """
        Test with only attack actions, no placements
        """
        result = self.model.recompute_turn_logprobs(
            node_feats=self.node_feats,
            edge_feats=self.edge_feats,
            ownership_mask=self.ownership_mask,
            placements_list=[],
            attacks_recorded=self.attacks_recorded,
            armies_left_before=self.armies_left_before,
            action=Phase.ATTACK_TRANSFER
        )
        expected_keys = ["attack_logp_steps", "attack_ent_steps", "attack_logp_new", "joint_logp_new", "joint_ent_new", "value"]
        for key in expected_keys:
            self.assertIn(key, result)
        self.assertTrue(torch.isfinite(result["joint_logp_new"]))
        self.assertTrue(result["attack_logp_new"].item() != 0.0)
        self.assertEqual(result["joint_logp_new"].item(), result["attack_logp_new"].item())
        self.assertNotIn("placement_logp_new", result)

if __name__ == "__main__":
    unittest.main()
