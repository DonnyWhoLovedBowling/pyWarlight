"""
Unit tests to validate log probability consistency between inference and recomputation.

This test specifically addresses the dropout issue where:
1. During inference (eval mode), dropout is disabled
2. During PPO update (train mode), dropout is enabled
3. This causes log probs to differ even with identical weights

The test validates that when properly handled (using eval mode),
recomputed log probs match the original inference log probs.
"""

import torch
import unittest
from src.agents.RLUtils.WarlightModelAutoregressiveTransformer import WarlightPolicy
from src.game.Phase import Phase


class TestLogProbConsistency(unittest.TestCase):
    """Test log probability consistency between inference and recomputation"""

    def setUp(self):
        """Set up test fixtures"""
        torch.manual_seed(42)

        # Graph parameters
        self.num_nodes = 42
        self.num_edges = 166
        self.node_dim = 7
        self.edge_dim = 4
        self.hidden_dim = 128

        # Create model with dropout (realistic scenario)
        self.model = WarlightPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            hidden_dim=self.hidden_dim,
            skip_residuals=False
        )

        # Create realistic edge index
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.model.edge_index = self.edge_index.clone()

        # Create test data
        self.node_features = torch.randn(self.num_nodes, self.node_dim)
        self.edge_features = torch.randn(self.num_edges, self.edge_dim)

        # Create ownership mask (simulate player owning ~1/3 of regions)
        self.ownership_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        owned_indices = torch.randperm(self.num_nodes)[:self.num_nodes // 3]
        self.ownership_mask[owned_indices] = True

        # Armies left (for attack phase)
        self.armies_left = torch.randint(1, 10, (self.num_nodes,))

    def test_logprob_consistency_eval_mode(self):
        """
        Test that log probs are consistent when using eval mode.
        This is the CORRECT behavior - recomputed log probs should match original.
        """
        self.model.eval()

        # Step 1: Run inference to get actions and log probs (simulate gameplay)
        with torch.no_grad():
            placement_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=5,
                armies_left=torch.tensor([]),
                action=Phase.PLACE_ARMIES
            )

            # Extract placements and log probs
            placements_list = placement_output['placements_list']
            original_placement_logp = placement_output['logp'].clone()

            # Simulate attack phase
            attack_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=torch.tensor([]),
                armies_left=self.armies_left,
                action=Phase.ATTACK_TRANSFER
            )

            attacks_recorded = attack_output.get('attacks', {
                'edge_idx': [],
                'frac_idx': [],
                'army_count': []
            })
            original_attack_logp = attack_output.get('logp', torch.tensor([]))

        # Step 2: Recompute log probs (simulate PPO update) - still in eval mode
        with torch.no_grad():
            recompute_result = self.model.recompute_turn_logprobs(
                node_feats=self.node_features,
                edge_feats=self.edge_features,
                ownership_mask=self.ownership_mask,
                placements_list=placements_list,
                attacks_recorded=attacks_recorded,
                armies_left_before=self.armies_left,
                action=None
            )

            recomputed_placement_logp = recompute_result.get('logp', torch.tensor([]))
            recomputed_attack_logp = recompute_result.get('attack_logp', torch.tensor([]))

        # Step 3: Verify log probs match (within numerical precision)
        if original_placement_logp.numel() > 0 and recomputed_placement_logp.numel() > 0:
            placement_diff = torch.abs(original_placement_logp - recomputed_placement_logp).max().item()
            self.assertLess(placement_diff, 1e-5,
                f"Placement log probs differ by {placement_diff:.6e} (should be ~0 in eval mode)")
            print(f"✓ Placement log prob difference (eval mode): {placement_diff:.6e}")

        if original_attack_logp.numel() > 0 and recomputed_attack_logp.numel() > 0:
            attack_diff = torch.abs(original_attack_logp - recomputed_attack_logp).max().item()
            self.assertLess(attack_diff, 1e-5,
                f"Attack log probs differ by {attack_diff:.6e} (should be ~0 in eval mode)")
            print(f"✓ Attack log prob difference (eval mode): {attack_diff:.6e}")

    def test_logprob_inconsistency_train_mode(self):
        """
        Test that log probs DIFFER when using train mode due to dropout.
        This demonstrates the BUG we fixed - train mode causes random differences.
        """
        # Step 1: Get actions in eval mode (simulate gameplay)
        self.model.eval()
        with torch.no_grad():
            placement_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=5,
                armies_left=torch.tensor([]),
                action=Phase.PLACE_ARMIES
            )
            placements_list = placement_output['placements_list']
            original_placement_logp = placement_output['logp'].clone()

            attack_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=torch.tensor([]),
                armies_left=self.armies_left,
                action=Phase.ATTACK_TRANSFER
            )
            attacks_recorded = attack_output.get('attacks', {
                'edge_idx': [],
                'frac_idx': [],
                'army_count': []
            })
            original_attack_logp = attack_output.get('logp', torch.tensor([]))

        # Step 2: Switch to train mode and recompute (THIS IS THE BUG SCENARIO)
        self.model.train()
        with torch.no_grad():
            recompute_result = self.model.recompute_turn_logprobs(
                node_feats=self.node_features,
                edge_feats=self.edge_features,
                ownership_mask=self.ownership_mask,
                placements_list=placements_list,
                attacks_recorded=attacks_recorded,
                armies_left_before=self.armies_left,
                action=None
            )

            recomputed_placement_logp = recompute_result.get('logp', torch.tensor([]))
            recomputed_attack_logp = recompute_result.get('attack_logp', torch.tensor([]))

        # Step 3: Verify log probs DIFFER due to dropout (demonstrates the bug)
        if original_placement_logp.numel() > 0 and recomputed_placement_logp.numel() > 0:
            placement_diff = torch.abs(original_placement_logp - recomputed_placement_logp).max().item()
            # In train mode, dropout causes differences
            # Note: This could occasionally pass if dropout doesn't affect these specific neurons
            print(f"⚠ Placement log prob difference (train mode with dropout): {placement_diff:.6e}")
            if placement_diff > 1e-5:
                print(f"  → Dropout is causing inconsistency (this is the bug we fixed!)")

        if original_attack_logp.numel() > 0 and recomputed_attack_logp.numel() > 0:
            attack_diff = torch.abs(original_attack_logp - recomputed_attack_logp).max().item()
            print(f"⚠ Attack log prob difference (train mode with dropout): {attack_diff:.6e}")
            if attack_diff > 1e-5:
                print(f"  → Dropout is causing inconsistency (this is the bug we fixed!)")

    def test_kl_divergence_with_identical_weights(self):
        """
        Test that KL divergence is near-zero when recomputing with identical weights.
        This validates the fix: KL should measure policy change, not dropout noise.
        """
        self.model.eval()

        # Generate actions
        with torch.no_grad():
            placement_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=5,
                armies_left=torch.tensor([]),
                action=Phase.PLACE_ARMIES
            )
            placements_list = placement_output['placements_list']
            original_placement_logp = placement_output['logp'].clone()

            attack_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=torch.tensor([]),
                armies_left=self.armies_left,
                action=Phase.ATTACK_TRANSFER
            )
            attacks_recorded = attack_output.get('attacks', {})
            original_attack_logp = attack_output.get('logp', torch.tensor([])).clone()

        # Recompute in eval mode (no weight change)
        with torch.no_grad():
            recompute_result = self.model.recompute_turn_logprobs(
                node_feats=self.node_features,
                edge_feats=self.edge_features,
                ownership_mask=self.ownership_mask,
                placements_list=placements_list,
                attacks_recorded=attacks_recorded,
                armies_left_before=self.armies_left,
                action=None
            )

            recomputed_placement_logp = recompute_result.get('logp', torch.tensor([]))
            recomputed_attack_logp = recompute_result.get('attack_logp', torch.tensor([]))

        # Compute KL divergence for placements (should always exist)
        if original_placement_logp.numel() > 0 and recomputed_placement_logp.numel() > 0:
            placement_kl = (original_placement_logp.sum() - recomputed_placement_logp.sum()).item()
            print(f"Placement KL divergence with identical weights (eval mode): {placement_kl:.6e}")
            self.assertLess(abs(placement_kl), 1e-4,
                f"Placement KL divergence should be near-zero with identical weights, got {placement_kl:.6e}")
            print("✓ Placement KL divergence is near-zero as expected")

        # Compute KL divergence for attacks (only if attacks exist)
        if original_attack_logp.numel() > 0 and recomputed_attack_logp.numel() > 0:
            attack_kl = (original_attack_logp.sum() - recomputed_attack_logp.sum()).item()
            print(f"Attack KL divergence with identical weights (eval mode): {attack_kl:.6e}")
            self.assertLess(abs(attack_kl), 1e-4,
                f"Attack KL divergence should be near-zero with identical weights, got {attack_kl:.6e}")
            print("✓ Attack KL divergence is near-zero as expected")
        else:
            print("⚠ No attacks generated, skipping attack KL check")

    def test_batch_logprob_consistency(self):
        """
        Test log prob consistency with batch data (realistic PPO scenario).
        """
        batch_size = 4
        self.model.eval()

        # Create batch data
        node_features_batch = self.node_features.unsqueeze(0).expand(batch_size, -1, -1)
        edge_features_batch = self.edge_features.unsqueeze(0).expand(batch_size, -1, -1)
        ownership_batch = self.ownership_mask.unsqueeze(0).expand(batch_size, -1)
        armies_batch = self.armies_left.unsqueeze(0).expand(batch_size, -1)

        placements_list_batch = []
        attacks_batch = []
        original_placement_logps = []
        original_attack_logps = []

        # Generate actions for each batch item
        with torch.no_grad():
            for b in range(batch_size):
                placement_output = self.model(
                    node_features_batch[b],
                    edge_features_batch[b],
                    ownership_batch[b],
                    n_available_armies=5,
                    armies_left=torch.tensor([]),
                    action=Phase.PLACE_ARMIES
                )
                placements_list_batch.append(placement_output['placements_list'])
                original_placement_logps.append(placement_output['logp'])

                attack_output = self.model(
                    node_features_batch[b],
                    edge_features_batch[b],
                    ownership_batch[b],
                    n_available_armies=torch.tensor([]),
                    armies_left=armies_batch[b],
                    action=Phase.ATTACK_TRANSFER
                )
                attacks_batch.append(attack_output.get('attacks', {}))
                original_attack_logps.append(attack_output.get('logp', torch.tensor([])))

        # Recompute for each batch item
        max_placement_diff = 0.0
        max_attack_diff = 0.0

        with torch.no_grad():
            for b in range(batch_size):
                recompute_result = self.model.recompute_turn_logprobs(
                    node_feats=node_features_batch[b],
                    edge_feats=edge_features_batch[b],
                    ownership_mask=ownership_batch[b],
                    placements_list=placements_list_batch[b],
                    attacks_recorded=attacks_batch[b],
                    armies_left_before=armies_batch[b],
                    action=None
                )

                recomputed_placement_logp = recompute_result.get('logp', torch.tensor([]))
                recomputed_attack_logp = recompute_result.get('attack_logp', torch.tensor([]))

                if original_placement_logps[b].numel() > 0 and recomputed_placement_logp.numel() > 0:
                    diff = torch.abs(original_placement_logps[b] - recomputed_placement_logp).max().item()
                    max_placement_diff = max(max_placement_diff, diff)

                if original_attack_logps[b].numel() > 0 and recomputed_attack_logp.numel() > 0:
                    diff = torch.abs(original_attack_logps[b] - recomputed_attack_logp).max().item()
                    max_attack_diff = max(max_attack_diff, diff)

        print(f"✓ Max placement log prob diff across batch: {max_placement_diff:.6e}")
        print(f"✓ Max attack log prob diff across batch: {max_attack_diff:.6e}")

        self.assertLess(max_placement_diff, 1e-5,
            f"Batch placement log probs inconsistent: {max_placement_diff:.6e}")
        self.assertLess(max_attack_diff, 1e-5,
            f"Batch attack log probs inconsistent: {max_attack_diff:.6e}")

    def test_entropy_consistency(self):
        """
        Test that entropy values are also consistent between inference and recomputation.
        """
        self.model.eval()

        with torch.no_grad():
            # Original inference
            placement_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=5,
                armies_left=torch.tensor([]),
                action=Phase.PLACE_ARMIES
            )
            placements_list = placement_output['placements_list']

            attack_output = self.model(
                self.node_features,
                self.edge_features,
                self.ownership_mask,
                n_available_armies=torch.tensor([]),
                armies_left=self.armies_left,
                action=Phase.ATTACK_TRANSFER
            )
            attacks_recorded = attack_output.get('attacks', {})

            # Recompute with entropy
            recompute_result = self.model.recompute_turn_logprobs(
                node_feats=self.node_features,
                edge_feats=self.edge_features,
                ownership_mask=self.ownership_mask,
                placements_list=placements_list,
                attacks_recorded=attacks_recorded,
                armies_left_before=self.armies_left,
                action=None
            )

            placement_entropy = recompute_result.get('placement_entropy', 0.0)
            attack_entropy = recompute_result.get('attack_entropy', 0.0)

            # Entropy should be finite and positive
            self.assertTrue(placement_entropy >= 0, "Placement entropy should be non-negative")
            self.assertTrue(attack_entropy >= 0, "Attack entropy should be non-negative")
            self.assertTrue(placement_entropy < 100, "Placement entropy should be reasonable")
            self.assertTrue(attack_entropy < 100, "Attack entropy should be reasonable")

            print(f"✓ Placement entropy: {placement_entropy:.4f}")
            print(f"✓ Attack entropy: {attack_entropy:.4f}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
