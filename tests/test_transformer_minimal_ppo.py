import torch
import torch.nn.functional as F

from src.agents.RLUtils.WarlightModelAutoregressiveTransformer import (
    WarlightPolicy,
)
from src.game.Phase import Phase


def _ring_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    src = torch.arange(num_nodes, device=device)
    dst = torch.roll(src, shifts=-1)
    return torch.stack([src, dst], dim=0)


def test_transformer_minimal_ppo_loop():
    device = torch.device("cpu")
    torch.manual_seed(1234)

    num_nodes = 8
    node_dim = 6
    edge_dim = 4
    num_edges = num_nodes

    policy = WarlightPolicy(
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_nodes=num_nodes,
        num_edges=num_edges,
        hidden_dim=64,
    ).to(device)
    policy.edge_index = _ring_edge_index(num_nodes, device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    node_feats = torch.randn(num_nodes, node_dim, device=device)
    node_feats[:, 0] = 5.0  # ensure positive armies feature

    edge_feats = torch.randn(num_edges, edge_dim, device=device)
    ownership_mask = torch.zeros(num_nodes, dtype=torch.int, device=device)
    ownership_mask[: num_nodes // 2] = 1
    armies_left = torch.full((num_nodes,), 5.0, device=device)

    n_available_armies = int(torch.randint(3, 6, (1,), device=device))

    placement_out = policy(
        node_feats,
        edge_feats,
        ownership_mask,
        n_available_armies,
        armies_left.clone(),
        Phase.PLACE_ARMIES,
    )
    placements_list = placement_out["placements_list"]
    assert len(placements_list) > 0, "Expected at least one placement"

    placement_counts = placement_out["placement_counts"].to(armies_left.dtype)
    armies_after_placements = armies_left + placement_counts

    attack_out = policy(
        node_feats,
        edge_feats,
        ownership_mask,
        n_available_armies,
        armies_after_placements.clone(),
        Phase.ATTACK_TRANSFER,
    )
    attacks_recorded = attack_out["attacks"]

    joint_logp_old = (placement_out["logp"] + attack_out["logp"]).detach()

    returns = torch.tensor(0.5, device=device)
    advantages = torch.tensor(0.3, device=device)

    clip_eps = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01

    for epoch in range(3):
        optimizer.zero_grad(set_to_none=True)

        recompute = policy.recompute_turn_logprobs(
            node_feats,
            edge_feats,
            ownership_mask,
            placements_list,
            attacks_recorded,
            armies_left,
            action=None,
        )

        joint_logp_new = recompute["joint_logp_new"]
        joint_entropy = recompute["joint_ent_new"]
        value_pred = recompute["value"]

        ratio = torch.exp(joint_logp_new - joint_logp_old)
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -torch.min(ratio * advantages, ratio_clipped * advantages)

        value_loss = F.mse_loss(value_pred, returns)
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * joint_entropy

        total_loss.backward()
        optimizer.step()

        assert torch.isfinite(total_loss), "Loss produced NaN or Inf"

    # Ensure parameters received gradients
    has_grad = any(p.grad is not None for p in policy.parameters())
    assert has_grad, "Expected gradients to flow through policy parameters"
