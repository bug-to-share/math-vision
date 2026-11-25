"""MoCA changes the already-computed GRPO advantage, not the PPO objective."""

import torch


def route_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    perception_mask: torch.Tensor,
    outcome: torch.Tensor,
    perception: torch.Tensor,
    protect: float = 0.2,
    punish: float = 0.2,
    route_only_negative_advantages: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not 0 <= protect <= 1 or not 0 <= punish <= 1:
        raise ValueError("MoCA protection and punishment must be in [0, 1]")
    if advantages.shape != response_mask.shape or advantages.shape != perception_mask.shape:
        raise ValueError("MoCA advantage and token-mask shapes differ")
    if outcome.shape != advantages.shape[:1] or perception.shape != outcome.shape:
        raise ValueError("MoCA reward shape does not match the response batch")
    if not torch.isfinite(advantages).all() or not torch.isfinite(outcome).all():
        raise ValueError("Non-finite MoCA advantages or outcome rewards")
    if not ((0 <= outcome) & (outcome <= 1)).all() or not ((perception == 0) | (perception == 1)).all():
        raise ValueError("MoCA requires outcome in [0, 1] and binary perception rewards")
    valid = response_mask.bool()
    target = perception_mask.bool() & valid
    failed = outcome.eq(0).unsqueeze(1)
    routable = target & failed
    if route_only_negative_advantages:
        routable &= advantages < 0
    protected = routable & perception.eq(1).unsqueeze(1)
    punished = routable & perception.eq(0).unsqueeze(1)
    delta = advantages.abs() * (protect * protected - punish * punished)
    result = (advantages + delta) * valid
    denominator = max(int(valid.sum()), 1)
    metrics = {
        "moca/perception_token_fraction": float(target.sum()) / denominator,
        "moca/protected_token_fraction": float(protected.sum()) / denominator,
        "moca/punished_token_fraction": float(punished.sum()) / denominator,
        "moca/advantage_delta_abs_mean": float(delta.abs().sum()) / denominator,
        "moca/outcome_mean": float(outcome.mean()),
        "moca/perception_mean": float(perception.float().mean()),
        "moca/failed_positive_advantage_fraction": float((failed & (advantages > 0) & valid).sum()) / denominator,
    }
    return result, metrics
