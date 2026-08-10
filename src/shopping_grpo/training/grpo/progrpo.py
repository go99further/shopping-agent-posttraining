"""Confidence-aware advantages for multi-turn shopping Agent trajectories.

This is an Agent-specific adaptation inspired by ProGRPO, not a verbatim
implementation of the reasoning-only paper.  Prompt confidence is replaced by
an intra-prompt confidence baseline, and only successful trajectories receive
the exploration reweighting.  Reward-v3 ordering remains the source of the base
advantage for unsuccessful and partially successful trajectories.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import torch
from verl.trainer.ppo.core_algos import register_adv_est


AGENT_PROGRPO_ESTIMATOR = "agent_progrpo"


def low_probability_token_confidence(
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    fraction: float = 0.2,
) -> torch.Tensor:
    """Return geometric-mean probability over the least-confident tokens."""
    if old_log_probs.shape != response_mask.shape:
        raise ValueError("old_log_probs and response_mask must have equal shapes")
    if old_log_probs.ndim != 2:
        raise ValueError("old_log_probs and response_mask must be rank-2 tensors")
    if not 0.0 < fraction <= 1.0 or not math.isfinite(fraction):
        raise ValueError("fraction must be finite and in (0, 1]")

    confidences = []
    for sample_log_probs, sample_mask in zip(
        old_log_probs, response_mask.bool(), strict=True
    ):
        selected = sample_log_probs[sample_mask]
        if selected.numel() == 0:
            raise ValueError("every trajectory must contain at least one response token")
        if not torch.isfinite(selected).all():
            raise ValueError("masked response log probabilities must be finite")
        token_count = max(1, math.ceil(selected.numel() * fraction))
        least_confident = torch.topk(
            selected,
            k=token_count,
            largest=False,
        ).values
        confidences.append(torch.exp(least_confident.mean()))
    return torch.stack(confidences)


@register_adv_est(AGENT_PROGRPO_ESTIMATOR)
def compute_agent_progrpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Hashable],
    old_log_probs: torch.Tensor,
    purchase_success: Sequence[bool],
    *,
    alpha: float = 0.3,
    low_probability_fraction: float = 0.2,
    epsilon: float = 1.0e-6,
    norm_adv_by_std_in_grpo: bool = False,
    config: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Reward-v3 GRPO advantages plus success-path exploration bias.

    Equal-reward successful groups receive a centered confidence signal: a
    lower-confidence successful path gets a larger advantage, while a dominant
    high-confidence path gets a smaller one.  Failed paths are never reweighted
    by confidence and therefore retain the original Reward-v3 ordering.
    """
    if config is not None:
        agent_config = config.get("agent_progrpo", {})
        alpha = float(agent_config.get("alpha", alpha))
        low_probability_fraction = float(
            agent_config.get("low_probability_fraction", low_probability_fraction)
        )
        norm_adv_by_std_in_grpo = bool(
            config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        )

    batch_size = token_level_rewards.shape[0]
    if response_mask.shape != token_level_rewards.shape:
        raise ValueError("token_level_rewards and response_mask must have equal shapes")
    if old_log_probs.shape != response_mask.shape:
        raise ValueError("old_log_probs and response_mask must have equal shapes")
    if len(index) != batch_size or len(purchase_success) != batch_size:
        raise ValueError("index and purchase_success must match the batch size")
    if alpha < 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite non-negative number")

    scores = token_level_rewards.sum(dim=-1)
    confidences = low_probability_token_confidence(
        old_log_probs,
        response_mask,
        fraction=low_probability_fraction,
    )
    grouped_indices: dict[Hashable, list[int]] = defaultdict(list)
    for sample_index, uid in enumerate(index):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"group uid is not hashable: {uid!r}") from exc
        grouped_indices[uid].append(sample_index)

    sequence_advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for group_indices in grouped_indices.values():
            group_scores = scores[group_indices]
            group_mean = group_scores.mean()
            if norm_adv_by_std_in_grpo and len(group_indices) > 1:
                group_scale = group_scores.std() + epsilon
            else:
                group_scale = torch.ones_like(group_mean)
            base_advantages = (group_scores - group_mean) / group_scale

            successful_local_indices = [
                local_index
                for local_index, sample_index in enumerate(group_indices)
                if bool(purchase_success[sample_index])
            ]
            if len(successful_local_indices) > 1 and alpha > 0.0:
                successful_batch_indices = [
                    group_indices[local_index]
                    for local_index in successful_local_indices
                ]
                successful_confidences = confidences[successful_batch_indices]
                confidence_baseline = successful_confidences.mean()
                for local_index, confidence in zip(
                    successful_local_indices,
                    successful_confidences,
                    strict=True,
                ):
                    base_advantages[local_index] += alpha * (
                        confidence_baseline - confidence
                    )

            sequence_advantages[group_indices] = base_advantages

    advantages = sequence_advantages.unsqueeze(-1) * response_mask
    return advantages, advantages.clone()
