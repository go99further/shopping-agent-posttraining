"""GiGPO-lite advantages for public-state shopping-agent trajectories.

The terminal Environment v2.1 / Reward v3 utility remains the macro objective.
The micro objective only compares decisions made from the same public anchor and
maps the centered return-to-go back to the assistant tokens that produced that
decision.  Hidden ShopSimulator goals are never consumed or retained.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
import math
from typing import Any

import torch
from verl.trainer.ppo.core_algos import register_adv_est

from shopping_grpo.training.grpo.process_verifier import (
    PROCESS_VERIFIER_SCHEMA,
    canonical_action_key,
    canonical_public_anchor,
    potential_delta_reward,
    potential_turn_reward,
    public_feature_potential,
    public_state_features,
    public_state_potential,
)


SHOPPING_GIGPO_ESTIMATOR = "shopping_gigpo"


def latest_assistant_token_span(response_mask: Sequence[object]) -> tuple[int, int]:
    """Locate the latest contiguous assistant-token block in a veRL response mask."""
    end = len(response_mask)
    while end and int(response_mask[end - 1]) == 0:
        end -= 1
    start = end
    while start and int(response_mask[start - 1]) == 1:
        start -= 1
    if start == end:
        raise ValueError("response mask has no assistant token span for the current tool call")
    return start, end


def shift_turn_spans(turns: list[dict[str, object]], removed_tokens: int) -> None:
    """Shift/drop recorded spans after context compaction, in place."""
    removed_tokens = int(removed_tokens)
    if removed_tokens <= 0:
        return
    write_index = 0
    for turn in turns:
        start, end = (int(value) for value in turn["token_span"])
        if end <= removed_tokens:
            continue
        turn["token_span"] = [max(start - removed_tokens, 0), end - removed_tokens]
        turns[write_index] = turn
        write_index += 1
    del turns[write_index:]


def extract_gigpo_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[list[str]], list[list[str]], list[list[float]], list[list[list[int]]]]:
    """Validate and extract aligned GiGPO turn fields from AgentLoop diagnostics."""
    all_anchors: list[list[str]] = []
    all_actions: list[list[str]] = []
    all_rewards: list[list[float]] = []
    all_spans: list[list[list[int]]] = []
    for sample_index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("gigpo"), Mapping):
            raise ValueError(
                f"shopping extra field at index {sample_index} is missing gigpo diagnostics"
            )
        gigpo = info["gigpo"]
        if gigpo.get("enabled") is not True:
            raise ValueError(
                f"gigpo process verifier is disabled at index {sample_index}"
            )
        if gigpo.get("verifier_schema") != PROCESS_VERIFIER_SCHEMA:
            raise ValueError(
                f"gigpo verifier schema mismatch at index {sample_index}"
            )
        if gigpo.get("instrumentation_error"):
            raise ValueError(
                f"gigpo instrumentation failed at index {sample_index}: "
                f"{gigpo['instrumentation_error']}"
            )
        anchors = list(gigpo.get("anchors") or [])
        actions = list(gigpo.get("action_keys") or [])
        rewards = list(gigpo.get("turn_rewards") or [])
        spans = list(gigpo.get("turn_token_spans") or [])
        checks = list(gigpo.get("verifier_checks") or [])
        failures = list(gigpo.get("verifier_failures") or [])
        if not (
            len(anchors)
            == len(actions)
            == len(rewards)
            == len(spans)
            == len(checks)
            == len(failures)
        ):
            raise ValueError(f"gigpo turn fields at index {sample_index} are not aligned")
        if not anchors:
            raise ValueError(f"gigpo trajectory at index {sample_index} has no tool turns")
        checked_spans: list[list[int]] = []
        checked_rewards: list[float] = []
        for turn_index, (reward, span, turn_checks, turn_failures) in enumerate(
            zip(rewards, spans, checks, failures, strict=True)
        ):
            if not isinstance(turn_checks, Mapping) or not isinstance(
                turn_failures, Mapping
            ):
                raise ValueError(
                    f"invalid gigpo verifier turn at sample {sample_index}, "
                    f"turn {turn_index}"
                )
            try:
                numeric_reward = float(reward)
                start, end = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"invalid gigpo turn at sample {sample_index}, turn {turn_index}"
                ) from exc
            if not math.isfinite(numeric_reward) or start < 0 or end <= start:
                raise ValueError(
                    f"invalid gigpo turn at sample {sample_index}, turn {turn_index}"
                )
            checked_rewards.append(numeric_reward)
            checked_spans.append([start, end])
        all_anchors.append([str(value) for value in anchors])
        all_actions.append([str(value) for value in actions])
        all_rewards.append(checked_rewards)
        all_spans.append(checked_spans)
    return all_anchors, all_actions, all_rewards, all_spans


@register_adv_est(SHOPPING_GIGPO_ESTIMATOR)
def compute_shopping_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Hashable],
    anchors: Sequence[Sequence[str]],
    action_keys: Sequence[Sequence[str]],
    turn_rewards: Sequence[Sequence[float]],
    turn_token_spans: Sequence[Sequence[Sequence[int]]],
    *,
    step_advantage_weight: float = 0.15,
    potential_gamma: float = 1.0,
    potential_reward_clip: float = 0.25,
    min_anchor_group_size: int = 2,
    micro_advantage_clip: float = 2.0,
    epsilon: float = 1.0e-6,
    norm_adv_by_std_in_grpo: bool = False,
    config: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine prompt-level GRPO macro advantage with public-anchor micro advantage."""
    if config is not None:
        gigpo_config = config.get("shopping_gigpo", {})
        if not bool(gigpo_config.get("process_verifier_enabled", False)):
            raise ValueError("shopping_gigpo process verifier is not enabled")
        step_advantage_weight = float(
            gigpo_config.get("step_advantage_weight", step_advantage_weight)
        )
        potential_gamma = float(gigpo_config.get("potential_gamma", potential_gamma))
        potential_reward_clip = float(
            gigpo_config.get("potential_reward_clip", potential_reward_clip)
        )
        min_anchor_group_size = int(
            gigpo_config.get("min_anchor_group_size", min_anchor_group_size)
        )
        micro_advantage_clip = float(
            gigpo_config.get("micro_advantage_clip", micro_advantage_clip)
        )
        norm_adv_by_std_in_grpo = bool(
            config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        )
        online_judge_weight = float(gigpo_config.get("online_judge_weight", 0.0))
        if not math.isfinite(online_judge_weight) or online_judge_weight != 0.0:
            raise ValueError("trajectory Judge must remain evaluation-only")

    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError("token_level_rewards and response_mask must be equal rank-2 tensors")
    batch_size, response_length = token_level_rewards.shape
    aligned = (index, anchors, action_keys, turn_rewards, turn_token_spans)
    if any(len(values) != batch_size for values in aligned):
        raise ValueError("all GiGPO trajectory fields must match the batch size")
    if step_advantage_weight < 0.0 or not math.isfinite(step_advantage_weight):
        raise ValueError("step_advantage_weight must be finite and non-negative")
    if not 0.0 < potential_gamma <= 1.0 or not math.isfinite(potential_gamma):
        raise ValueError("potential_gamma must be finite and in (0, 1]")
    if potential_reward_clip < 0.0 or not math.isfinite(potential_reward_clip):
        raise ValueError("potential_reward_clip must be finite and non-negative")
    if min_anchor_group_size < 2:
        raise ValueError("min_anchor_group_size must be at least 2")
    if micro_advantage_clip <= 0.0 or not math.isfinite(micro_advantage_clip):
        raise ValueError("micro_advantage_clip must be finite and positive")
    if any(
        abs(float(reward)) > potential_reward_clip + epsilon
        for rewards in turn_rewards
        for reward in rewards
    ):
        raise ValueError("turn reward exceeds configured potential_reward_clip")

    scores = token_level_rewards.sum(dim=-1)
    prompt_groups: dict[Hashable, list[int]] = defaultdict(list)
    for sample_index, uid in enumerate(index):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"group uid is not hashable: {uid!r}") from exc
        prompt_groups[uid].append(sample_index)

    sequence_advantages = torch.zeros_like(scores)
    micro_token_advantages = torch.zeros_like(token_level_rewards)
    anchor_occurrences: dict[
        tuple[Hashable, str], list[tuple[int, int, str]]
    ] = defaultdict(list)

    with torch.no_grad():
        for group_indices in prompt_groups.values():
            group_scores = scores[group_indices]
            group_mean = group_scores.mean()
            if norm_adv_by_std_in_grpo and len(group_indices) > 1:
                group_scale = group_scores.std() + epsilon
            else:
                group_scale = torch.ones_like(group_mean)
            sequence_advantages[group_indices] = (group_scores - group_mean) / group_scale

        max_turns = max((len(values) for values in turn_rewards), default=0)
        padded_turn_rewards = torch.tensor(
            [
                [*map(float, rewards), *([0.0] * (max_turns - len(rewards)))]
                for rewards in turn_rewards
            ],
            device=scores.device,
            dtype=scores.dtype,
        )
        turn_counts = torch.tensor(
            [len(rewards) for rewards in turn_rewards],
            device=scores.device,
        )
        valid_turns = (
            torch.arange(max_turns, device=scores.device).unsqueeze(0)
            < turn_counts.unsqueeze(1)
        )
        if potential_gamma == 1.0:
            turn_returns = torch.flip(
                torch.cumsum(torch.flip(padded_turn_rewards, dims=(1,)), dim=1),
                dims=(1,),
            )
            turn_returns.add_(scores.unsqueeze(1)).mul_(valid_turns)
        else:
            turn_returns = torch.zeros_like(padded_turn_rewards)
            running_returns = scores
            for turn_index in range(max_turns - 1, -1, -1):
                valid = valid_turns[:, turn_index]
                candidate = (
                    padded_turn_rewards[:, turn_index]
                    + potential_gamma * running_returns
                )
                running_returns = torch.where(valid, candidate, running_returns)
                turn_returns[:, turn_index] = torch.where(
                    valid,
                    running_returns,
                    torch.zeros_like(running_returns),
                )

        for sample_index, uid in enumerate(index):
            sample_fields = (
                anchors[sample_index],
                action_keys[sample_index],
                turn_rewards[sample_index],
                turn_token_spans[sample_index],
            )
            if len({len(values) for values in sample_fields}) != 1:
                raise ValueError(f"GiGPO turn fields are not aligned at sample {sample_index}")
            seen_anchors: set[str] = set()
            for turn_index, (anchor, action) in enumerate(
                zip(anchors[sample_index], action_keys[sample_index], strict=True)
            ):
                anchor = str(anchor)
                if anchor in seen_anchors:
                    continue
                seen_anchors.add(anchor)
                anchor_occurrences[(uid, anchor)].append(
                    (sample_index, turn_index, str(action))
                )

        eligible_occurrences: list[tuple[int, int, int, int, int]] = []
        group_sizes: list[int] = []
        for occurrences in anchor_occurrences.values():
            if len(occurrences) < min_anchor_group_size:
                continue
            if len({occurrence[2] for occurrence in occurrences}) < 2:
                continue
            group_index = len(group_sizes)
            group_sizes.append(len(occurrences))
            for sample_index, turn_index, _ in occurrences:
                span = turn_token_spans[sample_index][turn_index]
                start, end = int(span[0]), int(span[1])
                if start < 0 or end <= start or end > response_length:
                    raise ValueError(
                        f"GiGPO token span {(start, end)} is outside response "
                        f"length {response_length}"
                    )
                eligible_occurrences.append(
                    (sample_index, turn_index, start, end, group_index)
                )

        if eligible_occurrences:
            occurrence_tensor = torch.tensor(
                eligible_occurrences,
                device=scores.device,
            )
            sample_indices = occurrence_tensor[:, 0]
            turn_indices = occurrence_tensor[:, 1]
            starts = occurrence_tensor[:, 2]
            ends = occurrence_tensor[:, 3]
            group_indices = occurrence_tensor[:, 4]
            values = turn_returns[sample_indices, turn_indices]
            group_sums = torch.zeros(
                len(group_sizes),
                device=scores.device,
                dtype=scores.dtype,
            )
            group_sums.scatter_add_(0, group_indices, values)
            group_counts = torch.tensor(
                group_sizes,
                device=scores.device,
                dtype=scores.dtype,
            )
            centered = values - group_sums[group_indices] / group_counts[group_indices]
            centered.clamp_(min=-micro_advantage_clip, max=micro_advantage_clip)

            micro_differences = torch.zeros(
                (batch_size, response_length + 1),
                device=scores.device,
                dtype=scores.dtype,
            )
            micro_differences.index_put_(
                (sample_indices, starts),
                centered,
                accumulate=True,
            )
            micro_differences.index_put_(
                (sample_indices, ends),
                -centered,
                accumulate=True,
            )
            micro_token_advantages = micro_differences[:, :-1].cumsum(dim=1)

    # AgentLoop constructs spans from response_mask and compaction shifts them in lockstep.
    # Multiplying once here also guarantees that no tool token receives learning signal,
    # without introducing a GPU-to-CPU synchronization in every optimizer step.
    advantages = (
        sequence_advantages.unsqueeze(-1) + step_advantage_weight * micro_token_advantages
    ) * response_mask
    return advantages, advantages.clone()
