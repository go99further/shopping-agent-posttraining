"""GraphGPO-lite credit assignment over exact actor-visible shopping states.

The graph is rebuilt independently for each on-policy prompt group.  Nodes are
SHA-256 hashes of exact public observations, edges are public tool decisions,
and only a valid Reward v3 ``gold_purchase`` terminal marks a goal.  No graph is
persisted across batches, and evaluation trajectories are never consumed.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import torch
from verl.trainer.ppo.core_algos import register_adv_est

from shopping_grpo.training.grpo.gigpo import compute_shopping_gigpo_advantage

SHOPPING_GRAPHGPO_ESTIMATOR = "shopping_graphgpo"
GRAPHGPO_SCHEMA = "shopping-graphgpo-lite-v1"


def _validate_aligned_turns(
    exact_anchors: Sequence[Sequence[str]],
    next_exact_anchors: Sequence[Sequence[str]],
    action_keys: Sequence[Sequence[str]],
    turn_token_spans: Sequence[Sequence[Sequence[int]]],
) -> None:
    for sample_index, fields in enumerate(
        zip(
            exact_anchors,
            next_exact_anchors,
            action_keys,
            turn_token_spans,
            strict=True,
        )
    ):
        if len({len(field) for field in fields}) != 1:
            raise ValueError(f"GraphGPO turn fields are not aligned at sample {sample_index}")
        if not fields[0]:
            raise ValueError(f"GraphGPO trajectory at sample {sample_index} has no tool turns")


def extract_graphgpo_signals(
    shopping_infos: Sequence[object],
) -> tuple[
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[float]],
    list[list[list[int]]],
    list[bool],
]:
    """Extract exact graph fields and strict terminal goals from AgentLoop output."""
    exact_anchors: list[list[str]] = []
    next_exact_anchors: list[list[str]] = []
    structured_anchors: list[list[str]] = []
    action_keys: list[list[str]] = []
    turn_rewards: list[list[float]] = []
    turn_token_spans: list[list[list[int]]] = []
    terminal_gold: list[bool] = []
    for sample_index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping):
            raise TypeError(
                f"shopping extra field at index {sample_index} is not an object"
            )
        gigpo = info.get("gigpo")
        if not isinstance(gigpo, Mapping) or gigpo.get("enabled") is not True:
            raise ValueError(
                f"shopping extra field at index {sample_index} is missing process diagnostics"
            )
        if gigpo.get("graph_schema") != GRAPHGPO_SCHEMA:
            raise ValueError(f"GraphGPO schema mismatch at index {sample_index}")
        if gigpo.get("instrumentation_error"):
            raise ValueError(
                f"GraphGPO instrumentation failed at index {sample_index}: "
                f"{gigpo['instrumentation_error']}"
            )
        before = [str(value) for value in gigpo.get("exact_anchors") or []]
        after = [str(value) for value in gigpo.get("next_exact_anchors") or []]
        structured = [str(value) for value in gigpo.get("anchors") or []]
        actions = [str(value) for value in gigpo.get("action_keys") or []]
        rewards: list[float] = []
        spans: list[list[int]] = []
        for turn_index, raw_reward in enumerate(gigpo.get("turn_rewards") or []):
            try:
                reward = float(raw_reward)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"GraphGPO turn reward at sample {sample_index}, turn "
                    f"{turn_index} is not numeric"
                ) from exc
            if not math.isfinite(reward):
                raise ValueError(
                    f"GraphGPO turn reward at sample {sample_index}, turn "
                    f"{turn_index} is not finite"
                )
            rewards.append(reward)
        for turn_index, raw_span in enumerate(gigpo.get("turn_token_spans") or []):
            try:
                start, end = int(raw_span[0]), int(raw_span[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"GraphGPO token span at sample {sample_index}, turn {turn_index} is invalid"
                ) from exc
            if start < 0 or end <= start:
                raise ValueError(
                    f"GraphGPO token span at sample {sample_index}, turn {turn_index} is invalid"
                )
            spans.append([start, end])
        if (
            len({len(before), len(after), len(structured), len(actions), len(rewards), len(spans)})
            != 1
        ):
            raise ValueError(f"GraphGPO fields at index {sample_index} are not aligned")
        if not before:
            raise ValueError(f"GraphGPO trajectory at index {sample_index} has no turns")
        exact_anchors.append(before)
        next_exact_anchors.append(after)
        structured_anchors.append(structured)
        action_keys.append(actions)
        turn_rewards.append(rewards)
        turn_token_spans.append(spans)
        terminal_gold.append(
            info.get("reward_version") == "shopsimulator-reward-v3"
            and info.get("reward_type") == "gold_purchase"
            and info.get("reward_valid") is True
            and info.get("done") is True
        )
    return (
        exact_anchors,
        next_exact_anchors,
        structured_anchors,
        action_keys,
        turn_rewards,
        turn_token_spans,
        terminal_gold,
    )


def reference_graph_micro_values(
    index: Sequence[Hashable],
    exact_anchors: Sequence[Sequence[str]],
    next_exact_anchors: Sequence[Sequence[str]],
    action_keys: Sequence[Sequence[str]],
    terminal_gold: Sequence[bool],
    *,
    min_contrast_size: int = 2,
    micro_advantage_clip: float = 2.0,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Straightforward correctness oracle for shortest-distance micro credit."""
    batch_size = len(index)
    if not (
        len(exact_anchors)
        == len(next_exact_anchors)
        == len(action_keys)
        == len(terminal_gold)
        == batch_size
    ):
        raise ValueError("all GraphGPO fields must match the batch size")
    if min_contrast_size < 2:
        raise ValueError("min_contrast_size must be at least 2")
    if micro_advantage_clip <= 0.0 or not math.isfinite(micro_advantage_clip):
        raise ValueError("micro_advantage_clip must be finite and positive")
    dummy_spans = [[[0, 1] for _ in values] for values in exact_anchors]
    _validate_aligned_turns(exact_anchors, next_exact_anchors, action_keys, dummy_spans)

    micro = [[0.0] * len(turns) for turns in exact_anchors]
    prompt_groups: dict[Hashable, list[int]] = {}
    for sample_index, uid in enumerate(index):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"group uid is not hashable: {uid!r}") from exc
        prompt_groups.setdefault(uid, []).append(sample_index)

    goal_groups = 0
    contrastive_states = 0
    labeled_occurrences = 0
    for uid, sample_indices in prompt_groups.items():
        goal_node = f"goal:{uid!r}"
        goals = set()
        edges: list[tuple[int, int, str, str, str]] = []
        for sample_index in sample_indices:
            last_turn = len(exact_anchors[sample_index]) - 1
            for turn_index, (source, destination, action) in enumerate(
                zip(
                    exact_anchors[sample_index],
                    next_exact_anchors[sample_index],
                    action_keys[sample_index],
                    strict=True,
                )
            ):
                if terminal_gold[sample_index] and turn_index == last_turn:
                    destination = goal_node
                    goals.add(goal_node)
                edges.append((sample_index, turn_index, str(source), str(destination), str(action)))
        if not goals:
            continue
        goal_groups += 1
        reverse: dict[str, set[str]] = {}
        for _, _, source, destination, _ in edges:
            reverse.setdefault(destination, set()).add(source)
        distances = {goal: 0 for goal in goals}
        queue = list(goals)
        while queue:
            destination = queue.pop(0)
            next_distance = distances[destination] + 1
            for source in reverse.get(destination, ()):
                if source not in distances:
                    distances[source] = next_distance
                    queue.append(source)

        by_state: dict[str, list[tuple[int, int, str, str]]] = {}
        seen: set[tuple[int, str]] = set()
        for sample_index, turn_index, source, destination, action in edges:
            sample_state = (sample_index, source)
            if sample_state in seen:
                continue
            seen.add(sample_state)
            by_state.setdefault(source, []).append((sample_index, turn_index, destination, action))
        for source, occurrences in by_state.items():
            if len(occurrences) < min_contrast_size:
                continue
            if len({occurrence[3] for occurrence in occurrences}) < 2:
                continue
            source_distance = distances.get(source)
            if source_distance is None:
                continue
            progress = []
            for _, _, destination, _ in occurrences:
                destination_distance = distances.get(destination)
                progress.append(
                    -1.0
                    if destination_distance is None
                    else float(source_distance - destination_distance)
                )
            mean_progress = sum(progress) / len(progress)
            centered = [
                min(max(value - mean_progress, -micro_advantage_clip), micro_advantage_clip)
                for value in progress
            ]
            if not any(centered):
                continue
            contrastive_states += 1
            labeled_occurrences += len(occurrences)
            for (sample_index, turn_index, _, _), value in zip(occurrences, centered, strict=True):
                micro[sample_index][turn_index] = value
    return micro, {
        "schema": GRAPHGPO_SCHEMA,
        "prompt_groups": len(prompt_groups),
        "goal_groups": goal_groups,
        "contrastive_states": contrastive_states,
        "labeled_occurrences": labeled_occurrences,
    }


def optimized_graph_micro_values(
    index: Sequence[Hashable],
    exact_anchors: Sequence[Sequence[str]],
    next_exact_anchors: Sequence[Sequence[str]],
    action_keys: Sequence[Sequence[str]],
    terminal_gold: Sequence[bool],
    *,
    min_contrast_size: int = 2,
    micro_advantage_clip: float = 2.0,
    collect_diagnostics: bool = True,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Allocation-conscious equivalent of :func:`reference_graph_micro_values`."""
    batch_size = len(index)
    if not (
        len(exact_anchors)
        == len(next_exact_anchors)
        == len(action_keys)
        == len(terminal_gold)
        == batch_size
    ):
        raise ValueError("all GraphGPO fields must match the batch size")
    if min_contrast_size < 2:
        raise ValueError("min_contrast_size must be at least 2")
    if micro_advantage_clip <= 0.0 or not math.isfinite(micro_advantage_clip):
        raise ValueError("micro_advantage_clip must be finite and positive")

    micro = [[0.0] * len(turns) for turns in exact_anchors]
    prompt_groups: dict[Hashable, list[int]] = defaultdict(list)
    for sample_index, uid in enumerate(index):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"group uid is not hashable: {uid!r}") from exc
        if not (
            len(exact_anchors[sample_index])
            == len(next_exact_anchors[sample_index])
            == len(action_keys[sample_index])
        ):
            raise ValueError(f"GraphGPO turn fields are not aligned at sample {sample_index}")
        if not exact_anchors[sample_index]:
            raise ValueError(f"GraphGPO trajectory at sample {sample_index} has no tool turns")
        prompt_groups[uid].append(sample_index)

    goal_groups = 0
    strict_goal_terminal_count = 0
    contrastive_states = 0
    eligible_states = 0
    labeled_occurrences = 0
    micro_nonzero_occurrences = 0
    group_diagnostics = []
    for group_number, (uid, sample_indices) in enumerate(prompt_groups.items()):
        group_goal_terminal_count = sum(
            bool(terminal_gold[sample_index]) for sample_index in sample_indices
        )
        strict_goal_terminal_count += group_goal_terminal_count
        public_hashes: set[str] = set()
        transition_count = 0
        if collect_diagnostics:
            for sample_index in sample_indices:
                public_hashes.update(exact_anchors[sample_index])
                public_hashes.update(next_exact_anchors[sample_index])
                transition_count += len(exact_anchors[sample_index])
        if not group_goal_terminal_count:
            if collect_diagnostics:
                group_diagnostics.append(
                    {
                        "uid": uid,
                        "trajectory_count": len(sample_indices),
                        "transition_count": transition_count,
                        "exact_public_hash_count": len(public_hashes),
                        "exact_public_hashes": tuple(sorted(public_hashes)),
                        "strict_goal_terminal_count": 0,
                        "goal_branch": False,
                        "bfs_reachable_public_hash_count": 0,
                        "bfs_max_goal_distance": 0,
                        "eligible_state_count": 0,
                        "distance_labeled_state_count": 0,
                        "distance_labeled_occurrences": 0,
                        "micro_nonzero_occurrences": 0,
                        "finite": True,
                        "invariant_ok": True,
                    }
                )
            continue
        goal_groups += 1
        goal_node = f"goal:{group_number}"
        reverse: dict[str, list[str]] = defaultdict(list)
        by_state: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
        for sample_index in sample_indices:
            sources = exact_anchors[sample_index]
            destinations = next_exact_anchors[sample_index]
            actions = action_keys[sample_index]
            last_turn = len(sources) - 1
            sample_is_goal = bool(terminal_gold[sample_index])
            seen_sources: set[str] = set()
            for turn_index in range(len(sources)):
                source = str(sources[turn_index])
                destination = (
                    goal_node
                    if sample_is_goal and turn_index == last_turn
                    else str(destinations[turn_index])
                )
                reverse[destination].append(source)
                if source not in seen_sources:
                    seen_sources.add(source)
                    by_state[source].append(
                        (sample_index, turn_index, destination, str(actions[turn_index]))
                    )

        distances = {goal_node: 0}
        queue = deque([goal_node])
        while queue:
            destination = queue.popleft()
            next_distance = distances[destination] + 1
            for source in reverse.get(destination, ()):
                if source not in distances:
                    distances[source] = next_distance
                    queue.append(source)

        group_eligible_states = 0
        group_labeled_states = 0
        group_labeled_occurrences = 0
        group_micro_nonzero_occurrences = 0
        group_finite = True
        for source, occurrences in by_state.items():
            if len(occurrences) < min_contrast_size:
                continue
            first_action = occurrences[0][3]
            if all(occurrence[3] == first_action for occurrence in occurrences[1:]):
                continue
            eligible_states += 1
            group_eligible_states += 1
            source_distance = distances.get(source)
            if source_distance is None:
                continue
            progress = [
                -1.0
                if (distance := distances.get(occurrence[2])) is None
                else float(source_distance - distance)
                for occurrence in occurrences
            ]
            mean_progress = sum(progress) / len(progress)
            centered = [
                min(max(value - mean_progress, -micro_advantage_clip), micro_advantage_clip)
                for value in progress
            ]
            if not all(math.isfinite(value) for value in centered):
                group_finite = False
                raise RuntimeError("GraphGPO produced a non-finite graph micro advantage")
            if not any(centered):
                continue
            contrastive_states += 1
            group_labeled_states += 1
            labeled_occurrences += len(occurrences)
            group_labeled_occurrences += len(occurrences)
            for occurrence, value in zip(occurrences, centered, strict=True):
                micro[occurrence[0]][occurrence[1]] = value
                if value != 0.0:
                    micro_nonzero_occurrences += 1
                    group_micro_nonzero_occurrences += 1
        if collect_diagnostics:
            reachable_public_distances = [
                distance for node, distance in distances.items() if node != goal_node
            ]
            invariant_ok = (
                group_goal_terminal_count > 0
                and all(distance >= 0 for distance in distances.values())
                and group_micro_nonzero_occurrences <= group_labeled_occurrences
                and group_labeled_states <= group_eligible_states
            )
            group_diagnostics.append(
                {
                    "uid": uid,
                    "trajectory_count": len(sample_indices),
                    "transition_count": transition_count,
                    "exact_public_hash_count": len(public_hashes),
                    "exact_public_hashes": tuple(sorted(public_hashes)),
                    "strict_goal_terminal_count": group_goal_terminal_count,
                    "goal_branch": True,
                    "bfs_reachable_public_hash_count": len(reachable_public_distances),
                    "bfs_max_goal_distance": max(reachable_public_distances, default=0),
                    "eligible_state_count": group_eligible_states,
                    "distance_labeled_state_count": group_labeled_states,
                    "distance_labeled_occurrences": group_labeled_occurrences,
                    "micro_nonzero_occurrences": group_micro_nonzero_occurrences,
                    "finite": group_finite,
                    "invariant_ok": invariant_ok,
                }
            )
    return micro, {
        "schema": GRAPHGPO_SCHEMA,
        "prompt_groups": len(prompt_groups),
        "goal_groups": goal_groups,
        "strict_goal_terminal_count": strict_goal_terminal_count,
        "contrastive_states": contrastive_states,
        "eligible_states": eligible_states,
        "labeled_occurrences": labeled_occurrences,
        "micro_nonzero_occurrences": micro_nonzero_occurrences,
        "finite": all(group["finite"] for group in group_diagnostics),
        "invariant_ok": all(group["invariant_ok"] for group in group_diagnostics),
        "groups": tuple(group_diagnostics),
    }


def _fallback_process_signal(
    sample_indices: Sequence[int],
    scores: Sequence[float],
    structured_anchors: Sequence[Sequence[str]],
    action_keys: Sequence[Sequence[str]],
    turn_rewards: Sequence[Sequence[float]],
    *,
    gamma: float,
    tolerance: float,
) -> bool:
    """Mirror GiGPO's return-centering gate without constructing token tensors."""
    occurrences: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for sample_index in sample_indices:
        rewards = turn_rewards[sample_index]
        running = float(scores[sample_index])
        returns = [0.0] * len(rewards)
        for turn_index in range(len(rewards) - 1, -1, -1):
            running = float(rewards[turn_index]) + gamma * running
            returns[turn_index] = running
        seen = set()
        for anchor, action, value in zip(
            structured_anchors[sample_index],
            action_keys[sample_index],
            returns,
            strict=True,
        ):
            anchor = str(anchor)
            if anchor in seen:
                continue
            seen.add(anchor)
            occurrences[anchor].append((str(action), value))
    for values in occurrences.values():
        if len(values) < 2 or len({value[0] for value in values}) < 2:
            continue
        returns = [value[1] for value in values]
        if max(returns) - min(returns) > tolerance:
            return True
    return False


def select_graphgpo_signal_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    shopping_infos: Sequence[object],
    *,
    fallback: str = "shopping_gigpo",
    potential_gamma: float = 1.0,
    min_contrast_size: int = 2,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Keep a group iff at least one *actual* advantage source is non-zero."""
    if len(uids) != len(seq_rewards) or len(shopping_infos) != len(uids):
        raise ValueError("uids, seq_rewards, and shopping_infos must have equal length")
    if fallback not in {"shopping_gigpo", "grpo"}:
        raise ValueError("fallback must be 'shopping_gigpo' or 'grpo'")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if not 0.0 < potential_gamma <= 1.0 or not math.isfinite(potential_gamma):
        raise ValueError("potential_gamma must be finite and in (0, 1]")
    (
        exact_anchors,
        next_exact_anchors,
        structured_anchors,
        action_keys,
        turn_rewards,
        _,
        terminal_gold,
    ) = extract_graphgpo_signals(shopping_infos)
    numeric_scores = [float(value) for value in seq_rewards]
    if any(not math.isfinite(value) for value in numeric_scores):
        raise ValueError("seq_rewards must be finite")
    invalid = [
        bool(info.get("infrastructure_invalid"))
        or bool(info.get("reward_unverifiable"))
        or not bool(info.get("reward_valid", True))
        for info in shopping_infos
    ]
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for sample_index, uid in enumerate(uids):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {sample_index} is not hashable") from exc
        grouped[uid].append(sample_index)

    micro, graph_diagnostics = optimized_graph_micro_values(
        uids,
        exact_anchors,
        next_exact_anchors,
        action_keys,
        terminal_gold,
        min_contrast_size=min_contrast_size,
    )
    graph_by_uid = {group["uid"]: group for group in graph_diagnostics["groups"]}
    kept_uids = []
    groups = []
    for uid, sample_indices in grouped.items():
        group_scores = [numeric_scores[index] for index in sample_indices]
        macro_signal = max(group_scores) - min(group_scores) > tolerance
        has_goal = any(terminal_gold[index] for index in sample_indices)
        graph_signal = has_goal and any(
            abs(value) > tolerance for index in sample_indices for value in micro[index]
        )
        fallback_signal = False
        if not has_goal and fallback == "shopping_gigpo":
            fallback_signal = _fallback_process_signal(
                sample_indices,
                numeric_scores,
                structured_anchors,
                action_keys,
                turn_rewards,
                gamma=potential_gamma,
                tolerance=tolerance,
            )
        process_signal = graph_signal or fallback_signal
        invalid_group = any(invalid[index] for index in sample_indices)
        keep = not invalid_group and (macro_signal or process_signal)
        finite = all(math.isfinite(value) for value in group_scores)
        invariant_ok = (
            finite
            and (not graph_signal or has_goal)
            and (not fallback_signal or not has_goal)
            and bool(graph_by_uid[uid]["invariant_ok"])
        )
        if not invariant_ok:
            raise RuntimeError(f"GraphGPO diagnostics invariant failed for uid={uid!r}")
        if keep:
            kept_uids.append(uid)
        active_sources = tuple(
            source
            for source, enabled in (
                ("macro", macro_signal),
                ("graph", graph_signal),
                ("fallback", fallback_signal),
            )
            if enabled
        )
        groups.append(
            {
                "uid": uid,
                "indices": tuple(sample_indices),
                "rewards": tuple(group_scores),
                "terminal_utilities": tuple(group_scores),
                "purchase_success": tuple(terminal_gold[index] for index in sample_indices),
                "sampling_invalid": invalid_group,
                "sampling_invalid_reasons": (("sampling_invalid",) if invalid_group else ()),
                "macro_signal": macro_signal,
                "process_signal": process_signal,
                "graph_signal": graph_signal,
                "fallback_process_signal": fallback_signal,
                "active_advantage_sources": active_sources,
                "terminal_reward_variance_zero": not macro_signal,
                "invalid": invalid_group,
                "confidence_only": False,
                "kept": keep,
                "zero_signal_gate": "keep" if keep else "skip",
                "finite": finite,
                "invariant_ok": invariant_ok,
                "graph": graph_by_uid[uid],
                "drop_reason": "sampling_invalid"
                if invalid_group
                else (None if keep else "all_advantage_sources_zero"),
            }
        )
    kept_uid_set = set(kept_uids)
    selected = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    return selected, {
        "num_groups": len(groups),
        "num_trajectories": len(uids),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(groups) - len(kept_uids),
        "kept_uids": tuple(kept_uids),
        "all_equal_group_count": sum(group["terminal_reward_variance_zero"] for group in groups),
        "all_zero_utility_group_count": sum(
            max(abs(numeric_scores[index]) for index in group["indices"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(terminal_gold[index] for index in group["indices"]) for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(terminal_gold[index] for index in group["indices"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(group["invalid"] for group in groups),
        "confidence_only_group_count": 0,
        "process_only_group_count": sum(
            group["kept"]
            and group["terminal_reward_variance_zero"]
            and (group["graph_signal"] or group["fallback_process_signal"])
            for group in groups
        ),
        "macro_source_group_count": sum(group["macro_signal"] for group in groups),
        "process_source_group_count": sum(group["process_signal"] for group in groups),
        "graph_source_group_count": sum(group["graph_signal"] for group in groups),
        "fallback_source_group_count": sum(
            group["fallback_process_signal"] for group in groups
        ),
        "zero_signal_gate_keep_count": sum(group["kept"] for group in groups),
        "zero_signal_gate_skip_count": sum(not group["kept"] for group in groups),
        "finite": bool(graph_diagnostics["finite"])
        and all(group["finite"] for group in groups),
        "invariant_ok": bool(graph_diagnostics["invariant_ok"])
        and all(group["invariant_ok"] for group in groups),
        "graph": graph_diagnostics,
        "groups": tuple(groups),
    }


def serialize_graphgpo_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    accepted_uids: Sequence[Hashable] = (),
) -> dict[str, Any]:
    """Return a stable JSON-safe audit payload without observations or hidden gold."""
    accepted = set(accepted_uids)
    serialized_groups = []
    for group in diagnostics["groups"]:
        graph = group["graph"]
        serialized_groups.append(
            {
                "uid": str(group["uid"]),
                "indices": list(group["indices"]),
                "rewards": list(group["rewards"]),
                "terminal_utilities": list(group["terminal_utilities"]),
                "purchase_success": list(group["purchase_success"]),
                "sampling_invalid": bool(group["sampling_invalid"]),
                "sampling_invalid_reasons": list(group["sampling_invalid_reasons"]),
                "macro_signal": bool(group["macro_signal"]),
                "process_signal": bool(group["process_signal"]),
                "graph_signal": bool(group["graph_signal"]),
                "fallback_process_signal": bool(group["fallback_process_signal"]),
                "active_advantage_sources": list(group["active_advantage_sources"]),
                "terminal_reward_variance_zero": bool(
                    group["terminal_reward_variance_zero"]
                ),
                "kept": bool(group["kept"]),
                "accepted": group["uid"] in accepted,
                "zero_signal_gate": str(group["zero_signal_gate"]),
                "drop_reason": group["drop_reason"],
                "finite": bool(group["finite"]),
                "invariant_ok": bool(group["invariant_ok"]),
                "graph": {
                    "trajectory_count": int(graph["trajectory_count"]),
                    "transition_count": int(graph["transition_count"]),
                    "exact_public_hash_count": int(graph["exact_public_hash_count"]),
                    "exact_public_hashes": list(graph["exact_public_hashes"]),
                    "strict_goal_terminal_count": int(
                        graph["strict_goal_terminal_count"]
                    ),
                    "goal_branch": bool(graph["goal_branch"]),
                    "bfs_reachable_public_hash_count": int(
                        graph["bfs_reachable_public_hash_count"]
                    ),
                    "bfs_max_goal_distance": int(graph["bfs_max_goal_distance"]),
                    "eligible_state_count": int(graph["eligible_state_count"]),
                    "distance_labeled_state_count": int(
                        graph["distance_labeled_state_count"]
                    ),
                    "distance_labeled_occurrences": int(
                        graph["distance_labeled_occurrences"]
                    ),
                    "micro_nonzero_occurrences": int(
                        graph["micro_nonzero_occurrences"]
                    ),
                    "finite": bool(graph["finite"]),
                    "invariant_ok": bool(graph["invariant_ok"]),
                },
            }
        )
    graph = diagnostics["graph"]
    return {
        "schema": str(graph["schema"]),
        "num_groups": int(diagnostics["num_groups"]),
        "num_trajectories": int(diagnostics["num_trajectories"]),
        "macro_source_group_count": int(diagnostics["macro_source_group_count"]),
        "process_source_group_count": int(diagnostics["process_source_group_count"]),
        "graph_source_group_count": int(diagnostics["graph_source_group_count"]),
        "fallback_source_group_count": int(diagnostics["fallback_source_group_count"]),
        "zero_signal_gate_keep_count": int(diagnostics["zero_signal_gate_keep_count"]),
        "zero_signal_gate_skip_count": int(diagnostics["zero_signal_gate_skip_count"]),
        "strict_goal_terminal_count": int(graph["strict_goal_terminal_count"]),
        "eligible_state_count": int(graph["eligible_states"]),
        "distance_labeled_occurrences": int(graph["labeled_occurrences"]),
        "micro_nonzero_occurrences": int(graph["micro_nonzero_occurrences"]),
        "finite": bool(diagnostics["finite"]),
        "invariant_ok": bool(diagnostics["invariant_ok"]),
        "groups": serialized_groups,
    }


def graphgpo_diagnostic_metrics(diagnostics: Mapping[str, Any]) -> dict[str, float]:
    """Collapse GraphGPO evidence into scalar per-step trainer metrics."""
    graph = diagnostics["graph"]
    return {
        "graphgpo/strict_goal_terminals": float(graph["strict_goal_terminal_count"]),
        "graphgpo/goal_groups": float(graph["goal_groups"]),
        "graphgpo/eligible_states": float(graph["eligible_states"]),
        "graphgpo/distance_labeled_occurrences": float(graph["labeled_occurrences"]),
        "graphgpo/micro_nonzero_occurrences": float(graph["micro_nonzero_occurrences"]),
        "graphgpo/macro_source_groups": float(diagnostics["macro_source_group_count"]),
        "graphgpo/process_source_groups": float(diagnostics["process_source_group_count"]),
        "graphgpo/graph_source_groups": float(diagnostics["graph_source_group_count"]),
        "graphgpo/fallback_source_groups": float(diagnostics["fallback_source_group_count"]),
        "graphgpo/zero_signal_gate_keep": float(diagnostics["zero_signal_gate_keep_count"]),
        "graphgpo/zero_signal_gate_skip": float(diagnostics["zero_signal_gate_skip_count"]),
        "graphgpo/finite": float(bool(diagnostics["finite"])),
        "graphgpo/invariant_ok": float(bool(diagnostics["invariant_ok"])),
    }


@register_adv_est(SHOPPING_GRAPHGPO_ESTIMATOR)
def compute_shopping_graphgpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Hashable],
    exact_anchors: Sequence[Sequence[str]],
    next_exact_anchors: Sequence[Sequence[str]],
    structured_anchors: Sequence[Sequence[str]],
    action_keys: Sequence[Sequence[str]],
    turn_rewards: Sequence[Sequence[float]],
    turn_token_spans: Sequence[Sequence[Sequence[int]]],
    terminal_gold: Sequence[bool],
    *,
    micro_advantage_weight: float = 0.20,
    min_contrast_size: int = 2,
    micro_advantage_clip: float = 2.0,
    fallback: str = "shopping_gigpo",
    potential_gamma: float = 1.0,
    potential_reward_clip: float = 0.25,
    epsilon: float = 1.0e-6,
    norm_adv_by_std_in_grpo: bool = False,
    config: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine GRPO macro credit with graph distance progress on action spans."""
    if config is not None:
        graph_config = config.get("shopping_graphgpo", {})
        if not bool(graph_config.get("enabled", False)):
            raise ValueError("shopping_graphgpo is not enabled")
        micro_advantage_weight = float(
            graph_config.get("micro_advantage_weight", micro_advantage_weight)
        )
        min_contrast_size = int(graph_config.get("min_contrast_size", min_contrast_size))
        micro_advantage_clip = float(graph_config.get("micro_advantage_clip", micro_advantage_clip))
        fallback = str(graph_config.get("fallback", fallback))
        potential_gamma = float(graph_config.get("potential_gamma", potential_gamma))
        potential_reward_clip = float(
            graph_config.get("potential_reward_clip", potential_reward_clip)
        )
        norm_adv_by_std_in_grpo = bool(
            config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
        )
    if token_level_rewards.ndim != 2 or response_mask.shape != token_level_rewards.shape:
        raise ValueError("token_level_rewards and response_mask must be equal rank-2 tensors")
    if micro_advantage_weight <= 0.0 or not math.isfinite(micro_advantage_weight):
        raise ValueError("micro_advantage_weight must be finite and positive")
    if fallback not in {"shopping_gigpo", "grpo"}:
        raise ValueError("fallback must be 'shopping_gigpo' or 'grpo'")
    batch_size, response_length = token_level_rewards.shape
    aligned = (
        index,
        exact_anchors,
        next_exact_anchors,
        structured_anchors,
        action_keys,
        turn_rewards,
        turn_token_spans,
        terminal_gold,
    )
    if any(len(values) != batch_size for values in aligned):
        raise ValueError("all GraphGPO trajectory fields must match the batch size")
    _validate_aligned_turns(exact_anchors, next_exact_anchors, action_keys, turn_token_spans)

    scores = token_level_rewards.sum(dim=-1)
    prompt_groups: dict[Hashable, list[int]] = defaultdict(list)
    for sample_index, uid in enumerate(index):
        prompt_groups[uid].append(sample_index)
    sequence_advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for sample_indices in prompt_groups.values():
            group_scores = scores[sample_indices]
            centered = group_scores - group_scores.mean()
            if norm_adv_by_std_in_grpo and len(sample_indices) > 1:
                centered = centered / (group_scores.std() + epsilon)
            sequence_advantages[sample_indices] = centered

    if fallback == "shopping_gigpo":
        fallback_advantages, _ = compute_shopping_gigpo_advantage(
            token_level_rewards,
            response_mask,
            index,
            structured_anchors,
            action_keys,
            turn_rewards,
            turn_token_spans,
            step_advantage_weight=micro_advantage_weight,
            potential_gamma=potential_gamma,
            potential_reward_clip=potential_reward_clip,
            min_anchor_group_size=min_contrast_size,
            micro_advantage_clip=micro_advantage_clip,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    else:
        fallback_advantages = sequence_advantages.unsqueeze(-1) * response_mask

    micro_values, _ = optimized_graph_micro_values(
        index,
        exact_anchors,
        next_exact_anchors,
        action_keys,
        terminal_gold,
        min_contrast_size=min_contrast_size,
        micro_advantage_clip=micro_advantage_clip,
        collect_diagnostics=False,
    )
    graph_groups = {
        uid
        for uid, sample_indices in prompt_groups.items()
        if any(terminal_gold[index] for index in sample_indices)
    }
    graph_advantages = sequence_advantages.unsqueeze(-1).expand(-1, response_length).clone()
    with torch.no_grad():
        differences = torch.zeros(
            (batch_size, response_length + 1),
            dtype=scores.dtype,
            device=scores.device,
        )
        for sample_index, per_turn in enumerate(micro_values):
            for turn_index, value in enumerate(per_turn):
                if value == 0.0:
                    continue
                start, end = turn_token_spans[sample_index][turn_index]
                start, end = int(start), int(end)
                if start < 0 or end <= start or end > response_length:
                    raise ValueError(
                        f"GraphGPO token span {(start, end)} is outside response "
                        f"length {response_length}"
                    )
                differences[sample_index, start] += float(value)
                differences[sample_index, end] -= float(value)
        graph_advantages.add_(micro_advantage_weight * differences[:, :-1].cumsum(dim=1)).mul_(
            response_mask
        )
    advantages = fallback_advantages.clone()
    for sample_index, uid in enumerate(index):
        if uid in graph_groups:
            advantages[sample_index] = graph_advantages[sample_index]
    return advantages, advantages.clone()
