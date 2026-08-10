"""Streaming paired diagnostics for two raw Shopping Agent trajectory runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path

try:
    import orjson
except ImportError:  # Standard-library fallback keeps the tool portable.
    orjson = None


SCHEMA_VERSION = "shopping-paired-trajectory-comparison-v1"
REWARD_VERSION = "shopsimulator-reward-v3"


def _terminal_detail(record: Mapping) -> Mapping:
    terminal = record.get("terminal_result")
    if not isinstance(terminal, Mapping):
        return {}
    detail = terminal.get("reward_detail")
    return detail if isinstance(detail, Mapping) else {}


def _strict_success(record: Mapping, detail: Mapping) -> bool:
    return bool(
        record.get("status") == "done"
        and record.get("done") is True
        and detail.get("reward_version") == REWARD_VERSION
        and detail.get("reward_type") == "gold_purchase"
        and detail.get("reward_valid") is True
        and detail.get("purchase_success") is True
        and detail.get("termination_reason") == "gold_purchase"
    )


def _instruction(record: Mapping) -> str:
    initial = record.get("initial_result")
    if isinstance(initial, Mapping) and initial.get("instruction"):
        return str(initial["instruction"])
    for message in record.get("messages") or ():
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _action_key(step: Mapping) -> str:
    name = str(step.get("tool_name") or "unknown")
    parameters = step.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    return name + ":" + json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize_record(record: Mapping) -> dict:
    detail = _terminal_detail(record)
    terminal = record.get("terminal_result")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    steps = [step for step in record.get("steps") or () if isinstance(step, Mapping)]
    tool_counts = Counter(str(step.get("tool_name") or "unknown") for step in steps)
    action_counts = Counter(_action_key(step) for step in steps)
    context_tokens = [
        int(item.get("input_tokens", 0))
        for item in record.get("context_turn_tokens") or ()
        if isinstance(item, Mapping)
    ]
    error = str(record.get("error") or "")
    purchase = terminal.get("purchase")
    purchase = purchase if isinstance(purchase, Mapping) else {}
    return {
        "task_id": int(record["task_id"]),
        "instruction": _instruction(record),
        "status": str(record.get("status") or "unknown"),
        "strict_success": _strict_success(record, detail),
        "reward_type": str(detail.get("reward_type") or "unknown"),
        "reward_valid": detail.get("reward_valid") is True,
        "final_reward": float(record.get("final_reward") or 0.0),
        "steps": len(steps),
        "tool_counts": dict(sorted(tool_counts.items())),
        "action_sequence": [
            {
                "tool": str(step.get("tool_name") or "unknown"),
                "parameters": (
                    dict(step["parameters"])
                    if isinstance(step.get("parameters"), Mapping)
                    else {}
                ),
            }
            for step in steps
        ],
        "duplicate_action_count": sum(count - 1 for count in action_counts.values()),
        "blocked_tool_calls": len(record.get("blocked_tool_calls") or ()),
        "context_compactions": len(record.get("context_compactions") or ()),
        "max_context_input_tokens": max(context_tokens, default=0),
        "context_error": "context" in error.lower(),
        "error": error,
        "purchase": {
            "asin": purchase.get("asin"),
            "options": dict(purchase.get("options") or {}),
            "price": purchase.get("price"),
        },
    }


def load_summaries(path: Path) -> dict[int, dict]:
    summaries: dict[int, dict] = {}
    loads = orjson.loads if orjson is not None else json.loads
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = loads(line)
            summary = summarize_record(record)
            task_id = summary["task_id"]
            if task_id in summaries:
                raise ValueError(f"{path} contains duplicate task_id {task_id}")
            summaries[task_id] = summary
    return summaries


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_runs(
    source: Mapping[int, Mapping],
    target: Mapping[int, Mapping],
    *,
    source_label: str,
    target_label: str,
) -> dict:
    source_ids = set(source)
    target_ids = set(target)
    if source_ids != target_ids:
        raise ValueError(
            "paired runs must contain identical task IDs: "
            f"source_only={sorted(source_ids - target_ids)} "
            f"target_only={sorted(target_ids - source_ids)}"
        )

    strict_transitions = Counter()
    reward_transitions = Counter()
    changed_cases = []
    step_deltas = []
    reward_deltas = []
    duplicate_deltas = []
    blocked_deltas = []
    for task_id in sorted(source_ids):
        left = source[task_id]
        right = target[task_id]
        transition = (
            f"{'success' if left['strict_success'] else 'failure'}_to_"
            f"{'success' if right['strict_success'] else 'failure'}"
        )
        strict_transitions[transition] += 1
        reward_transition = f"{left['reward_type']} -> {right['reward_type']}"
        reward_transitions[reward_transition] += 1
        step_delta = int(right["steps"]) - int(left["steps"])
        reward_delta = float(right["final_reward"]) - float(left["final_reward"])
        duplicate_delta = int(right["duplicate_action_count"]) - int(
            left["duplicate_action_count"]
        )
        blocked_delta = int(right["blocked_tool_calls"]) - int(
            left["blocked_tool_calls"]
        )
        step_deltas.append(step_delta)
        reward_deltas.append(reward_delta)
        duplicate_deltas.append(duplicate_delta)
        blocked_deltas.append(blocked_delta)
        if transition not in {"success_to_success", "failure_to_failure"}:
            changed_cases.append(
                {
                    "task_id": task_id,
                    "transition": transition,
                    "reward_type_transition": reward_transition,
                    "instruction": right["instruction"] or left["instruction"],
                    "source": dict(left),
                    "target": dict(right),
                }
            )

    gained = [
        case["task_id"]
        for case in changed_cases
        if case["transition"] == "failure_to_success"
    ]
    regressed = [
        case["task_id"]
        for case in changed_cases
        if case["transition"] == "success_to_failure"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source_label,
        "target": target_label,
        "paired_tasks": len(source_ids),
        "strict_success_transitions": dict(sorted(strict_transitions.items())),
        "reward_type_transitions": dict(sorted(reward_transitions.items())),
        "gained_strict_success_task_ids": gained,
        "regressed_strict_success_task_ids": regressed,
        "net_strict_success_delta": len(gained) - len(regressed),
        "mean_target_minus_source": {
            "steps": _mean(step_deltas),
            "final_reward": _mean(reward_deltas),
            "duplicate_actions": _mean(duplicate_deltas),
            "blocked_tool_calls": _mean(blocked_deltas),
        },
        "changed_cases": changed_cases,
    }


def compare_files(
    source_path: Path,
    target_path: Path,
    *,
    source_label: str,
    target_label: str,
) -> dict:
    return compare_runs(
        load_summaries(source_path),
        load_summaries(target_path),
        source_label=source_label,
        target_label=target_label,
    )
