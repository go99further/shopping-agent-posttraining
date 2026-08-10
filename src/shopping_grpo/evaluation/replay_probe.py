"""Offline next-action replay probes for saved successful trajectories.

The probe never contacts ShopSimulator.  It replays actor-visible message
prefixes and asks a model for the next tool call, which makes it safe to run
beside online GRPO while still measuring policy/tool-use regressions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean

from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.tools import tool_call_to_action


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    task_id: int
    trajectory_id: str
    message_index: int
    messages: list[dict]
    reference_tool_call: dict
    latest_observation: str


def _plain_tool_call(value: Mapping) -> dict:
    function = value.get("function") or {}
    return {
        "id": str(value.get("id") or "replay-tool-call"),
        "type": "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": function.get("arguments") or "{}",
        },
    }


def sanitize_message(value: Mapping) -> dict:
    """Keep only OpenAI-compatible, actor-visible message fields."""

    role = str(value.get("role") or "")
    message = {"role": role, "content": value.get("content") or ""}
    if role == "assistant" and value.get("tool_calls"):
        message["tool_calls"] = [
            _plain_tool_call(tool_call)
            for tool_call in value["tool_calls"]
            if isinstance(tool_call, Mapping)
        ]
    if role == "tool":
        message["tool_call_id"] = str(value.get("tool_call_id") or "")
        if value.get("name"):
            message["name"] = str(value["name"])
    return message


def _arguments(tool_call: Mapping) -> dict | None:
    function = tool_call.get("function") or {}
    value = function.get("arguments") or "{}"
    try:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_name(tool_call: Mapping) -> str:
    return str((tool_call.get("function") or {}).get("name") or "")


def _latest_observation(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return str(message.get("content") or "")
    return ""


def iter_replay_cases(
    trajectories: Iterable[Mapping],
    *,
    max_prompt_chars: int = 16_000,
) -> Iterable[ReplayCase]:
    """Yield deterministic next-action cases from strict-success trajectories."""

    for row in trajectories:
        if row.get("status") != "done" or float(row.get("final_reward") or 0.0) != 1.0:
            continue
        raw_messages = row.get("messages") or []
        if not isinstance(raw_messages, list):
            continue
        messages = [
            sanitize_message(message)
            for message in raw_messages
            if isinstance(message, Mapping)
        ]
        trajectory_id = str(row.get("trajectory_id") or "")
        task_id = int(row["task_id"])
        prefix_messages = []
        prompt_chars = 0
        for index, message in enumerate(messages):
            tool_calls = message.get("tool_calls") or []
            if message.get("role") == "assistant" and len(tool_calls) == 1:
                reference = _plain_tool_call(tool_calls[0])
                if (
                    prompt_chars <= max_prompt_chars
                    and _tool_name(reference)
                    and _arguments(reference) is not None
                ):
                    prefix = list(prefix_messages)
                    yield ReplayCase(
                        case_id=f"{trajectory_id}:m{index}",
                        task_id=task_id,
                        trajectory_id=trajectory_id,
                        message_index=index,
                        messages=prefix,
                        reference_tool_call=reference,
                        latest_observation=_latest_observation(prefix),
                    )
            prefix_messages.append(message)
            prompt_chars += len(str(message.get("content") or ""))


def balanced_sample(cases: Iterable[ReplayCase], limit: int | None) -> list[ReplayCase]:
    """Round-robin reference tools so frequent searches do not dominate."""

    grouped: dict[str, deque[ReplayCase]] = defaultdict(deque)
    for case in cases:
        grouped[_tool_name(case.reference_tool_call)].append(case)
    names = sorted(grouped)
    selected = []
    while names and (limit is None or len(selected) < limit):
        retained = []
        for name in names:
            if limit is not None and len(selected) >= limit:
                break
            queue = grouped[name]
            if queue:
                selected.append(queue.popleft())
            if queue:
                retained.append(name)
        names = retained
    return selected


def score_prediction(case: ReplayCase, assistant: Mapping) -> dict:
    """Compare one predicted assistant tool call with the saved reference."""

    predicted_calls = assistant.get("tool_calls") or []
    predicted = _plain_tool_call(predicted_calls[0]) if predicted_calls else None
    reference = case.reference_tool_call
    reference_name = _tool_name(reference)
    reference_args = _arguments(reference)
    predicted_name = _tool_name(predicted) if predicted else ""
    predicted_args = _arguments(predicted) if predicted else None
    valid = bool(predicted and predicted_name and predicted_args is not None)
    predicted_action = None
    legal_reason = "missing_or_invalid_tool_call"
    if valid:
        try:
            predicted_action = tool_call_to_action(predicted_name, predicted_args)
            legal_reason = action_reject_reason(
                predicted_name,
                predicted_args,
                case.latest_observation,
            )
        except (KeyError, TypeError, ValueError):
            legal_reason = "tool_conversion_error"
    reference_action = tool_call_to_action(reference_name, reference_args)
    return {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "trajectory_id": case.trajectory_id,
        "message_index": case.message_index,
        "reference_tool_name": reference_name,
        "reference_arguments": reference_args,
        "reference_action": reference_action,
        "predicted_tool_name": predicted_name or None,
        "predicted_arguments": predicted_args,
        "predicted_action": predicted_action,
        "tool_call_count": len(predicted_calls),
        "valid_tool_call": valid,
        "single_tool_call": len(predicted_calls) == 1,
        "legal_tool_call": valid and legal_reason is None,
        "legality_error": legal_reason,
        "tool_name_match": predicted_name == reference_name,
        "arguments_match": predicted_args == reference_args,
        "action_match": predicted_action == reference_action,
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_results(results: list[Mapping], *, wall_seconds: float) -> dict:
    completed = [row for row in results if not row.get("request_error")]
    latencies = [float(row["latency_seconds"]) for row in completed]

    def rate(name: str) -> float | None:
        if not completed:
            return None
        return sum(bool(row.get(name)) for row in completed) / len(completed)

    by_tool = {}
    tool_names = sorted({str(row.get("reference_tool_name")) for row in completed})
    for tool_name in tool_names:
        rows = [row for row in completed if row.get("reference_tool_name") == tool_name]
        by_tool[tool_name] = {
            "cases": len(rows),
            "legal_tool_call_rate": sum(bool(row.get("legal_tool_call")) for row in rows)
            / len(rows),
            "tool_name_accuracy": sum(bool(row.get("tool_name_match")) for row in rows)
            / len(rows),
            "action_accuracy": sum(bool(row.get("action_match")) for row in rows) / len(rows),
        }
    return {
        "schema_version": "shopping-offline-replay-probe-v1",
        "requested_cases": len(results),
        "completed_cases": len(completed),
        "request_errors": len(results) - len(completed),
        "valid_tool_call_rate": rate("valid_tool_call"),
        "single_tool_call_rate": rate("single_tool_call"),
        "legal_tool_call_rate": rate("legal_tool_call"),
        "tool_name_accuracy": rate("tool_name_match"),
        "argument_accuracy": rate("arguments_match"),
        "action_accuracy": rate("action_match"),
        "wall_seconds": wall_seconds,
        "cases_per_second": len(completed) / wall_seconds if wall_seconds > 0 else None,
        "latency_seconds": {
            "mean": fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p90": _percentile(latencies, 0.9),
        },
        "by_reference_tool": by_tool,
    }
