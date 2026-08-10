"""Streaming GRPO performance log parsing and conservative canary promotion."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Iterable


METRIC_NAMES = (
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/step",
    "perf/total_num_tokens",
    "response_length/mean",
    "timing_per_token_ms/gen",
    "timing_per_token_ms/update_actor",
    "perf/throughput",
    "group/resample_batches",
    "reward/strict_mean",
)
_ERROR_RE = re.compile(
    r"CUDA out of memory|EngineDeadError|NCCL[^\n]*(?:error|failed)|RuntimeError:"
)
_BENIGN_DATALOADER_TEARDOWN_RE = re.compile(
    r"RuntimeError: DataLoader worker .* is killed by signal: Killed"
)


@dataclass(frozen=True)
class StepMetrics:
    step: int
    values: dict[str, float]


def _field(line: str, name: str, *, first: bool = False) -> str | None:
    marker = f"{name}:" if first else f" - {name}:"
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find(" - ", start)
    value = line[start:] if end < 0 else line[start:end]
    if value.startswith("np."):
        opening = value.find("(")
        closing = value.rfind(")")
        if opening >= 0 and closing > opening:
            value = value[opening + 1 : closing]
    return value


def parse_metric_lines(lines: Iterable[str]) -> tuple[list[StepMetrics], int]:
    """Parse veRL metric lines in one pass while retaining only the last row per step."""
    rows: dict[int, StepMetrics] = {}
    errors = 0
    for line in lines:
        # veRL/Ray can print this from StatefulDataLoader.__del__ after a
        # successful final metric and exit.  It is noisy teardown, not a failed
        # training step; real RuntimeError/OOM/NCCL lines still block promotion.
        if _ERROR_RE.search(line) and not _BENIGN_DATALOADER_TEARDOWN_RE.search(line):
            errors += 1
        if "timing_s/step:" not in line:
            continue
        step_value = _field(line, "step", first=True)
        if step_value is None:
            continue
        values = {}
        for name in METRIC_NAMES:
            value = _field(line, name)
            if value is not None:
                values[name] = float(value)
        if all(name in values for name in METRIC_NAMES[:4]):
            step = int(step_value)
            rows[step] = StepMetrics(step=step, values=values)
    return [rows[step] for step in sorted(rows)], errors


def parse_metric_file(path: Path) -> tuple[list[StepMetrics], int]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return parse_metric_lines(handle)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[StepMetrics], *, warmup_steps: int = 1) -> dict:
    retained = rows[warmup_steps:]
    if not retained:
        raise ValueError("no metric rows remain after warmup removal")
    metrics = {}
    for name in METRIC_NAMES:
        values = [row.values[name] for row in retained if name in row.values]
        if values:
            metrics[name] = {
                "mean": fmean(values),
                "median": median(values),
                "p90": percentile(values, 0.9),
                "samples": len(values),
            }
    step_mean = metrics["timing_s/step"]["mean"]
    metrics["timing_share"] = {
        name: metrics[name]["mean"] / step_mean
        for name in METRIC_NAMES[:3]
    }
    return {
        "first_step": retained[0].step,
        "last_step": retained[-1].step,
        "samples": len(retained),
        "metrics": metrics,
    }


def compare_summaries(
    baseline: dict,
    candidate: dict,
    *,
    candidate_errors: int = 0,
    minimum_samples: int = 3,
    minimum_improvement: float = 0.05,
    maximum_hotspot_regression: float = 0.03,
    semantic_check: dict | None = None,
) -> dict:
    base_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    improvements = {}
    for name in METRIC_NAMES[:4]:
        base_mean = base_metrics[name]["mean"]
        candidate_mean = candidate_metrics[name]["mean"]
        improvements[name] = (base_mean - candidate_mean) / base_mean
    normalized_hotspots = {
        "timing_s/gen": "timing_per_token_ms/gen",
        "timing_s/update_actor": "timing_per_token_ms/update_actor",
        "timing_s/update_weights": "timing_s/update_weights",
    }
    normalized_improvements = {}
    for hotspot, metric_name in normalized_hotspots.items():
        if metric_name not in base_metrics or metric_name not in candidate_metrics:
            metric_name = hotspot
        base_mean = base_metrics[metric_name]["mean"]
        candidate_mean = candidate_metrics[metric_name]["mean"]
        normalized_improvements[hotspot] = (base_mean - candidate_mean) / base_mean

    reasons = []
    if semantic_check is None:
        reasons.append("semantic parity report is required before promotion")
    elif semantic_check.get("passed") is not True:
        reasons.append("semantic parity report did not pass")
    if candidate_errors:
        reasons.append(f"candidate log contains {candidate_errors} error match(es)")
    if baseline["samples"] < minimum_samples or candidate["samples"] < minimum_samples:
        reasons.append(f"both runs require at least {minimum_samples} post-warmup samples")
    if improvements["timing_s/step"] < minimum_improvement:
        reasons.append(
            "end-to-end step improvement is below "
            f"{minimum_improvement:.1%}"
        )
    for name in METRIC_NAMES[:3]:
        if normalized_improvements[name] < -maximum_hotspot_regression:
            reasons.append(
                f"{name} normalized cost regressed by "
                f"{-normalized_improvements[name]:.1%}, above "
                f"{maximum_hotspot_regression:.1%}"
            )
    return {
        "promote": not reasons,
        "improvements": improvements,
        "normalized_hotspot_improvements": normalized_improvements,
        "semantic_check": semantic_check,
        "reasons": reasons,
    }
