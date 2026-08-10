"""Auditable gates between a credit-ablation checkpoint and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from shopping_grpo.evaluation.paired_trajectories import compare_files
from shopping_grpo.evaluation.summary import summarize_trajectories

SCHEMA_VERSION = "shopping-credit-postcanary-gate-v1"
SUPPORTED_METHODS = frozenset({"grpo", "shopping_gigpo", "shopping_graphgpo"})
FROZEN_FINAL200_SHA256 = "2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208"
FROZEN_DEV50_SHA256 = "52540f0ea0d68eaa0a14594f9769bf6323aade95e04d869555fac15d7f8af33a"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_METRIC_PATTERNS = {
    "global_step": re.compile(r"training/global_step:(\d+)"),
    "optimizer_updated": re.compile(r"training/optimizer_updated:(\d+)"),
    "graph_source_groups": re.compile(rf"graphgpo/graph_source_groups:({_FLOAT})"),
    "graph_finite": re.compile(rf"graphgpo/finite:({_FLOAT})"),
    "graph_invariant_ok": re.compile(rf"graphgpo/invariant_ok:({_FLOAT})"),
    "gigpo_instrumentation_error_rate": re.compile(
        rf"gigpo/instrumentation_error_rate:({_FLOAT})"
    ),
    "actor_loss": re.compile(rf"actor/loss:(?:np\.float\d+\()?({_FLOAT})"),
    "actor_grad_norm": re.compile(rf"actor/grad_norm:(?:np\.float\d+\()?({_FLOAT})"),
}
_FATAL_LOG_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "ray.exceptions.RayTaskError(OutOfMemoryError)",
    "training exited without",
)


class GateError(RuntimeError):
    """Raised when an artifact cannot safely advance to the next stage."""


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file with bounded memory and large sequential reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(directory: Path) -> list[Path]:
    patterns = ("model*.safetensors", "pytorch_model*.bin")
    return sorted({path for pattern in patterns for path in directory.glob(pattern)})


def weight_digest(directory: Path) -> dict:
    """Return a canonical digest for every standalone model weight shard."""
    directory = Path(directory)
    files = _weight_files(directory)
    if not files:
        raise GateError(f"no standalone model weights found in {directory}")
    shards = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    canonical = json.dumps(shards, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "shards": shards}


def _require_file(path: Path, errors: list[str], *, nonempty: bool = True) -> None:
    if not path.is_file():
        errors.append(f"missing file: {path}")
    elif nonempty and path.stat().st_size == 0:
        errors.append(f"empty file: {path}")


def _extract_metrics(log_path: Path) -> tuple[dict[str, list[float]], list[str]]:
    values = {name: [] for name in _METRIC_PATTERNS}
    fatal_markers: list[str] = []
    with Path(log_path).open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = _ANSI_ESCAPE.sub("", raw_line)
            for marker in _FATAL_LOG_MARKERS:
                if marker in line and marker not in fatal_markers:
                    fatal_markers.append(marker)
            for name, pattern in _METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    values[name].append(float(match.group(1)))
    return values, fatal_markers


def audit_training_checkpoint(
    run_dir: Path,
    supervisor_log: Path,
    *,
    expected_step: int,
    expected_method: str = "shopping_graphgpo",
    expected_train_commit: str | None = None,
    hash_model_shards: bool = True,
) -> dict:
    """Reject incomplete, numerically invalid, zero-signal, or failed canaries."""
    run_dir = Path(run_dir)
    supervisor_log = Path(supervisor_log)
    errors: list[str] = []
    if expected_step < 1:
        raise ValueError("expected_step must be positive")
    if expected_method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported credit method: {expected_method}")

    lineage_path = run_dir / "lineage.json"
    latest_path = run_dir / "latest_checkpointed_iteration.txt"
    actor_dir = run_dir / f"global_step_{expected_step}" / "actor"
    _require_file(lineage_path, errors)
    _require_file(latest_path, errors)
    _require_file(supervisor_log, errors)
    for relative in (
        "fsdp_config.json",
        "lora_train_meta.json",
        "extra_state_world_size_1_rank_0.pt",
        "model_world_size_1_rank_0.pt",
        "optim_world_size_1_rank_0.pt",
    ):
        _require_file(actor_dir / relative, errors)

    lineage: dict = {}
    if lineage_path.is_file():
        try:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid lineage.json: {exc}")
    required_lineage = {
        "run_id",
        "method",
        "git_commit",
        "git_dirty",
        "config_sha256",
        "launcher_sha256",
        "train_data_sha256",
        "val_data_sha256",
    }
    missing_lineage = sorted(required_lineage - set(lineage))
    if missing_lineage:
        errors.append(f"lineage fields missing: {missing_lineage}")
    if lineage.get("git_dirty") is not False:
        errors.append("training worktree was dirty")
    if expected_train_commit and lineage.get("git_commit") != expected_train_commit:
        errors.append(
            "training commit mismatch: "
            f"expected={expected_train_commit} actual={lineage.get('git_commit')}"
        )
    if lineage.get("run_id") and lineage["run_id"] != run_dir.name:
        errors.append(
            f"run ID mismatch: directory={run_dir.name} lineage={lineage['run_id']}"
        )
    if lineage.get("method") and lineage["method"] != expected_method:
        errors.append(
            "training method mismatch: "
            f"expected={expected_method} actual={lineage.get('method')}"
        )
    if latest_path.is_file():
        latest = latest_path.read_text(encoding="utf-8").strip()
        if latest != str(expected_step):
            errors.append(f"latest checkpoint is {latest!r}, expected {expected_step}")

    metrics = {name: [] for name in _METRIC_PATTERNS}
    fatal_markers: list[str] = []
    if supervisor_log.is_file():
        metrics, fatal_markers = _extract_metrics(supervisor_log)
    if fatal_markers:
        errors.append(f"fatal training log markers: {fatal_markers}")
    if expected_step not in {int(value) for value in metrics["global_step"]}:
        errors.append(f"training/global_step:{expected_step} not found")
    if not metrics["optimizer_updated"] or max(metrics["optimizer_updated"]) < 1:
        errors.append("no optimizer update was recorded")
    for name in ("actor_loss", "actor_grad_norm"):
        if not metrics[name] or not all(math.isfinite(value) for value in metrics[name]):
            errors.append(f"{name} is missing or non-finite")
    if metrics["actor_grad_norm"] and not any(
        abs(value) > 1e-12 for value in metrics["actor_grad_norm"]
    ):
        errors.append("all actor gradients are zero")
    if expected_method == "shopping_gigpo":
        instrumentation = metrics["gigpo_instrumentation_error_rate"]
        if not instrumentation or any(value != 0 for value in instrumentation):
            errors.append("GiGPO instrumentation is missing or recorded an error")
    elif expected_method == "shopping_graphgpo":
        if not metrics["graph_source_groups"] or max(metrics["graph_source_groups"]) <= 0:
            errors.append("no GraphGPO graph-signal batch was recorded")
        if not metrics["graph_finite"] or min(metrics["graph_finite"]) < 1:
            errors.append("GraphGPO finite invariant did not hold for every logged update")
        if not metrics["graph_invariant_ok"] or min(metrics["graph_invariant_ok"]) < 1:
            errors.append("GraphGPO structural invariant did not hold for every logged update")

    model_shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
    checkpoint_files = []
    if hash_model_shards and not errors:
        checkpoint_files = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_shards
        ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "gate": "training_checkpoint",
        "passed": not errors,
        "errors": errors,
        "run_dir": str(run_dir),
        "supervisor_log": str(supervisor_log),
        "expected_step": expected_step,
        "expected_method": expected_method,
        "lineage": lineage,
        "metrics": {
            name: {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "last": values[-1] if values else None,
            }
            for name, values in metrics.items()
        },
        "checkpoint_files": checkpoint_files,
    }
    if errors:
        raise GateError("; ".join(errors))
    return receipt


def audit_fsdp_export(export_dir: Path) -> dict:
    """Verify that veRL exported both a base and its trained LoRA sidecar."""
    export_dir = Path(export_dir)
    adapter_dir = export_dir / "lora_adapter"
    errors: list[str] = []
    for path in (
        export_dir / "config.json",
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapter_model.safetensors",
    ):
        _require_file(path, errors)
    base_weights = None
    adapter_hash = None
    try:
        base_weights = weight_digest(export_dir)
    except GateError as exc:
        errors.append(str(exc))
    if (adapter_dir / "adapter_model.safetensors").is_file():
        adapter_hash = sha256_file(adapter_dir / "adapter_model.safetensors")
    if errors:
        raise GateError("; ".join(errors))
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "fsdp_export",
        "passed": True,
        "export_dir": str(export_dir),
        "base_weights": base_weights,
        "adapter": {
            "path": str(adapter_dir),
            "sha256": adapter_hash,
            "size": (adapter_dir / "adapter_model.safetensors").stat().st_size,
        },
    }


def audit_merged_model(base_model: Path, adapter: Path, merged_model: Path) -> dict:
    """Prove that a standalone merge is not the unchanged SFT base."""
    base_model = Path(base_model)
    adapter = Path(adapter)
    merged_model = Path(merged_model)
    manifest_path = merged_model / "merge_manifest.json"
    errors: list[str] = []
    for path in (
        base_model / "config.json",
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        merged_model / "config.json",
        manifest_path,
    ):
        _require_file(path, errors)
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"invalid merge manifest: {exc}")
    if manifest.get("operation") != "peft_merge_and_unload":
        errors.append("merge manifest does not record peft_merge_and_unload")
    manifest_adapter = str((manifest.get("source") or {}).get("adapter") or "")
    if manifest_adapter and Path(manifest_adapter).resolve() != adapter.resolve():
        errors.append(
            f"merge manifest adapter mismatch: expected={adapter} actual={manifest_adapter}"
        )

    base_weights = merged_weights = None
    try:
        base_weights = weight_digest(base_model)
    except GateError as exc:
        errors.append(str(exc))
    try:
        merged_weights = weight_digest(merged_model)
    except GateError as exc:
        errors.append(str(exc))
    if base_weights and merged_weights and base_weights["sha256"] == merged_weights["sha256"]:
        errors.append("merged weights are byte-identical to the SFT base")
    if errors:
        raise GateError("; ".join(errors))
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "standalone_merged_model",
        "passed": True,
        "base_model": str(base_model),
        "adapter": {
            "path": str(adapter),
            "sha256": sha256_file(adapter / "adapter_model.safetensors"),
        },
        "merged_model": str(merged_model),
        "base_weights": base_weights,
        "merged_weights": merged_weights,
        "manifest": manifest,
    }


def _load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_task_ids(path: Path) -> set[int]:
    """Load task IDs from JSONL or Parquet without imposing a base dependency."""
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            from pyarrow import parquet
        except ImportError as exc:  # pragma: no cover - exercised in the veRL environment
            raise GateError("pyarrow is required to audit a Parquet training set") from exc
        schema_names = set(parquet.read_schema(path).names)
        if "task_id" in schema_names:
            table = parquet.read_table(path, columns=["task_id"])
            return {int(value) for value in table.column("task_id").to_pylist()}
        if "extra_info" in schema_names:
            table = parquet.read_table(path, columns=["extra_info"])
            rows = table.column("extra_info").to_pylist()
            try:
                return {int(row["task_id"]) for row in rows}
            except (KeyError, TypeError) as exc:
                raise GateError("Parquet extra_info does not contain task_id") from exc
        raise GateError("Parquet training set has neither task_id nor extra_info.task_id")
    return {int(record["task_id"]) for record in _load_jsonl(path)}


def prepare_frozen_slice(
    benchmark: Path,
    sft_trajectories: Path,
    output_dir: Path,
    *,
    count: int = 20,
    salt: str = "graphgpo-heldout-20260809",
    expected_benchmark_sha256: str | None = FROZEN_FINAL200_SHA256,
    training_data: Path | None = None,
    excluded_task_ids: Iterable[int] = (),
) -> dict:
    """Pre-register a deterministic held-out slice and its paired SFT rows."""
    benchmark = Path(benchmark)
    sft_trajectories = Path(sft_trajectories)
    output_dir = Path(output_dir)
    benchmark_hash = sha256_file(benchmark)
    if expected_benchmark_sha256 and benchmark_hash != expected_benchmark_sha256:
        raise GateError(
            "benchmark hash mismatch: "
            f"expected={expected_benchmark_sha256} actual={benchmark_hash}"
        )
    tasks = _load_jsonl(benchmark)
    task_ids = [int(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise GateError("benchmark contains duplicate task IDs")
    if count < 1 or count > len(tasks):
        raise GateError(f"slice count must be within 1..{len(tasks)}")
    excluded = {int(task_id) for task_id in excluded_task_ids}
    unknown_exclusions = excluded - set(task_ids)
    if unknown_exclusions:
        raise GateError(f"excluded task IDs are absent from benchmark: {sorted(unknown_exclusions)}")
    eligible_ids = set(task_ids) - excluded
    if count > len(eligible_ids):
        raise GateError(
            f"slice count {count} exceeds {len(eligible_ids)} tasks after exclusions"
        )
    ranked = sorted(
        eligible_ids,
        key=lambda task_id: hashlib.sha256(f"{salt}:{task_id}".encode()).digest(),
    )
    selected = set(ranked[:count])
    if training_data is not None:
        overlap = selected & load_task_ids(training_data)
        if overlap:
            raise GateError(f"held-out slice overlaps training tasks: {sorted(overlap)}")

    selected_tasks = [task for task in tasks if int(task["task_id"]) in selected]
    sft_rows = _load_jsonl(sft_trajectories)
    sft_by_id: dict[int, dict] = {}
    for row in sft_rows:
        task_id = int(row["task_id"])
        if task_id in selected:
            if task_id in sft_by_id:
                raise GateError(f"SFT trajectories contain duplicate task_id {task_id}")
            sft_by_id[task_id] = row
    missing = selected - set(sft_by_id)
    if missing:
        raise GateError(f"SFT trajectories are missing selected tasks: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "tasks.jsonl"
    sft_path = output_dir / "sft_trajectories.jsonl"
    tasks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_tasks),
        encoding="utf-8",
    )
    sft_path.write_text(
        "".join(
            json.dumps(sft_by_id[int(task["task_id"])], ensure_ascii=False) + "\n"
            for task in selected_tasks
        ),
        encoding="utf-8",
    )
    sft_summary = summarize_trajectories(
        [task["task_id"] for task in selected_tasks],
        [sft_by_id[int(task["task_id"])] for task in selected_tasks],
    )
    (output_dir / "sft_summary.json").write_text(
        json.dumps(sft_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gate": "frozen_heldout_slice",
        "benchmark": str(benchmark),
        "benchmark_sha256": benchmark_hash,
        "selection": "lowest_sha256(salt:task_id)",
        "salt": salt,
        "count": count,
        "task_ids": [int(task["task_id"]) for task in selected_tasks],
        "excluded_task_ids": sorted(excluded),
        "tasks_sha256": sha256_file(tasks_path),
        "sft_trajectories": str(sft_trajectories),
        "sft_trajectories_sha256": sha256_file(sft_trajectories),
        "sft_slice_sha256": sha256_file(sft_path),
        "training_data": str(training_data) if training_data else None,
        "training_overlap": [],
        "usage": "development promotion gate only; Final-200 remains winner-only",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def compare_heldout_slice(
    sft_trajectories: Path,
    candidate_trajectories: Path,
    output_path: Path,
    *,
    candidate_label: str = "graphgpo",
    max_strict_regression: int = 0,
    min_reward_delta: float = -0.02,
) -> dict:
    """Apply a pre-registered stability gate before any expanded evaluation."""
    if not candidate_label or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", candidate_label):
        raise ValueError(f"invalid candidate label: {candidate_label!r}")
    comparison = compare_files(
        Path(sft_trajectories),
        Path(candidate_trajectories),
        source_label="sft",
        target_label=candidate_label,
    )
    transitions = comparison["strict_success_transitions"]
    repairs = int(transitions.get("failure_to_success", 0))
    regressions = int(transitions.get("success_to_failure", 0))
    reward_delta = float(comparison["mean_target_minus_source"]["final_reward"])
    integrity_pass = comparison["paired_tasks"] > 0
    promotion_pass = (
        integrity_pass
        and repairs - regressions >= -int(max_strict_regression)
        and reward_delta >= float(min_reward_delta)
    )
    comparison["gate"] = {
        "integrity_pass": integrity_pass,
        "promotion_pass": promotion_pass,
        "max_strict_regression": int(max_strict_regression),
        "min_reward_delta": float(min_reward_delta),
        "decision": "EXPAND_DEV" if promotion_pass else "HOLD_AND_ANALYZE",
        "scope": "diagnostic slice; does not establish a Final-200 improvement",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison


def write_json_receipt(receipt: Mapping, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_equivalence_reports(
    reports: Iterable[Mapping],
    *,
    min_top_k_overlap: int = 9,
    max_mean_abs_diff: float = 0.05,
    max_abs_diff: float = 0.25,
) -> dict:
    """Numerical gate for CPU adapter-vs-merged logit checks."""
    reports = [dict(report) for report in reports]
    failures = []
    for index, report in enumerate(reports):
        if report.get("reference_top1") != report.get("candidate_top1"):
            failures.append(f"prompt {index}: top-1 mismatch")
        if int(report.get("top_k_overlap", -1)) < min_top_k_overlap:
            failures.append(f"prompt {index}: insufficient top-k overlap")
        if float(report.get("mean_abs_diff", math.inf)) > max_mean_abs_diff:
            failures.append(f"prompt {index}: mean absolute difference too large")
        if float(report.get("max_abs_diff", math.inf)) > max_abs_diff:
            failures.append(f"prompt {index}: maximum absolute difference too large")
    return {
        "passed": bool(reports) and not failures,
        "failures": failures,
        "thresholds": {
            "min_top_k_overlap": min_top_k_overlap,
            "max_mean_abs_diff": max_mean_abs_diff,
            "max_abs_diff": max_abs_diff,
        },
        "prompts": reports,
    }
