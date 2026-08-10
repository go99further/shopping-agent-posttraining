#!/usr/bin/env python3
"""Deterministic, restart-safe promotion controller for guarded remote attempts.

The controller never invents a promotion decision.  It launches only an argv
registered in the manifest, and it promotes only when every preregistered
machine predicate succeeds.  The CLI is globally dry-run unless ``--execute``
is supplied; a real launch additionally requires ``allow_execute: true`` and
an exact command-spec SHA-256 match in the manifest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "shopping-unattended-promotion-v1"
STATE_SCHEMA_VERSION = "shopping-unattended-promotion-state-v1"
STATES = frozenset(
    {
        "READY",
        "RUNNING",
        "COMPLETE",
        "FAILED",
        "ANALYZED",
        "PROMOTED",
        "DIAGNOSTIC_EXPAND",
        "STOPPED",
    }
)
TERMINAL_STATES = frozenset({"PROMOTED", "DIAGNOSTIC_EXPAND", "STOPPED"})
PROMOTION_STAGES = (
    "origin",
    "held_out",
    "diagnostic_expand",
    "small_dev",
    "one_step",
    "five_step",
)
PREVIOUS_STAGES = {
    "held_out": {"origin"},
    "diagnostic_expand": {"held_out"},
    "small_dev": {"held_out", "diagnostic_expand"},
    "one_step": {"small_dev"},
    "five_step": {"one_step"},
}
GPU_WORKLOADS = frozenset(
    {
        "inference_ab",
        "logprob_forward_parity",
        "sft_canary",
        "grpo_canary",
        "weight_sync",
        "resume",
        "end_to_end_profile",
    }
)
CPU_WORKLOADS = frozenset({"cpu_research"})
QUEUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest cannot satisfy the deterministic controller contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_spec(launch: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact, portable launch fields covered by command_sha256."""

    return {
        "argv": launch.get("argv"),
        "cwd": launch.get("cwd", "."),
        "env": launch.get("env", {}),
    }


def command_sha256(launch: Mapping[str, Any]) -> str:
    return sha256_value(command_spec(launch))


def decision_spec(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every decision-bearing field while permitting only authorization toggle."""

    frozen = json.loads(json.dumps(queue))
    launch = frozen.get("launch")
    if isinstance(launch, dict):
        launch.pop("allow_execute", None)
    return frozen


def _require_type(value: Any, expected: type, label: str) -> None:
    if not isinstance(value, expected):
        raise ManifestError(f"{label} must be {expected.__name__}")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION!r}")
    hygiene = manifest.get("context_hygiene")
    _require_type(hygiene, dict, "context_hygiene")
    for name in ("goal_path", "git_worktree", "gpu_inventory_path", "compaction_marker"):
        if not isinstance(hygiene.get(name), str):
            raise ManifestError(f"context_hygiene.{name} must be a path")
    limits = {
        "interval_seconds": 3600,
        "max_state_bytes": 204800,
        "max_events": 50,
        "max_attempts_per_branch": 3,
    }
    for name, maximum in limits.items():
        value = hygiene.get(name)
        if not isinstance(value, int) or value <= 0 or value > maximum:
            raise ManifestError(f"context_hygiene.{name} must be in [1, {maximum}]")
    queues = manifest.get("queues")
    if not isinstance(queues, list) or not queues:
        raise ManifestError("queues must be a non-empty list")
    seen: set[str] = set()
    for index, queue in enumerate(queues):
        label = f"queues[{index}]"
        _require_type(queue, dict, label)
        queue_id = queue.get("id")
        if not isinstance(queue_id, str) or not QUEUE_ID_RE.fullmatch(queue_id):
            raise ManifestError(f"{label}.id is invalid")
        if queue_id in seen:
            raise ManifestError(f"duplicate queue id: {queue_id}")
        seen.add(queue_id)
        if queue.get("initial_state", "READY") != "READY":
            raise ManifestError(f"{label}.initial_state must be READY")
        stage = queue.get("promotion_stage")
        if stage not in PROMOTION_STAGES:
            raise ManifestError(
                f"{label}.promotion_stage must be one of {list(PROMOTION_STAGES)}"
            )
        prerequisites = queue.get("prerequisites", [])
        if not isinstance(prerequisites, list) or not all(
            isinstance(item, str) for item in prerequisites
        ):
            raise ManifestError(f"{label}.prerequisites must be a string list")
        prerequisite_states = queue.get("prerequisite_states", {})
        if not isinstance(prerequisite_states, dict) or not all(
            key in prerequisites and value in {"PROMOTED", "DIAGNOSTIC_EXPAND"}
            for key, value in prerequisite_states.items()
        ):
            raise ManifestError(f"{label}.prerequisite_states is invalid")
        prerequisite_hashes = queue.get("prerequisite_decision_sha256")
        if not isinstance(prerequisite_hashes, dict) or set(prerequisite_hashes) != set(
            prerequisites
        ) or not all(
            isinstance(value, str) and HEX64_RE.fullmatch(value)
            for value in prerequisite_hashes.values()
        ):
            raise ManifestError(
                f"{label}.prerequisite_decision_sha256 must cover every prerequisite"
            )

        launch = queue.get("launch")
        _require_type(launch, dict, f"{label}.launch")
        argv = launch.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise ManifestError(f"{label}.launch.argv must be a non-empty string list")
        if not isinstance(launch.get("cwd", "."), str):
            raise ManifestError(f"{label}.launch.cwd must be a string")
        if not isinstance(launch.get("owned_process_receipt"), str):
            raise ManifestError(f"{label}.launch.owned_process_receipt must be a path")
        env = launch.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            raise ManifestError(f"{label}.launch.env must map strings to strings")
        if not isinstance(launch.get("allow_execute", False), bool):
            raise ManifestError(f"{label}.launch.allow_execute must be boolean")
        expected_hash = launch.get("command_sha256")
        if not isinstance(expected_hash, str) or not HEX64_RE.fullmatch(expected_hash):
            raise ManifestError(f"{label}.launch.command_sha256 must be lowercase SHA-256")

        markers = queue.get("markers")
        _require_type(markers, dict, f"{label}.markers")
        for marker in ("complete", "failed"):
            if not isinstance(markers.get(marker), str) or not markers[marker]:
                raise ManifestError(f"{label}.markers.{marker} must be a path string")

        resources = queue.get("resources")
        _require_type(resources, dict, f"{label}.resources")
        resource_class = queue.get("resource_class")
        workload_kind = queue.get("workload_kind")
        if resource_class not in {"GPU_ELIGIBLE", "CPU_ONLY"}:
            raise ManifestError(f"{label}.resource_class is invalid")
        if resource_class == "GPU_ELIGIBLE" and workload_kind not in GPU_WORKLOADS:
            raise ManifestError(
                f"{label}.workload_kind must be a real GPU-eligible workload"
            )
        if resource_class == "CPU_ONLY" and workload_kind not in CPU_WORKLOADS:
            raise ManifestError(f"{label}.workload_kind must be cpu_research")
        ports = resources.get("ports")
        gpu_uuids = resources.get("gpu_uuids")
        if not isinstance(ports, list) or not all(
            isinstance(port, int) and 0 < port < 65536 for port in ports
        ):
            raise ManifestError(f"{label}.resources.ports must contain valid ports")
        if not isinstance(gpu_uuids, list) or not all(
            isinstance(uuid, str) and uuid for uuid in gpu_uuids
        ):
            raise ManifestError(f"{label}.resources.gpu_uuids must be a string list")
        _require_type(queue.get("checkpoint"), dict, f"{label}.checkpoint")
        lineage = queue.get("attempt_lineage")
        _require_type(lineage, dict, f"{label}.attempt_lineage")
        if lineage.get("attempt_id") != queue_id:
            raise ManifestError(f"{label}.attempt_lineage.attempt_id must equal id")
        if not isinstance(lineage.get("branch_id"), str) or not lineage["branch_id"]:
            raise ManifestError(f"{label}.attempt_lineage.branch_id is required")

        preflight = queue.get("preflight")
        _require_type(preflight, dict, f"{label}.preflight")
        authentication = preflight.get("authentication")
        _require_type(authentication, dict, f"{label}.preflight.authentication")
        if authentication.get("type") not in {"none", "environment", "file"}:
            raise ManifestError(f"{label}.preflight.authentication.type is invalid")
        directories = preflight.get("directories")
        if not isinstance(directories, list) or not directories:
            raise ManifestError(f"{label}.preflight.directories must be non-empty")
        disk = preflight.get("disk")
        _require_type(disk, dict, f"{label}.preflight.disk")
        if not isinstance(disk.get("path"), str) or not isinstance(
            disk.get("min_free_bytes"), int
        ):
            raise ManifestError(f"{label}.preflight.disk path/min_free_bytes are required")
        ports_preflight = preflight.get("ports")
        if not isinstance(ports_preflight, list):
            raise ManifestError(f"{label}.preflight.ports must be a list")
        if not all(
            isinstance(item, dict)
            and item.get("host") in {"127.0.0.1", "localhost", "::1"}
            and isinstance(item.get("port"), int)
            and 0 < item["port"] < 65536
            for item in ports_preflight
        ):
            raise ManifestError(f"{label}.preflight.ports entries are invalid")
        registered_ports = sorted(resources["ports"])
        checked_ports = sorted(item["port"] for item in ports_preflight)
        if checked_ports != registered_ports:
            raise ManifestError(
                f"{label}.preflight.ports must cover every registered resource port"
            )
        heartbeat = preflight.get("heartbeat")
        _require_type(heartbeat, dict, f"{label}.preflight.heartbeat")
        if not isinstance(heartbeat.get("path"), str) or not all(
            isinstance(heartbeat.get(name), int) and heartbeat[name] > 0
            for name in ("stale_seconds", "startup_grace_seconds")
        ):
            raise ManifestError(f"{label}.preflight.heartbeat is invalid")
        checkpoint_resume = preflight.get("checkpoint_resume")
        _require_type(checkpoint_resume, dict, f"{label}.preflight.checkpoint_resume")
        if checkpoint_resume.get("mode") not in {"fresh", "resume"}:
            raise ManifestError(f"{label}.preflight.checkpoint_resume.mode is invalid")
        if checkpoint_resume.get("mode") == "resume" and not isinstance(
            checkpoint_resume.get("receipt"), str
        ):
            raise ManifestError(
                f"{label}.preflight.checkpoint_resume.receipt is required for resume"
            )
        http_probes = preflight.get("http_probes", [])
        if not isinstance(http_probes, list):
            raise ManifestError(f"{label}.preflight.http_probes must be a list")

        analysis = queue.get("analysis")
        _require_type(analysis, dict, f"{label}.analysis")
        conditions = analysis.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ManifestError(f"{label}.analysis.conditions must be non-empty")
        for condition_index, condition in enumerate(conditions):
            _require_type(condition, dict, f"{label}.analysis.conditions[{condition_index}]")
            if condition.get("type") not in {
                "file_exists",
                "file_absent",
                "sha256_equals",
                "text_contains",
                "json_value",
            }:
                raise ManifestError(
                    f"{label}.analysis.conditions[{condition_index}].type is unsupported"
                )
            on_failure = condition.get("on_failure", "STOP")
            if on_failure not in {"STOP", "DIAGNOSTIC_EXPAND"}:
                raise ManifestError(
                    f"{label}.analysis.conditions[{condition_index}].on_failure is invalid"
                )
            if condition.get("protocol_invariant", False) and on_failure != "STOP":
                raise ManifestError(
                    f"{label}.analysis.conditions[{condition_index}] protocol invariant "
                    "must hard STOP"
                )
        diagnostic_conditions = [
            condition
            for condition in conditions
            if condition.get("on_failure") == "DIAGNOSTIC_EXPAND"
        ]
        diagnostic_expansion = queue.get("diagnostic_expansion")
        if diagnostic_conditions:
            if stage != "held_out":
                raise ManifestError(
                    f"{label} may request diagnostic expansion only at held_out"
                )
            _require_type(
                diagnostic_expansion, dict, f"{label}.diagnostic_expansion"
            )
            if (
                not isinstance(diagnostic_expansion.get("queue_id"), str)
                or not isinstance(diagnostic_expansion.get("max_gpu_hours"), (int, float))
                or diagnostic_expansion["max_gpu_hours"] <= 0
                or not isinstance(diagnostic_expansion.get("max_tasks"), int)
                or diagnostic_expansion["max_tasks"] <= 0
            ):
                raise ManifestError(f"{label}.diagnostic_expansion is invalid")
        elif diagnostic_expansion is not None:
            raise ManifestError(
                f"{label}.diagnostic_expansion requires a DIAGNOSTIC_EXPAND condition"
            )

    unknown = {
        prerequisite
        for queue in queues
        for prerequisite in queue.get("prerequisites", [])
        if prerequisite not in seen
    }
    if unknown:
        raise ManifestError(f"unknown prerequisites: {sorted(unknown)}")
    queue_by_id = {queue["id"]: queue for queue in queues}
    for queue in queues:
        for dependency in queue.get("prerequisites", []):
            actual = sha256_value(decision_spec(queue_by_id[dependency]))
            expected = queue["prerequisite_decision_sha256"][dependency]
            if actual != expected:
                raise ManifestError(
                    f"stale prerequisite reference {queue['id']} -> {dependency}: "
                    f"expected {expected}, current {actual}"
                )
    diagnostic_targets: set[str] = set()
    for queue in queues:
        stage = queue["promotion_stage"]
        if stage == "origin":
            continue
        permitted_previous = PREVIOUS_STAGES[stage]
        if not any(
            queue_by_id[dependency]["promotion_stage"] in permitted_previous
            for dependency in queue.get("prerequisites", [])
        ):
            raise ManifestError(
                f"queue {queue['id']} must depend on one of the previous stages "
                f"{sorted(permitted_previous)}"
            )
        diagnostic = queue.get("diagnostic_expansion")
        if diagnostic is None:
            continue
        target_id = diagnostic["queue_id"]
        if target_id not in queue_by_id:
            raise ManifestError(f"diagnostic queue does not exist: {target_id}")
        target = queue_by_id[target_id]
        if (
            target["promotion_stage"] != "diagnostic_expand"
            or queue["id"] not in target.get("prerequisites", [])
            or target.get("prerequisite_states", {}).get(queue["id"])
            != "DIAGNOSTIC_EXPAND"
        ):
            raise ManifestError(
                f"diagnostic queue {target_id} must be a bounded diagnostic_expand child "
                f"of {queue['id']}"
            )
        if target_id in diagnostic_targets:
            raise ManifestError(f"diagnostic queue reused by multiple sources: {target_id}")
        diagnostic_targets.add(target_id)
    unregistered_diagnostics = {
        queue["id"]
        for queue in queues
        if queue["promotion_stage"] == "diagnostic_expand"
        and queue["id"] not in diagnostic_targets
    }
    if unregistered_diagnostics:
        raise ManifestError(
            f"diagnostic queues lack one preregistered source: {sorted(unregistered_diagnostics)}"
        )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(queue_id: str) -> None:
        if queue_id in visiting:
            raise ManifestError(f"prerequisite cycle includes queue {queue_id}")
        if queue_id in visited:
            return
        visiting.add(queue_id)
        for dependency in queue_by_id[queue_id].get("prerequisites", []):
            visit(dependency)
        visiting.remove(queue_id)
        visited.add(queue_id)

    for queue_id in queue_by_id:
        visit(queue_id)


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tail_text(path: Path, limit: int = 65_536) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def count_nonempty_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def command_output(argv: Sequence[str], cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def launch_detached(
    argv: Sequence[str], cwd: Path, environment: Mapping[str, str], log_handle: Any
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def proc_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, separator, suffix = text.rpartition(")")
    fields = suffix.split() if separator else text.split()
    index = 19 if separator else 21
    try:
        return int(fields[index])
    except (IndexError, ValueError):
        return None


def process_identity_matches(receipt: Mapping[str, Any], proc_root: Path = Path("/proc")) -> bool:
    pid = receipt.get("pid")
    ticks = receipt.get("start_ticks")
    return (
        isinstance(pid, int)
        and isinstance(ticks, int)
        and proc_start_ticks(pid, proc_root) == ticks
    )


def read_key_value_receipt(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
    return values


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = value
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(token)
    return current


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "in":
        return actual in expected
    if operator == "not_in":
        return actual not in expected
    raise ValueError(f"unsupported comparison operator: {operator}")


def evaluate_condition(condition: Mapping[str, Any], manifest_dir: Path) -> dict[str, Any]:
    kind = condition["type"]
    path_value = condition.get("path")
    result: dict[str, Any] = {"condition": dict(condition), "passed": False}
    try:
        if not isinstance(path_value, str):
            raise TypeError("condition path must be a string")
        path = resolve_path(path_value, manifest_dir)
        result["resolved_path"] = str(path)
        if kind == "file_exists":
            passed = path.is_file() and (
                not condition.get("nonempty", False) or path.stat().st_size > 0
            )
            actual: Any = {
                "exists": path.is_file(),
                "size": path.stat().st_size if path.is_file() else None,
            }
        elif kind == "file_absent":
            passed = not path.exists()
            actual = {"exists": path.exists()}
        elif kind == "sha256_equals":
            actual = sha256_file(path)
            passed = actual == condition.get("expected")
        elif kind == "text_contains":
            needle = condition.get("text")
            if not isinstance(needle, str):
                raise ValueError("text_contains.text must be a string")
            actual = needle in path.read_text(encoding="utf-8", errors="replace")
            passed = bool(actual)
        else:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            pointer = condition.get("pointer", "")
            operator = condition.get("op", "eq")
            if not isinstance(pointer, str) or not isinstance(operator, str):
                raise ValueError("json_value pointer and op must be strings")
            actual = _json_pointer(payload, pointer)
            passed = _compare(actual, operator, condition.get("expected"))
        result.update({"passed": passed, "actual": actual})
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def http_status(url: str, timeout_seconds: float) -> int | None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "shopping-unattended-promotion-controller/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, urllib.error.URLError):
        return None


def local_port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError:
        return False
    return True


class PromotionController:
    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path,
        *,
        execute: bool = False,
        proc_root: Path = Path("/proc"),
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest_dir = self.manifest_path.parent
        self.output_dir = output_dir.resolve()
        self.execute = execute
        self.proc_root = proc_root
        self._context_triggers: set[str] = set()

    def load_manifest(self) -> dict[str, Any]:
        manifest = read_json(self.manifest_path)
        validate_manifest(manifest)
        return manifest

    def queue_dir(self, queue_id: str) -> Path:
        return self.output_dir / "queues" / queue_id

    def state_path(self, queue_id: str) -> Path:
        return self.queue_dir(queue_id) / "state.json"

    def _event(self, queue_id: str, event: str, **fields: Any) -> None:
        append_jsonl(
            self.queue_dir(queue_id) / "events.jsonl",
            {"timestamp": utc_now(), "event": event, **fields},
        )

    def _marker(self, queue_id: str, name: str, payload: Mapping[str, Any]) -> None:
        atomic_write_json(
            self.queue_dir(queue_id) / name,
            {"timestamp": utc_now(), **payload},
        )

    def _initial_state(
        self, queue: Mapping[str, Any], manifest_sha: str, hygiene_sha: str
    ) -> dict[str, Any]:
        now = utc_now()
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "queue_id": queue["id"],
            "state": "READY",
            "original_outcome": None,
            "decision_sha256": sha256_value(decision_spec(queue)),
            "context_hygiene_sha256": hygiene_sha,
            "manifest_sha256": manifest_sha,
            "latest_manifest_sha256": manifest_sha,
            "created_at": now,
            "updated_at": now,
            "launch_receipt": None,
            "analysis": None,
            "history": [{"state": "READY", "timestamp": now, "reason": "initialized"}],
        }

    def _load_state(
        self, queue: Mapping[str, Any], manifest_sha: str, hygiene_sha: str
    ) -> dict[str, Any]:
        path = self.state_path(queue["id"])
        if not path.exists():
            state = self._initial_state(queue, manifest_sha, hygiene_sha)
            atomic_write_json(path, state)
            self._event(queue["id"], "STATE_INITIALIZED", state="READY")
            return state
        state = read_json(path)
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ManifestError(f"unsupported state schema for {queue['id']}")
        if state.get("queue_id") != queue["id"] or state.get("state") not in STATES:
            raise ManifestError(f"invalid persisted state for {queue['id']}")
        state["latest_manifest_sha256"] = manifest_sha
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path(state["queue_id"]), state)

    def _transition(self, state: dict[str, Any], target: str, reason: str) -> None:
        source = state["state"]
        allowed = {
            "READY": {"RUNNING"},
            "RUNNING": {"COMPLETE", "FAILED"},
            "COMPLETE": {"ANALYZED"},
            "FAILED": {"ANALYZED"},
            "ANALYZED": {"PROMOTED", "DIAGNOSTIC_EXPAND", "STOPPED"},
            "PROMOTED": set(),
            "DIAGNOSTIC_EXPAND": set(),
            "STOPPED": set(),
        }
        if target not in allowed[source]:
            raise RuntimeError(f"invalid transition {source} -> {target}")
        now = utc_now()
        state["state"] = target
        state["history"].append({"state": target, "timestamp": now, "reason": reason})
        self._event(
            state["queue_id"],
            "STATE_TRANSITION",
            source=source,
            target=target,
            reason=reason,
        )
        self._save_state(state)
        if target in TERMINAL_STATES or target in {"COMPLETE", "FAILED"}:
            self._context_triggers.add(f"terminal_change:{state['queue_id']}:{target}")

    def _block(self, queue_id: str, marker: str, reason: str, **evidence: Any) -> None:
        payload = {"reason": reason, **evidence}
        self._marker(queue_id, marker, payload)
        self._event(queue_id, marker, **payload)

    def _safe_block(self, queue_id: str, marker: str, reason: str, **evidence: Any) -> None:
        try:
            self._block(queue_id, marker, reason, **evidence)
        except OSError as error:
            append_jsonl(
                self.output_dir / "controller-errors.jsonl",
                {
                    "timestamp": utc_now(),
                    "queue_id": queue_id,
                    "marker": marker,
                    "reason": reason,
                    "marker_write_error": f"{type(error).__name__}: {error}",
                    "evidence": evidence,
                },
            )

    def _requirements(self, queue: Mapping[str, Any]) -> tuple[str | None, list[str]]:
        launch = queue["launch"]
        preflight = queue["preflight"]
        cwd = resolve_path(launch.get("cwd", "."), self.manifest_dir)
        approval: list[str] = []
        blocked: list[str] = []
        if not cwd.is_dir():
            blocked.append(f"cwd_missing:{cwd}")
        elif not os.access(cwd, os.R_OK | os.X_OK):
            approval.append(f"cwd_permission:{cwd}")
        if self.execute and not (self.proc_root / "self/stat").is_file():
            blocked.append(f"proc_start_ticks_unavailable:{self.proc_root}")

        authentication = preflight["authentication"]
        auth_type = authentication["type"]
        if auth_type == "environment":
            name = authentication.get("name")
            if not isinstance(name, str) or not os.environ.get(name):
                approval.append(f"authentication_environment_missing:{name}")
        elif auth_type == "file":
            value = authentication.get("path")
            if not isinstance(value, str):
                blocked.append("authentication_file_path_missing")
            else:
                path = resolve_path(value, self.manifest_dir)
                if not path.is_file():
                    blocked.append(f"authentication_file_missing:{path}")
                elif not os.access(path, os.R_OK):
                    approval.append(f"authentication_file_permission:{path}")

        for directory in preflight["directories"]:
            if not isinstance(directory, dict) or not isinstance(directory.get("path"), str):
                blocked.append("malformed_preflight_directory")
                continue
            path = resolve_path(directory["path"], self.manifest_dir)
            if not path.is_dir():
                blocked.append(f"directory_missing:{path}")
                continue
            access = directory.get("access", "rx")
            mode = 0
            for character, flag in (("r", os.R_OK), ("w", os.W_OK), ("x", os.X_OK)):
                if character in access:
                    mode |= flag
            if mode and not os.access(path, mode):
                approval.append(f"directory_permission_{access}:{path}")

        disk = preflight["disk"]
        disk_path = resolve_path(disk["path"], self.manifest_dir)
        if not disk_path.exists():
            blocked.append(f"disk_probe_path_missing:{disk_path}")
        else:
            try:
                free_bytes = shutil.disk_usage(disk_path).free
            except OSError as error:
                blocked.append(f"disk_probe_failed:{disk_path}:{error}")
            else:
                if free_bytes < disk["min_free_bytes"]:
                    blocked.append(
                        f"disk_free_below_minimum:{disk_path}:{free_bytes}:"
                        f"{disk['min_free_bytes']}"
                    )

        for port_probe in preflight["ports"]:
            if not local_port_available(port_probe["host"], port_probe["port"]):
                blocked.append(
                    f"port_unavailable:{port_probe['host']}:{port_probe['port']}"
                )

        checkpoint_path_value = queue["checkpoint"].get("input")
        if not isinstance(checkpoint_path_value, str):
            blocked.append("checkpoint_input_missing")
        else:
            checkpoint_path = resolve_path(checkpoint_path_value, self.manifest_dir)
            if not checkpoint_path.exists():
                blocked.append(f"checkpoint_missing:{checkpoint_path}")
        checkpoint_resume = preflight["checkpoint_resume"]
        if checkpoint_resume["mode"] == "resume":
            receipt = resolve_path(checkpoint_resume["receipt"], self.manifest_dir)
            if not receipt.is_file():
                blocked.append(f"resume_receipt_missing:{receipt}")

        for probe in preflight.get("http_probes", []):
            if not isinstance(probe, dict) or not isinstance(probe.get("url"), str):
                blocked.append("malformed_http_probe")
                continue
            timeout_seconds = probe.get("timeout_seconds", 2.0)
            expected_status = probe.get("expected_status", [200])
            if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
                blocked.append(f"http_probe_invalid_timeout:{probe.get('url')}")
                continue
            if not isinstance(expected_status, list) or not all(
                isinstance(status, int) for status in expected_status
            ):
                blocked.append(f"http_probe_invalid_expected_status:{probe.get('url')}")
                continue
            status = http_status(probe["url"], float(timeout_seconds))
            if status not in expected_status:
                blocked.append(f"http_probe_status:{probe['url']}:{status}")

        for requirement in queue.get("requires", []):
            if not isinstance(requirement, dict):
                blocked.append("malformed_requirement")
                continue
            kind = requirement.get("type")
            if kind == "executable":
                name = requirement.get("name")
                if not isinstance(name, str) or shutil.which(name) is None:
                    blocked.append(f"executable_missing:{name}")
            elif kind == "environment":
                name = requirement.get("name")
                if not isinstance(name, str) or not os.environ.get(name):
                    blocked.append(f"environment_missing:{name}")
            elif kind == "path":
                value = requirement.get("path")
                if not isinstance(value, str):
                    blocked.append("path_requirement_missing_path")
                    continue
                path = resolve_path(value, self.manifest_dir)
                if not path.exists():
                    blocked.append(f"path_missing:{path}")
                    continue
                access = requirement.get("access", "r")
                mode = 0
                for character, flag in (("r", os.R_OK), ("w", os.W_OK), ("x", os.X_OK)):
                    if character in access:
                        mode |= flag
                if mode and not os.access(path, mode):
                    approval.append(f"path_permission_{access}:{path}")
            else:
                blocked.append(f"unsupported_requirement:{kind}")
        if approval:
            return "APPROVAL_REQUIRED", approval
        if blocked:
            return "BLOCKED_EVIDENCE", blocked
        return None, []

    def _launch(self, queue: Mapping[str, Any], state: dict[str, Any]) -> None:
        queue_id = queue["id"]
        launch = queue["launch"]
        actual_hash = command_sha256(launch)
        dry_run_payload = {
            "command_sha256": actual_hash,
            "registered_command_sha256": launch["command_sha256"],
            "command": command_spec(launch),
            "controller_execute_enabled": self.execute,
            "manifest_allow_execute": launch.get("allow_execute", False),
        }
        self._marker(queue_id, "DRY_RUN", dry_run_payload)
        if not self.execute or not launch.get("allow_execute", False):
            self._event(queue_id, "DRY_RUN", **dry_run_payload)
            return
        if actual_hash != launch["command_sha256"]:
            self._block(
                queue_id,
                "BLOCKED_EVIDENCE",
                "command_hash_mismatch",
                actual=actual_hash,
                expected=launch["command_sha256"],
            )
            return
        marker, evidence = self._requirements(queue)
        if marker is not None:
            self._block(queue_id, marker, "launch_preflight_failed", evidence=evidence)
            return
        previous_block_text = tail_text(
            self.queue_dir(queue_id) / "BLOCKED_EVIDENCE", 16_384
        )
        if "http_probe_status" in previous_block_text and ":502" in previous_block_text:
            self._context_triggers.add(f"502_reconnect:{queue_id}")

        queue_dir = self.queue_dir(queue_id)
        queue_dir.mkdir(parents=True, exist_ok=True)
        log_path = queue_dir / "launch.log"
        cwd = resolve_path(launch.get("cwd", "."), self.manifest_dir)
        environment = os.environ.copy()
        environment.update(launch.get("env", {}))
        with log_path.open("ab", buffering=0) as log_handle:
            process = launch_detached(launch["argv"], cwd, environment, log_handle)
        start_ticks = proc_start_ticks(process.pid, self.proc_root)
        receipt = {
            "pid": process.pid,
            "start_ticks": start_ticks,
            "command_sha256": actual_hash,
            "argv": launch["argv"],
            "cwd": str(cwd),
            "ports": queue["resources"]["ports"],
            "gpu_uuids": queue["resources"]["gpu_uuids"],
            "checkpoint": queue["checkpoint"],
            "attempt_lineage": queue["attempt_lineage"],
            "heartbeat": queue["preflight"]["heartbeat"],
            "owned_process_receipt": str(
                resolve_path(launch["owned_process_receipt"], self.manifest_dir)
            ),
            "launched_at": utc_now(),
            "launched_at_unix": time.time(),
        }
        state["launch_receipt"] = receipt
        atomic_write_json(queue_dir / "launch-receipt.json", receipt)
        self._transition(state, "RUNNING", "registered command launched")
        if start_ticks is None:
            self._block(
                queue_id,
                "BLOCKED_EVIDENCE",
                "launched_wrapper_missing_start_ticks_branch_frozen",
                pid=process.pid,
            )

    def _refresh_owned_process(
        self, queue: Mapping[str, Any], state: dict[str, Any]
    ) -> dict[str, Any] | None:
        receipt = state.get("launch_receipt")
        if not isinstance(receipt, dict):
            return None
        ownership_path = resolve_path(
            queue["launch"]["owned_process_receipt"], self.manifest_dir
        )
        values = read_key_value_receipt(ownership_path)
        try:
            pid = int(values["pid"])
            start_ticks = int(values["start_ticks"])
        except (KeyError, ValueError):
            owned = receipt.get("owned_process")
            return owned if isinstance(owned, dict) else None
        owned = {
            "pid": pid,
            "start_ticks": start_ticks,
            "role": values.get("role", ownership_path.stem),
            "scope": values.get("scope", "pid"),
            "pgid": int(values["pgid"]) if values.get("pgid", "").isdigit() else None,
            "sid": int(values["sid"]) if values.get("sid", "").isdigit() else None,
            "observed_command_sha256": values.get("command_sha256"),
            "receipt_path": str(ownership_path),
        }
        if receipt.get("owned_process") != owned:
            receipt["owned_process"] = owned
            state["launch_receipt"] = receipt
            atomic_write_json(self.queue_dir(queue["id"]) / "launch-receipt.json", receipt)
            self._save_state(state)
        return owned

    def _observe_running(self, queue: Mapping[str, Any], state: dict[str, Any]) -> None:
        queue_id = queue["id"]
        complete = resolve_path(queue["markers"]["complete"], self.manifest_dir).is_file()
        failed = resolve_path(queue["markers"]["failed"], self.manifest_dir).is_file()
        if complete and failed:
            state["original_outcome"] = "FAILED"
            self._block(queue_id, "BLOCKED_EVIDENCE", "conflicting_terminal_markers")
            self._transition(state, "FAILED", "conflicting terminal markers")
            return
        if complete:
            state["original_outcome"] = "COMPLETE"
            self._transition(state, "COMPLETE", "complete marker observed")
            return
        if failed:
            state["original_outcome"] = "FAILED"
            self._transition(state, "FAILED", "failed marker observed")
            return
        receipt = state.get("launch_receipt")
        owned = self._refresh_owned_process(queue, state)
        wrapper_alive = isinstance(receipt, dict) and process_identity_matches(
            receipt, self.proc_root
        )
        owned_alive = isinstance(owned, dict) and process_identity_matches(
            owned, self.proc_root
        )
        if not wrapper_alive and not owned_alive:
            state["original_outcome"] = "FAILED"
            self._block(
                queue_id,
                "BLOCKED_EVIDENCE",
                "process_exited_or_identity_mismatch_without_terminal_marker",
                launch_receipt=receipt,
            )
            self._transition(state, "FAILED", "owned process unavailable without marker")
            return
        heartbeat = queue["preflight"]["heartbeat"]
        heartbeat_path = resolve_path(heartbeat["path"], self.manifest_dir)
        launched_at = receipt.get("launched_at_unix")
        now = time.time()
        grace_deadline = (
            float(launched_at) + heartbeat["startup_grace_seconds"]
            if isinstance(launched_at, (int, float))
            else now
        )
        try:
            heartbeat_age = max(0.0, now - heartbeat_path.stat().st_mtime)
        except OSError:
            if now > grace_deadline:
                self._block(
                    queue_id,
                    "BLOCKED_EVIDENCE",
                    "heartbeat_missing_branch_frozen",
                    heartbeat_path=str(heartbeat_path),
                )
            return
        if heartbeat_age > heartbeat["stale_seconds"]:
            self._block(
                queue_id,
                "BLOCKED_EVIDENCE",
                "heartbeat_stale_branch_frozen",
                heartbeat_path=str(heartbeat_path),
                heartbeat_age_seconds=heartbeat_age,
                stale_seconds=heartbeat["stale_seconds"],
            )

    def _analyze(self, queue: Mapping[str, Any], state: dict[str, Any]) -> None:
        results = [
            evaluate_condition(condition, self.manifest_dir)
            for condition in queue["analysis"]["conditions"]
        ]
        analysis = {
            "evaluated_at": utc_now(),
            "conditions_sha256": sha256_value(queue["analysis"]["conditions"]),
            "results": results,
            "all_passed": all(result["passed"] for result in results),
        }
        state["analysis"] = analysis
        atomic_write_json(self.queue_dir(queue["id"]) / "analysis.json", analysis)
        if any("error" in result for result in results):
            self._block(
                queue["id"],
                "BLOCKED_EVIDENCE",
                "analysis_evidence_missing_or_invalid",
                failed_conditions=[result for result in results if not result["passed"]],
            )
            self._save_state(state)
            return
        self._transition(state, "ANALYZED", "preregistered machine predicates evaluated")

    def _decide(self, queue: Mapping[str, Any], state: dict[str, Any]) -> None:
        analysis = state.get("analysis") or {}
        results = analysis.get("results", [])
        failed_results = [result for result in results if not result.get("passed")]
        complete = state.get("original_outcome") == "COMPLETE"
        if complete and not failed_results:
            self._marker(
                queue["id"],
                "PROMOTED",
                {
                    "reason": "complete outcome and every preregistered condition passed",
                    "conditions_sha256": analysis.get("conditions_sha256"),
                },
            )
            self._transition(state, "PROMOTED", "all promotion gates passed")
            return
        diagnostic_failures = [
            result
            for result in failed_results
            if result.get("condition", {}).get("on_failure") == "DIAGNOSTIC_EXPAND"
        ]
        hard_stop_failures = [
            result
            for result in failed_results
            if result.get("condition", {}).get("on_failure", "STOP") == "STOP"
        ]
        diagnostic = queue.get("diagnostic_expansion")
        if complete and diagnostic_failures and not hard_stop_failures and diagnostic:
            self._marker(
                queue["id"],
                "DIAGNOSTIC_EXPAND",
                {
                    "reason": "bounded diagnostic expansion preregistered for soft failure",
                    "diagnostic_queue_id": diagnostic["queue_id"],
                    "max_gpu_hours": diagnostic["max_gpu_hours"],
                    "max_tasks": diagnostic["max_tasks"],
                    "conditions_sha256": analysis.get("conditions_sha256"),
                    "failed_conditions": diagnostic_failures,
                },
            )
            self._transition(
                state,
                "DIAGNOSTIC_EXPAND",
                "soft failure requires one bounded diagnostic expansion",
            )
            return
        self._marker(
            queue["id"],
            "STOPPED",
            {
                "reason": "failed outcome or at least one preregistered condition failed",
                "original_outcome": state.get("original_outcome"),
                "conditions_sha256": analysis.get("conditions_sha256"),
            },
        )
        self._transition(state, "STOPPED", "promotion gates did not pass")

    def process_queue(
        self,
        queue: Mapping[str, Any],
        states_by_id: Mapping[str, Mapping[str, Any]],
        manifest_sha: str,
        hygiene_sha: str,
    ) -> dict[str, Any]:
        state = self._load_state(queue, manifest_sha, hygiene_sha)
        if state.get("context_hygiene_sha256") != hygiene_sha:
            self._block(
                queue["id"],
                "BLOCKED_EVIDENCE",
                "context_hygiene_changed_after_queue_initialization",
                original=state.get("context_hygiene_sha256"),
                current=hygiene_sha,
            )
            self._save_state(state)
            return state
        current_digest = sha256_value(decision_spec(queue))
        if state["decision_sha256"] != current_digest:
            self._block(
                queue["id"],
                "BLOCKED_EVIDENCE",
                "decision_bearing_manifest_fields_changed",
                original=state["decision_sha256"],
                current=current_digest,
            )
            self._save_state(state)
            return state
        self._save_state(state)
        if state["state"] in TERMINAL_STATES:
            return state
        if state["state"] == "READY":
            required_states = queue.get("prerequisite_states", {})
            unmet = [
                dependency
                for dependency in queue.get("prerequisites", [])
                if states_by_id.get(dependency, {}).get("state")
                != required_states.get(dependency, "PROMOTED")
            ]
            if unmet:
                self._event(queue["id"], "PREREQUISITES_UNMET", prerequisites=unmet)
                return state
            self._launch(queue, state)
        elif state["state"] == "RUNNING":
            self._observe_running(queue, state)
        elif state["state"] in {"COMPLETE", "FAILED"}:
            self._analyze(queue, state)
        elif state["state"] == "ANALYZED":
            self._decide(queue, state)
        return state

    def _gpu_readiness(
        self,
        queue: Mapping[str, Any],
        state: Mapping[str, Any],
        states_by_id: Mapping[str, Mapping[str, Any]],
    ) -> str:
        if queue["resource_class"] != "GPU_ELIGIBLE":
            return "CPU_ONLY"
        if state.get("state") != "READY":
            return "NOT_READY"
        if state.get("decision_sha256") != sha256_value(decision_spec(queue)):
            return "BLOCKED_EVIDENCE"
        if command_sha256(queue["launch"]) != queue["launch"]["command_sha256"]:
            return "BLOCKED_EVIDENCE"
        if not queue["launch"].get("allow_execute", False):
            return "NOT_AUTHORIZED"
        required_states = queue.get("prerequisite_states", {})
        unmet = [
            dependency
            for dependency in queue.get("prerequisites", [])
            if states_by_id.get(dependency, {}).get("state")
            != required_states.get(dependency, "PROMOTED")
        ]
        if unmet:
            if any(
                states_by_id.get(dependency, {}).get("state") in {"STOPPED"}
                for dependency in unmet
            ):
                return "NOT_READY"
            return "READY_AFTER_CURRENT"
        marker, _ = self._requirements(queue)
        return "READY_NOW" if marker is None else marker

    def _path_evidence(self, path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
        except OSError:
            return {"path": str(path), "exists": False}
        evidence: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        }
        if path.is_file():
            try:
                evidence["sha256"] = sha256_file(path)
            except OSError:
                pass
        return evidence

    def _queue_context(
        self, queue: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = state.get("launch_receipt")
        identities: list[dict[str, Any]] = []
        if isinstance(receipt, dict):
            for role, candidate in (
                ("launch_wrapper", receipt),
                ("owned_process", receipt.get("owned_process")),
            ):
                if not isinstance(candidate, dict):
                    continue
                pid = candidate.get("pid")
                current_ticks = (
                    proc_start_ticks(pid, self.proc_root) if isinstance(pid, int) else None
                )
                identities.append(
                    {
                        "role": role,
                        "pid": pid,
                        "recorded_start_ticks": candidate.get("start_ticks"),
                        "current_start_ticks": current_ticks,
                        "identity_matches": current_ticks is not None
                        and current_ticks == candidate.get("start_ticks"),
                        "command_sha256": candidate.get(
                            "command_sha256", candidate.get("observed_command_sha256")
                        ),
                    }
                )
        marker_evidence = {
            name: self._path_evidence(resolve_path(value, self.manifest_dir))
            for name, value in queue["markers"].items()
        }
        marker_evidence.update(
            {
                name: self._path_evidence(self.queue_dir(queue["id"]) / name)
                for name in (
                    "APPROVAL_REQUIRED",
                    "BLOCKED_EVIDENCE",
                    "PROMOTED",
                    "DIAGNOSTIC_EXPAND",
                    "STOPPED",
                )
            }
        )
        logs = [self.queue_dir(queue["id"]) / "launch.log"]
        logs.extend(
            resolve_path(value, self.manifest_dir)
            for value in queue.get("evidence_logs", [])
        )
        checkpoint = {
            key: self._path_evidence(resolve_path(value, self.manifest_dir))
            for key, value in queue["checkpoint"].items()
            if isinstance(value, str)
        }
        heartbeat_path = resolve_path(
            queue["preflight"]["heartbeat"]["path"], self.manifest_dir
        )
        return {
            "queue_id": queue["id"],
            "promotion_stage": queue["promotion_stage"],
            "state": state.get("state"),
            "gpu_readiness": state.get("gpu_readiness", "UNKNOWN"),
            "decision_sha256": state.get("decision_sha256"),
            "ports": queue["resources"]["ports"],
            "gpu_uuids": queue["resources"]["gpu_uuids"],
            "attempt_lineage": queue["attempt_lineage"],
            "process_identities": identities,
            "markers": marker_evidence,
            "heartbeat": self._path_evidence(heartbeat_path),
            "checkpoint": checkpoint,
            "logs": [
                {"path": str(path), "tail": tail_text(path, 16_384)} for path in logs
            ],
        }

    def _context_metrics(
        self, manifest: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        state_bytes = 0
        event_count = 0
        for queue_id in states:
            for name in ("state.json", "events.jsonl"):
                path = self.queue_dir(queue_id) / name
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                if name == "state.json":
                    state_bytes += size
                else:
                    event_count += count_nonempty_lines(path)
        branch_attempts: dict[str, int] = {}
        for queue in manifest["queues"]:
            branch = queue["attempt_lineage"]["branch_id"]
            branch_attempts[branch] = branch_attempts.get(branch, 0) + 1
        return {
            "state_bytes": state_bytes,
            "event_count": event_count,
            "branch_attempts": branch_attempts,
        }

    def _context_triggers_for(
        self,
        manifest: Mapping[str, Any],
        metrics: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
    ) -> tuple[list[str], float | None]:
        hygiene = manifest["context_hygiene"]
        now = time.time()
        triggers = set(self._context_triggers)
        previous_created = previous.get("created_at_unix") if previous else None
        if previous is None:
            triggers.add("initial_snapshot")
        elif not isinstance(previous_created, (int, float)) or (
            now - previous_created >= hygiene["interval_seconds"]
        ):
            triggers.add("periodic_60m")
        if metrics["state_bytes"] > hygiene["max_state_bytes"]:
            triggers.add("state_over_200kb")
        previous_events = previous.get("metrics", {}).get("event_count", 0) if previous else 0
        if metrics["event_count"] - previous_events > hygiene["max_events"]:
            triggers.add("events_over_50")
        if any(
            attempts >= hygiene["max_attempts_per_branch"]
            for attempts in metrics["branch_attempts"].values()
        ):
            triggers.add("branch_attempts_at_least_3")
        compaction_path = resolve_path(hygiene["compaction_marker"], self.manifest_dir)
        try:
            compaction_mtime: float | None = compaction_path.stat().st_mtime
        except OSError:
            compaction_mtime = None
        previous_compaction = previous.get("compaction_mtime") if previous else None
        if (
            compaction_mtime is not None
            and previous_compaction is not None
            and compaction_mtime > previous_compaction
        ):
            triggers.add("compaction")
        return sorted(triggers), compaction_mtime

    def _write_current_state(
        self,
        manifest: Mapping[str, Any],
        states: Mapping[str, Mapping[str, Any]],
        manifest_sha: str,
    ) -> None:
        current_path = self.output_dir / "CURRENT_STATE.json"
        try:
            previous = read_json(current_path) if current_path.is_file() else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            previous = None
        metrics = self._context_metrics(manifest, states)
        triggers, compaction_mtime = self._context_triggers_for(
            manifest, metrics, previous
        )
        if not triggers:
            return
        trigger_signature = sha256_value(
            {
                "triggers": triggers,
                "metrics": metrics,
                "states": {key: value.get("state") for key, value in states.items()},
                "compaction_mtime": compaction_mtime,
            }
        )
        if (
            previous is not None
            and previous.get("trigger_signature") == trigger_signature
            and "periodic_60m" not in triggers
        ):
            self._context_triggers.clear()
            return
        hygiene = manifest["context_hygiene"]
        goal_path = resolve_path(hygiene["goal_path"], self.manifest_dir)
        git_worktree = resolve_path(hygiene["git_worktree"], self.manifest_dir)
        gpu_path = resolve_path(hygiene["gpu_inventory_path"], self.manifest_dir)
        queue_context = {
            queue["id"]: self._queue_context(queue, states.get(queue["id"], {}))
            for queue in manifest["queues"]
        }
        created_at = utc_now()
        created_at_unix = time.time()
        snapshot_core = {
            "created_at": created_at,
            "created_at_unix": created_at_unix,
            "manifest_sha256": manifest_sha,
            "triggers": triggers,
            "trigger_signature": trigger_signature,
            "compaction_mtime": compaction_mtime,
            "metrics": metrics,
            "goal": {
                **self._path_evidence(goal_path),
                "tail": tail_text(goal_path, 32_768),
            },
            "git": {
                "worktree": str(git_worktree),
                "head": command_output(["git", "rev-parse", "HEAD"], git_worktree),
                "branch": command_output(
                    ["git", "branch", "--show-current"], git_worktree
                ),
                "status": command_output(
                    ["git", "status", "--short", "--branch"], git_worktree
                ),
            },
            "gpu_inventory": {
                **self._path_evidence(gpu_path),
                "tail": tail_text(gpu_path, 32_768),
            },
            "active_queue_ids": [
                queue_id
                for queue_id, state in states.items()
                if state.get("state") not in TERMINAL_STATES
            ],
            "queues": queue_context,
        }
        snapshot_id = f"{int(created_at_unix * 1_000_000)}-{sha256_value(snapshot_core)[:12]}"
        snapshot = {
            "schema_version": "shopping-current-state-v1",
            "snapshot_id": snapshot_id,
            **snapshot_core,
        }
        lines = [
            "# Shopping Agent CURRENT_STATE",
            "",
            f"- snapshot_id: `{snapshot_id}`",
            f"- created_at: `{created_at}`",
            f"- triggers: `{', '.join(triggers)}`",
            f"- git: `{snapshot['git']['branch']}@{snapshot['git']['head']}`",
            f"- active queues: `{', '.join(snapshot['active_queue_ids']) or 'none'}`",
            "",
            "| Queue | Stage | State | PID identities | Ports | GPU UUIDs |",
            "|---|---|---|---:|---|---|",
        ]
        for queue_id, evidence in queue_context.items():
            lines.append(
                f"| {queue_id} | {evidence['promotion_stage']} | {evidence['state']} | "
                f"{len(evidence['process_identities'])} | {evidence['ports']} | "
                f"{evidence['gpu_uuids']} |"
            )
        markdown = "\n".join(lines) + "\n"
        snapshots_dir = self.output_dir / "context-snapshots"
        atomic_write_json(snapshots_dir / f"{snapshot_id}.json", snapshot)
        atomic_write_text(snapshots_dir / f"{snapshot_id}.md", markdown)
        atomic_write_json(current_path, snapshot)
        atomic_write_text(self.output_dir / "CURRENT_STATE.md", markdown)

        index_path = self.output_dir / "SUPERSEDED_INDEX.json"
        try:
            index = read_json(index_path) if index_path.is_file() else {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            index = {}
        entries = index.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        if previous is not None and previous.get("snapshot_id") != snapshot_id:
            entries.append(
                {
                    "snapshot_id": previous.get("snapshot_id"),
                    "superseded_by": snapshot_id,
                    "reason": triggers,
                    "superseded_at": created_at,
                }
            )
        atomic_write_json(
            index_path,
            {
                "schema_version": "shopping-superseded-index-v1",
                "current_snapshot_id": snapshot_id,
                "entries": entries,
            },
        )
        self._context_triggers.clear()

    def run_once(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / "controller.lock"
        with lock_path.open("a+") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "LOCKED", "queues": {}}
            manifest = self.load_manifest()
            manifest_sha = sha256_file(self.manifest_path)
            hygiene_sha = sha256_value(manifest["context_hygiene"])
            current_states: dict[str, dict[str, Any]] = {}
            for queue in manifest["queues"]:
                try:
                    current_states[queue["id"]] = self._load_state(
                        queue, manifest_sha, hygiene_sha
                    )
                except (OSError, TypeError, ValueError, ManifestError) as error:
                    self._safe_block(
                        queue["id"],
                        "BLOCKED_EVIDENCE",
                        "state_load_failed",
                        error=f"{type(error).__name__}: {error}",
                    )
            for queue in manifest["queues"]:
                try:
                    state = self.process_queue(
                        queue, current_states, manifest_sha, hygiene_sha
                    )
                    current_states[queue["id"]] = state
                except PermissionError as error:
                    self._safe_block(
                        queue["id"],
                        "APPROVAL_REQUIRED",
                        "permission_denied",
                        error=f"{type(error).__name__}: {error}",
                    )
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    ManifestError,
                    subprocess.SubprocessError,
                ) as error:
                    self._safe_block(
                        queue["id"],
                        "BLOCKED_EVIDENCE",
                        "queue_processing_failed",
                        error=f"{type(error).__name__}: {error}",
                    )
            readiness: dict[str, str] = {}
            for queue in manifest["queues"]:
                state = current_states.get(queue["id"], {})
                value = self._gpu_readiness(queue, state, current_states)
                readiness[queue["id"]] = value
                if isinstance(state, dict):
                    state["gpu_readiness"] = value
            summary = {
                "schema_version": STATE_SCHEMA_VERSION,
                "timestamp": utc_now(),
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": manifest_sha,
                "execute_enabled": self.execute,
                "queues": {
                    queue_id: state.get("state", "UNKNOWN")
                    for queue_id, state in current_states.items()
                },
                "gpu_ready": {
                    "READY_NOW": sorted(
                        queue_id
                        for queue_id, value in readiness.items()
                        if value == "READY_NOW"
                    ),
                    "READY_AFTER_CURRENT": sorted(
                        queue_id
                        for queue_id, value in readiness.items()
                        if value == "READY_AFTER_CURRENT"
                    ),
                },
            }
            atomic_write_json(self.output_dir / "summary.json", summary)
            append_jsonl(self.output_dir / "controller-events.jsonl", summary)
            try:
                self._write_current_state(manifest, current_states, manifest_sha)
            except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
                append_jsonl(
                    self.output_dir / "controller-errors.jsonl",
                    {
                        "timestamp": utc_now(),
                        "reason": "context_snapshot_failed",
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hash_parser = subparsers.add_parser("hash-command", help="print a queue command SHA-256")
    hash_parser.add_argument("--manifest", type=Path, required=True)
    hash_parser.add_argument("--queue-id", required=True)
    run_parser = subparsers.add_parser("run", help="run one tick or a persistent controller loop")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--execute", action="store_true", help="enable manifest-authorized launches"
    )
    run_parser.add_argument("--once", action="store_true")
    run_parser.add_argument("--interval-seconds", type=int, default=60)
    run_parser.add_argument("--stop-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "hash-command":
        manifest = read_json(args.manifest)
        queues = manifest.get("queues", [])
        queue = next((item for item in queues if item.get("id") == args.queue_id), None)
        if not isinstance(queue, dict) or not isinstance(queue.get("launch"), dict):
            raise SystemExit(f"queue not found or malformed: {args.queue_id}")
        print(command_sha256(queue["launch"]))
        return 0
    if args.interval_seconds < 1:
        raise SystemExit("--interval-seconds must be positive")
    controller = PromotionController(args.manifest, args.output_dir, execute=args.execute)
    while True:
        controller.run_once()
        if args.once or (args.stop_file is not None and args.stop_file.exists()):
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
