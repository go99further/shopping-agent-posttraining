"""CPU-only tests for the deterministic unattended promotion controller."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import unattended_promotion_controller as controller


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def queue_spec(
    root: Path,
    queue_id: str,
    *,
    allow_execute: bool = False,
    promotion_stage: str = "origin",
) -> dict:
    (root / "checkpoints/parent").mkdir(parents=True, exist_ok=True)
    launch = {
        "argv": ["/bin/echo", queue_id],
        "cwd": str(root),
        "env": {"ATTEMPT_ID": queue_id},
        "owned_process_receipt": str(root / "runs" / queue_id / "owned-pids/supervisor.pid"),
        "allow_execute": allow_execute,
        "command_sha256": "0" * 64,
    }
    launch["command_sha256"] = controller.command_sha256(launch)
    return {
        "id": queue_id,
        "initial_state": "READY",
        "promotion_stage": promotion_stage,
        "prerequisites": [],
        "resource_class": "CPU_ONLY",
        "workload_kind": "cpu_research",
        "requires": [{"type": "executable", "name": "echo"}],
        "launch": launch,
        "markers": {
            "complete": str(root / "runs" / queue_id / "COMPLETE"),
            "failed": str(root / "runs" / queue_id / "FAILED"),
        },
        "resources": {"ports": [18080], "gpu_uuids": ["GPU-test-uuid"]},
        "checkpoint": {
            "input": str(root / "checkpoints" / "parent"),
            "expected_output": str(root / "checkpoints" / queue_id),
        },
        "attempt_lineage": {
            "attempt_id": queue_id,
            "branch_id": queue_id.split("-attempt", 1)[0],
            "parent_attempt_id": "parent-attempt",
            "parent_checkpoint": str(root / "checkpoints" / "parent"),
        },
        "preflight": {
            "authentication": {"type": "none"},
            "directories": [{"path": str(root), "access": "rwx"}],
            "disk": {"path": str(root), "min_free_bytes": 0},
            "ports": [{"host": "127.0.0.1", "port": 18080}],
            "checkpoint_resume": {"mode": "fresh"},
            "heartbeat": {
                "path": str(root / "runs" / queue_id / "heartbeat"),
                "stale_seconds": 120,
                "startup_grace_seconds": 120,
            },
            "http_probes": [],
        },
        "analysis": {
            "conditions": [
                {
                    "type": "json_value",
                    "path": str(root / "runs" / queue_id / "metrics.json"),
                    "pointer": "/strict_success_rate",
                    "op": "gte",
                    "expected": 0.5,
                },
                {
                    "type": "json_value",
                    "path": str(root / "runs" / queue_id / "metrics.json"),
                    "pointer": "/wrong_purchase_rate",
                    "op": "lte",
                    "expected": 0.0,
                },
            ]
        },
    }


def manifest_for(*queues: dict) -> dict:
    queue_list = list(queues)
    by_id = {queue["id"]: queue for queue in queue_list}
    for queue in queue_list:
        queue["prerequisite_decision_sha256"] = {
            dependency: controller.sha256_value(controller.decision_spec(by_id[dependency]))
            for dependency in queue.get("prerequisites", [])
        }
    return {
        "schema_version": controller.SCHEMA_VERSION,
        "context_hygiene": {
            "goal_path": str(Path(__file__).resolve()),
            "git_worktree": str(Path(__file__).resolve().parents[1]),
            "gpu_inventory_path": str(Path(__file__).resolve()),
            "interval_seconds": 3600,
            "max_state_bytes": 204800,
            "max_events": 50,
            "max_attempts_per_branch": 3,
            "compaction_marker": str(Path(__file__).resolve()),
        },
        "queues": queue_list,
    }


def make_runner(
    root: Path, manifest_path: Path, *, execute: bool
) -> controller.PromotionController:
    proc_root = root / "proc"
    if execute:
        (proc_root / "self").mkdir(parents=True, exist_ok=True)
        (proc_root / "self/stat").write_text("controller stat\n", encoding="utf-8")
    return controller.PromotionController(
        manifest_path, root / "audit", execute=execute, proc_root=proc_root
    )


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class UnattendedPromotionControllerTest(unittest.TestCase):
    def test_default_is_dry_run_and_records_exact_hash_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "dry-run", allow_execute=True)
            write_json(manifest_path, manifest_for(queue))

            with mock.patch.object(controller, "launch_detached") as popen:
                summary = controller.PromotionController(
                    manifest_path, root / "audit", execute=False
                ).run_once()

            popen.assert_not_called()
            self.assertEqual(summary["queues"], {"dry-run": "READY"})
            receipt = json.loads(
                (root / "audit/queues/dry-run/DRY_RUN").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["command_sha256"], queue["launch"]["command_sha256"])
            self.assertFalse(receipt["controller_execute_enabled"])

    def test_authorization_toggle_is_only_manifest_change_allowed_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "toggle", allow_execute=False)
            write_json(manifest_path, manifest_for(queue))
            runner = make_runner(root, manifest_path, execute=True)
            runner.run_once()

            queue["launch"]["allow_execute"] = True
            write_json(manifest_path, manifest_for(queue))
            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(123)
            ), mock.patch.object(controller, "proc_start_ticks", return_value=777):
                summary = runner.run_once()

            self.assertEqual(summary["queues"]["toggle"], "RUNNING")
            self.assertFalse((root / "audit/queues/toggle/BLOCKED_EVIDENCE").exists())

    def test_success_path_records_lineage_and_requires_all_registered_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "success", allow_execute=True)
            write_json(manifest_path, manifest_for(queue))
            runner = make_runner(root, manifest_path, execute=True)

            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(321)
            ), mock.patch.object(controller, "proc_start_ticks", return_value=888):
                self.assertEqual(runner.run_once()["queues"]["success"], "RUNNING")

            receipt = json.loads(
                (root / "audit/queues/success/launch-receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["pid"], 321)
            self.assertEqual(receipt["start_ticks"], 888)
            self.assertEqual(receipt["command_sha256"], queue["launch"]["command_sha256"])
            self.assertEqual(receipt["ports"], [18080])
            self.assertEqual(receipt["gpu_uuids"], ["GPU-test-uuid"])
            self.assertEqual(receipt["attempt_lineage"]["attempt_id"], "success")
            self.assertIn("expected_output", receipt["checkpoint"])

            run_dir = root / "runs/success"
            write_json(
                run_dir / "metrics.json",
                {"strict_success_rate": 0.75, "wrong_purchase_rate": 0.0},
            )
            (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
            self.assertEqual(runner.run_once()["queues"]["success"], "COMPLETE")
            self.assertEqual(runner.run_once()["queues"]["success"], "ANALYZED")
            self.assertEqual(runner.run_once()["queues"]["success"], "PROMOTED")
            analysis = json.loads(
                (root / "audit/queues/success/analysis.json").read_text(encoding="utf-8")
            )
            self.assertTrue(analysis["all_passed"])

            with mock.patch.object(controller, "launch_detached") as popen:
                self.assertEqual(runner.run_once()["queues"]["success"], "PROMOTED")
            popen.assert_not_called()

    def test_failed_or_regressed_attempt_is_analyzed_then_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "regression", allow_execute=True)
            write_json(manifest_path, manifest_for(queue))
            runner = make_runner(root, manifest_path, execute=True)
            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(444)
            ), mock.patch.object(controller, "proc_start_ticks", return_value=999):
                runner.run_once()

            run_dir = root / "runs/regression"
            write_json(
                run_dir / "metrics.json",
                {"strict_success_rate": 0.8, "wrong_purchase_rate": 0.2},
            )
            (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
            runner.run_once()
            runner.run_once()
            summary = runner.run_once()

            self.assertEqual(summary["queues"]["regression"], "STOPPED")
            self.assertTrue((root / "audit/queues/regression/STOPPED").is_file())
            self.assertFalse((root / "audit/queues/regression/PROMOTED").exists())

    def test_hash_mismatch_blocks_one_queue_but_independent_queue_still_launches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            bad = queue_spec(root, "bad-hash", allow_execute=True)
            bad["launch"]["command_sha256"] = "f" * 64
            good = queue_spec(root, "independent", allow_execute=True)
            write_json(manifest_path, manifest_for(bad, good))
            runner = make_runner(root, manifest_path, execute=True)

            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(555)
            ) as popen, mock.patch.object(controller, "proc_start_ticks", return_value=1001):
                summary = runner.run_once()

            self.assertEqual(summary["queues"]["bad-hash"], "READY")
            self.assertEqual(summary["queues"]["independent"], "RUNNING")
            self.assertTrue((root / "audit/queues/bad-hash/BLOCKED_EVIDENCE").is_file())
            self.assertEqual(popen.call_count, 1)

    def test_permission_marker_does_not_stall_other_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            first = queue_spec(root, "permission", allow_execute=True)
            second = queue_spec(root, "next", allow_execute=True)
            write_json(manifest_path, manifest_for(first, second))
            runner = make_runner(root, manifest_path, execute=True)
            requirements = [
                ("APPROVAL_REQUIRED", ["path_permission_x:/protected"]),
                (None, []),
            ]
            with mock.patch.object(
                runner, "_requirements", side_effect=requirements
            ), mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(556)
            ) as popen, mock.patch.object(controller, "proc_start_ticks", return_value=1002):
                summary = runner.run_once()

            self.assertEqual(summary["queues"]["permission"], "READY")
            self.assertEqual(summary["queues"]["next"], "RUNNING")
            self.assertTrue((root / "audit/queues/permission/APPROVAL_REQUIRED").is_file())
            self.assertEqual(popen.call_count, 1)

    def test_restart_recovers_running_attempt_without_relaunch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "recover", allow_execute=True)
            write_json(manifest_path, manifest_for(queue))
            first = make_runner(root, manifest_path, execute=True)
            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(600)
            ), mock.patch.object(controller, "proc_start_ticks", return_value=2020):
                first.run_once()

            owned_receipt = Path(queue["launch"]["owned_process_receipt"])
            owned_receipt.parent.mkdir(parents=True, exist_ok=True)
            owned_receipt.write_text(
                "pid=999\nstart_ticks=3030\nrole=supervisor\n"
                "command_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
                encoding="utf-8",
            )

            restarted = make_runner(root, manifest_path, execute=True)
            with mock.patch.object(controller, "launch_detached") as popen, mock.patch.object(
                controller,
                "proc_start_ticks",
                side_effect=lambda pid, _root: 3030 if pid == 999 else None,
            ):
                summary = restarted.run_once()
            popen.assert_not_called()
            self.assertEqual(summary["queues"]["recover"], "RUNNING")
            launch_receipt = json.loads(
                (root / "audit/queues/recover/launch-receipt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(launch_receipt["owned_process"]["pid"], 999)
            self.assertEqual(
                launch_receipt["owned_process"]["observed_command_sha256"], "a" * 64
            )

            run_dir = root / "runs/recover"
            write_json(
                run_dir / "metrics.json",
                {"strict_success_rate": 0.5, "wrong_purchase_rate": 0.0},
            )
            (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
            self.assertEqual(restarted.run_once()["queues"]["recover"], "COMPLETE")

    def test_stale_heartbeat_freezes_only_running_branch_without_signalling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            frozen = queue_spec(root, "heartbeat-frozen", allow_execute=True)
            independent = queue_spec(root, "heartbeat-independent", allow_execute=False)
            frozen["preflight"]["heartbeat"]["startup_grace_seconds"] = 1
            write_json(manifest_path, manifest_for(frozen, independent))
            runner = make_runner(root, manifest_path, execute=True)
            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(650)
            ), mock.patch.object(controller, "proc_start_ticks", return_value=2526):
                runner.run_once()
            state_path = root / "audit/queues/heartbeat-frozen/state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["launch_receipt"]["launched_at_unix"] = 0
            write_json(state_path, state)

            with mock.patch.object(controller, "launch_detached") as popen, mock.patch.object(
                controller, "proc_start_ticks", return_value=2526
            ):
                summary = runner.run_once()

            popen.assert_not_called()
            self.assertEqual(summary["queues"]["heartbeat-frozen"], "RUNNING")
            self.assertEqual(summary["queues"]["heartbeat-independent"], "READY")
            marker = json.loads(
                (root / "audit/queues/heartbeat-frozen/BLOCKED_EVIDENCE").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(marker["reason"], "heartbeat_missing_branch_frozen")

    def test_http_502_freezes_one_ready_branch_and_checks_independent_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            unavailable = queue_spec(root, "http-502", allow_execute=True)
            unavailable["preflight"]["http_probes"] = [
                {
                    "url": "http://127.0.0.1:19999/health",
                    "expected_status": [200],
                    "timeout_seconds": 1,
                }
            ]
            independent = queue_spec(root, "http-independent", allow_execute=True)
            independent["resources"]["ports"] = [18081]
            independent["preflight"]["ports"] = [
                {"host": "127.0.0.1", "port": 18081}
            ]
            write_json(manifest_path, manifest_for(unavailable, independent))
            runner = make_runner(root, manifest_path, execute=True)

            with mock.patch.object(controller, "http_status", return_value=502), mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(651)
            ) as popen, mock.patch.object(controller, "proc_start_ticks", return_value=2527):
                summary = runner.run_once()

            self.assertEqual(summary["queues"]["http-502"], "READY")
            self.assertEqual(summary["queues"]["http-independent"], "RUNNING")
            self.assertEqual(popen.call_count, 1)
            marker = json.loads(
                (root / "audit/queues/http-502/BLOCKED_EVIDENCE").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("http_probe_status", " ".join(marker["evidence"]))

    def test_decision_gate_mutation_is_blocked_after_initial_dry_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "immutable", allow_execute=False)
            write_json(manifest_path, manifest_for(queue))
            runner = make_runner(root, manifest_path, execute=True)
            runner.run_once()

            queue["analysis"]["conditions"][0]["expected"] = 0.1
            write_json(manifest_path, manifest_for(queue))
            with mock.patch.object(controller, "launch_detached") as popen:
                summary = runner.run_once()

            popen.assert_not_called()
            self.assertEqual(summary["queues"]["immutable"], "READY")
            marker = json.loads(
                (root / "audit/queues/immutable/BLOCKED_EVIDENCE").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["reason"], "decision_bearing_manifest_fields_changed")

    def test_promoted_prerequisite_unlocks_downstream_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            parent = queue_spec(root, "parent", allow_execute=True)
            child = queue_spec(
                root, "child", allow_execute=True, promotion_stage="held_out"
            )
            child["prerequisites"] = ["parent"]
            write_json(manifest_path, manifest_for(parent, child))
            runner = make_runner(root, manifest_path, execute=True)
            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(701)
            ) as popen, mock.patch.object(controller, "proc_start_ticks", return_value=3030):
                summary = runner.run_once()
            self.assertEqual(summary["queues"]["parent"], "RUNNING")
            self.assertEqual(summary["queues"]["child"], "READY")
            self.assertEqual(popen.call_count, 1)

    def test_manifest_rejects_skipping_origin_to_small_dev(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = queue_spec(root, "origin")
            small_dev = queue_spec(root, "small-dev", promotion_stage="small_dev")
            small_dev["prerequisites"] = ["origin"]

            with self.assertRaisesRegex(
                controller.ManifestError, "must depend on one of the previous stages"
            ):
                controller.validate_manifest(manifest_for(origin, small_dev))

    def test_soft_held_out_failure_unlocks_one_bounded_diagnostic_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            origin = queue_spec(root, "origin-attempt-01", allow_execute=True)
            held_out = queue_spec(
                root,
                "held-out-attempt-01",
                allow_execute=True,
                promotion_stage="held_out",
            )
            held_out["prerequisites"] = [origin["id"]]
            diagnostic = queue_spec(
                root,
                "diagnostic-attempt-01",
                allow_execute=True,
                promotion_stage="diagnostic_expand",
            )
            diagnostic["prerequisites"] = [held_out["id"]]
            diagnostic["prerequisite_states"] = {
                held_out["id"]: "DIAGNOSTIC_EXPAND"
            }
            held_out["analysis"]["conditions"][0]["on_failure"] = "DIAGNOSTIC_EXPAND"
            held_out["diagnostic_expansion"] = {
                "queue_id": diagnostic["id"],
                "max_gpu_hours": 0.5,
                "max_tasks": 20,
            }
            write_json(manifest_path, manifest_for(origin, held_out, diagnostic))
            runner = make_runner(root, manifest_path, execute=True)

            fake_processes = [FakeProcess(801), FakeProcess(802), FakeProcess(803)]
            with mock.patch.object(
                controller, "launch_detached", side_effect=fake_processes
            ), mock.patch.object(controller, "proc_start_ticks", return_value=4040):
                runner.run_once()
                origin_run = root / "runs" / origin["id"]
                write_json(
                    origin_run / "metrics.json",
                    {"strict_success_rate": 0.75, "wrong_purchase_rate": 0.0},
                )
                (origin_run / "COMPLETE").write_text("complete\n", encoding="utf-8")
                runner.run_once()
                runner.run_once()
                promoted = runner.run_once()
                self.assertEqual(promoted["queues"][origin["id"]], "PROMOTED")
                self.assertEqual(promoted["queues"][held_out["id"]], "RUNNING")

                held_run = root / "runs" / held_out["id"]
                write_json(
                    held_run / "metrics.json",
                    {"strict_success_rate": 0.0, "wrong_purchase_rate": 0.0},
                )
                (held_run / "COMPLETE").write_text("complete\n", encoding="utf-8")
                runner.run_once()
                runner.run_once()
                expanded = runner.run_once()

            self.assertEqual(
                expanded["queues"][held_out["id"]], "DIAGNOSTIC_EXPAND"
            )
            self.assertEqual(expanded["queues"][diagnostic["id"]], "RUNNING")
            marker = json.loads(
                (
                    root
                    / f"audit/queues/{held_out['id']}/DIAGNOSTIC_EXPAND"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(marker["max_gpu_hours"], 0.5)
            self.assertEqual(marker["max_tasks"], 20)

    def test_protocol_invariant_cannot_be_softened_to_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = queue_spec(root, "invariant")
            condition = queue["analysis"]["conditions"][0]
            condition["protocol_invariant"] = True
            condition["on_failure"] = "DIAGNOSTIC_EXPAND"

            with self.assertRaisesRegex(controller.ManifestError, "must hard STOP"):
                controller.validate_manifest(manifest_for(queue))

    def test_current_state_is_atomic_and_prior_snapshot_is_superseded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            queue = queue_spec(root, "context", allow_execute=True)
            write_json(manifest_path, manifest_for(queue))
            runner = make_runner(root, manifest_path, execute=True)
            with mock.patch.object(
                controller, "launch_detached", return_value=FakeProcess(901)
            ), mock.patch.object(controller, "proc_start_ticks", return_value=5050):
                runner.run_once()
                first = json.loads(
                    (root / "audit/CURRENT_STATE.json").read_text(encoding="utf-8")
                )
                run_dir = root / "runs/context"
                write_json(
                    run_dir / "metrics.json",
                    {"strict_success_rate": 1.0, "wrong_purchase_rate": 0.0},
                )
                (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
                runner.run_once()
                second = json.loads(
                    (root / "audit/CURRENT_STATE.json").read_text(encoding="utf-8")
                )

            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
            index = json.loads(
                (root / "audit/SUPERSEDED_INDEX.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["current_snapshot_id"], second["snapshot_id"])
            self.assertEqual(index["entries"][-1]["superseded_by"], second["snapshot_id"])
            self.assertFalse(list((root / "audit").rglob("*.tmp")))

    def test_stale_prerequisite_decision_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = queue_spec(root, "fresh-parent")
            child = queue_spec(root, "stale-child", promotion_stage="held_out")
            child["prerequisites"] = [parent["id"]]
            manifest = manifest_for(parent, child)
            child["prerequisite_decision_sha256"][parent["id"]] = "f" * 64

            with self.assertRaisesRegex(
                controller.ManifestError, "stale prerequisite reference"
            ):
                controller.validate_manifest(manifest)

    def test_gpu_ready_two_level_queue_excludes_cpu_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            gpu_now = queue_spec(root, "gpu-now", allow_execute=True)
            gpu_now["resource_class"] = "GPU_ELIGIBLE"
            gpu_now["workload_kind"] = "logprob_forward_parity"
            cpu = queue_spec(root, "cpu-research", allow_execute=False)
            gpu_after = queue_spec(
                root, "gpu-after", allow_execute=True, promotion_stage="held_out"
            )
            gpu_after["resource_class"] = "GPU_ELIGIBLE"
            gpu_after["workload_kind"] = "inference_ab"
            gpu_after["prerequisites"] = [gpu_now["id"]]
            write_json(manifest_path, manifest_for(gpu_now, cpu, gpu_after))
            runner = make_runner(root, manifest_path, execute=False)

            summary = runner.run_once()

            self.assertEqual(summary["gpu_ready"]["READY_NOW"], ["gpu-now"])
            self.assertEqual(
                summary["gpu_ready"]["READY_AFTER_CURRENT"], ["gpu-after"]
            )
            gpu_ready_ids = {
                queue_id
                for queue_ids in summary["gpu_ready"].values()
                for queue_id in queue_ids
            }
            self.assertNotIn("cpu-research", gpu_ready_ids)


if __name__ == "__main__":
    unittest.main()
