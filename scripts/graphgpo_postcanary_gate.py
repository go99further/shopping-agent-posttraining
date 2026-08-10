#!/usr/bin/env python3
"""Audit credit-ablation artifacts and prepare a paired held-out promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.postcanary import (
    FROZEN_DEV50_SHA256,
    GateError,
    audit_fsdp_export,
    audit_merged_model,
    audit_training_checkpoint,
    compare_heldout_slice,
    prepare_frozen_slice,
    write_json_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint", help="audit a completed canary")
    checkpoint.add_argument("--run-dir", type=Path, required=True)
    checkpoint.add_argument("--supervisor-log", type=Path, required=True)
    checkpoint.add_argument("--expected-step", type=int, required=True)
    checkpoint.add_argument(
        "--expected-method",
        choices=("grpo", "shopping_gigpo", "shopping_graphgpo"),
        default="shopping_graphgpo",
    )
    checkpoint.add_argument("--expected-train-commit")
    checkpoint.add_argument("--skip-model-hash", action="store_true")
    checkpoint.add_argument("--output", type=Path, required=True)

    exported = subparsers.add_parser("export", help="audit a veRL FSDP export")
    exported.add_argument("--export-dir", type=Path, required=True)
    exported.add_argument("--output", type=Path, required=True)

    merged = subparsers.add_parser("merged", help="audit a standalone PEFT merge")
    merged.add_argument("--base-model", type=Path, required=True)
    merged.add_argument("--adapter", type=Path, required=True)
    merged.add_argument("--merged-model", type=Path, required=True)
    merged.add_argument("--output", type=Path, required=True)

    heldout = subparsers.add_parser("prepare-heldout", help="freeze a paired diagnostic slice")
    heldout.add_argument("--benchmark", type=Path, required=True)
    heldout.add_argument("--sft-trajectories", type=Path, required=True)
    heldout.add_argument("--output-dir", type=Path, required=True)
    heldout.add_argument("--training-data", type=Path)
    heldout.add_argument("--count", type=int, default=20)
    heldout.add_argument("--salt", default="graphgpo-dev-heldout-20260809")
    heldout.add_argument("--benchmark-sha256", default=FROZEN_DEV50_SHA256)
    heldout.add_argument(
        "--exclude-task-id",
        type=int,
        action="append",
        default=[],
        help="exclude tasks already observed by training-time validation",
    )

    compare = subparsers.add_parser("compare", help="compare GraphGPO with paired SFT rows")
    compare.add_argument("--sft-trajectories", type=Path, required=True)
    compare.add_argument("--candidate-trajectories", type=Path, required=True)
    compare.add_argument("--candidate-label", default="graphgpo")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--max-strict-regression", type=int, default=0)
    compare.add_argument("--min-reward-delta", type=float, default=-0.02)
    compare.add_argument("--require-promotion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "checkpoint":
            receipt = audit_training_checkpoint(
                args.run_dir,
                args.supervisor_log,
                expected_step=args.expected_step,
                expected_method=args.expected_method,
                expected_train_commit=args.expected_train_commit,
                hash_model_shards=not args.skip_model_hash,
            )
            write_json_receipt(receipt, args.output)
        elif args.command == "export":
            receipt = audit_fsdp_export(args.export_dir)
            write_json_receipt(receipt, args.output)
        elif args.command == "merged":
            receipt = audit_merged_model(args.base_model, args.adapter, args.merged_model)
            write_json_receipt(receipt, args.output)
        elif args.command == "prepare-heldout":
            receipt = prepare_frozen_slice(
                args.benchmark,
                args.sft_trajectories,
                args.output_dir,
                count=args.count,
                salt=args.salt,
                expected_benchmark_sha256=args.benchmark_sha256 or None,
                training_data=args.training_data,
                excluded_task_ids=args.exclude_task_id,
            )
        else:
            receipt = compare_heldout_slice(
                args.sft_trajectories,
                args.candidate_trajectories,
                args.output,
                candidate_label=args.candidate_label,
                max_strict_regression=args.max_strict_regression,
                min_reward_delta=args.min_reward_delta,
            )
            if args.require_promotion and not receipt["gate"]["promotion_pass"]:
                print(json.dumps(receipt["gate"], ensure_ascii=False, sort_keys=True))
                raise SystemExit(3)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    except GateError as exc:
        failure = {
            "schema_version": "shopping-credit-postcanary-gate-v1",
            "gate": args.command,
            "passed": False,
            "error": str(exc),
        }
        output = getattr(args, "output", None)
        if output is not None:
            write_json_receipt(failure, output)
        print(json.dumps(failure, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
