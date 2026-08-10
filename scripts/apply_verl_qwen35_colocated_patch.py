#!/usr/bin/env python3
"""Apply the pinned veRL 0.8 Qwen3.5 colocated LoRA compatibility patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
BACKUP_SUFFIX = ".shopping-grpo-qwen35-colocated.orig"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = PROJECT_ROOT / "patches/verl-0.8.0-qwen35-colocated-lora.patch"

TARGETS = {
    Path("verl/utils/vllm/utils.py"): (
        "8e2577fd0c1e9f3c44c91256573fc13cf0657c5f6ef715c0bbdbd94da812ee4c",
        "df1b0c39cada6c22cfbb409c6febd7c7299ec326651d155e19e8f5b5ca04725e",
        "expand_grouped_merged_lora_slices",
    ),
    Path("verl/utils/fsdp_utils.py"): (
        "b3e05f870dc86fc9e4e6482c76a37994f743f0b6e5a905bcc322089034f98a17",
        "9e7f64506324094e48c97dc244895b4e7c41850c69a40ce3a85f6bff8f453361",
        '"conv1d"]',
    ),
    Path("verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"): (
        "83cb67d6911206d6456e8581828882ff622a254b6ba78ad82b9802d876083c86",
        "f7373720fd15e07d2e95a98bc3d535be00636a52be183c82ee299bb9e0193ed0",
        "Malformed CUDA IPC reduction",
    ),
    Path("verl/workers/engine/fsdp/transformer_impl.py"): (
        "b4f6243471f22c08dbfa472d13f708e8de609705ecff13f23713258160cf7902",
        "595739a91de2ca51f8de9c1b186afc824c067ac5e531b30f92731121bfd31ee5",
        "no_padding_2_padding contract consumes one flat value",
    ),
    Path("verl/models/transformers/qwen3_5.py"): (
        "088a773b93c8b561248a61501da5dd37124d534d9826fe1b67507b123b4d97c1",
        "a426919cae1fd3309c55e4316678c330636a878703f507761d9d9497b8e935a2",
        "calculate_entropy=calculate_entropy",
    ),
    Path("verl/utils/experimental/torch_functional.py"): (
        "09f93ac5f28784d85f7f16e7160e697be256de67ba36f8a36a3e5e4bcac24a8f",
        "b639a65551ecc8200afeb4c50c5754063715143992130c03de1c7f958c87e554",
        "probs.scatter_add_",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_verl_root(override: Path | None) -> Path:
    if override is not None:
        root = override.expanduser().resolve()
    else:
        installed_version = importlib.metadata.version("verl")
        if installed_version != EXPECTED_VERL_VERSION:
            raise RuntimeError(
                f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
            )
        import verl

        package = Path(verl.__file__).resolve().parent
        if not package.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(
                "verl.__file__ is not from the active Python environment: "
                f"source={package}, prefix={Path(sys.prefix).resolve()}"
            )
        root = package.parent
    if not (root / "verl/__init__.py").is_file():
        raise RuntimeError(f"veRL root must contain verl/__init__.py: {root}")
    return root


def target_state(root: Path) -> dict[Path, str]:
    states = {}
    for relative, (original_hash, patched_hash, _) in TARGETS.items():
        target = root / relative
        if not target.is_file():
            raise RuntimeError(f"veRL patch target does not exist: {target}")
        current_hash = sha256(target)
        if current_hash == original_hash:
            states[relative] = "original"
        elif current_hash == patched_hash:
            states[relative] = "patched"
        else:
            raise RuntimeError(
                f"refusing unknown veRL source {target}: got SHA256 {current_hash}"
            )
    return states


def verify_patched(root: Path) -> None:
    for relative, (_, patched_hash, marker) in TARGETS.items():
        target = root / relative
        current_hash = sha256(target)
        if current_hash != patched_hash:
            raise RuntimeError(
                f"patched SHA256 mismatch for {target}: expected {patched_hash}, "
                f"got {current_hash}"
            )
        if marker not in target.read_text(encoding="utf-8"):
            raise RuntimeError(f"patched source is missing marker {marker!r}: {target}")
        py_compile.compile(str(target), doraise=True)


def build_patched_tree(root: Path, destination: Path, states: dict[Path, str]) -> None:
    for relative, (original_hash, _, _) in TARGETS.items():
        target = root / relative
        backup = Path(str(target) + BACKUP_SUFFIX)
        source = target
        if states[relative] == "patched":
            if not backup.is_file() or sha256(backup) != original_hash:
                raise RuntimeError(f"verified original backup is required: {backup}")
            source = backup
        staged = destination / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staged)

    patch_program = shutil.which("patch")
    if patch_program is None:
        raise RuntimeError("required system 'patch' executable is unavailable")
    subprocess.run(
        [
            patch_program,
            "--batch",
            "--forward",
            "--silent",
            "-p1",
            "-d",
            str(destination),
            "-i",
            str(PATCH_FILE),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    verify_patched(destination)


def apply_patch(root: Path) -> None:
    states = target_state(root)
    if all(state == "patched" for state in states.values()):
        verify_patched(root)
        print(f"veRL Qwen3.5 colocated patch already applied: {root}")
        return
    if not PATCH_FILE.is_file():
        raise RuntimeError(f"patch file is missing: {PATCH_FILE}")

    with tempfile.TemporaryDirectory(prefix="verl-qwen35-patch-") as temp_dir:
        temporary = Path(temp_dir)
        staged_root = temporary / "staged"
        rollback_root = temporary / "rollback"
        build_patched_tree(root, staged_root, states)

        for relative in TARGETS:
            rollback = rollback_root / relative
            rollback.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, rollback)

        try:
            for relative, (original_hash, _, _) in TARGETS.items():
                target = root / relative
                backup = Path(str(target) + BACKUP_SUFFIX)
                if backup.exists() and sha256(backup) != original_hash:
                    raise RuntimeError(f"refusing invalid original backup: {backup}")
                if not backup.exists():
                    source = target if states[relative] == "original" else rollback_root / relative
                    if sha256(source) != original_hash:
                        raise RuntimeError(f"cannot create verified original backup for {target}")
                    shutil.copy2(source, backup)
                shutil.copy2(staged_root / relative, target)
            verify_patched(root)
        except Exception:
            for relative in TARGETS:
                shutil.copy2(rollback_root / relative, root / relative)
            raise

    print(f"applied veRL Qwen3.5 colocated patch: {root}")


def restore_patch(root: Path) -> None:
    states = target_state(root)
    for relative, (original_hash, _, _) in TARGETS.items():
        if states[relative] == "original":
            continue
        target = root / relative
        backup = Path(str(target) + BACKUP_SUFFIX)
        if not backup.is_file() or sha256(backup) != original_hash:
            raise RuntimeError(f"cannot restore without verified original backup: {backup}")
        shutil.copy2(backup, target)
        py_compile.compile(str(target), doraise=True)
    print(f"restored original veRL Qwen3.5 colocated sources: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--restore", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check and args.restore:
        raise SystemExit("--check and --restore are mutually exclusive")
    try:
        root = resolve_verl_root(args.verl_root)
        if args.check:
            verify_patched(root)
            print(f"verified veRL Qwen3.5 colocated patch: {root}")
        elif args.restore:
            restore_patch(root)
        else:
            apply_patch(root)
    except (OSError, RuntimeError, subprocess.CalledProcessError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL Qwen3.5 colocated patch error: {exc}") from exc


if __name__ == "__main__":
    main()
