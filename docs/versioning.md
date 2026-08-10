# Code and Experiment Lineage

This project uses Git to identify code. Labels such as `canary-01`, `final-200`,
or `attempt-02` describe executions only; they are not software versions.

## Branches

- `main` and the upstream baseline branches remain reproducible references.
- `feat/<method>` contains one algorithmic hypothesis, such as GraphGPO-lite.
- `fix/<scope>` contains a narrowly scoped correction to a committed parent.
- Performance work belongs on the method branch unless it changes only shared
  infrastructure, in which case it uses `perf/<scope>`.

Each mergeable change has a commit with a semantic message and tests. A code
change is never made directly inside a running environment after GPU admission.
The committed patch is applied to a fresh environment and its installed-file
hash is recorded in the run receipt.

## Runs

Every training or evaluation run creates a new ignored directory and records:

```text
run_id
git_commit
git_dirty=false
config_sha256
train_data_sha256
base_model_or_adapter_sha256
environment_freeze_sha256
launcher_sha256
```

The launcher rejects a dirty worktree by default. A run can be resumed only
from a checkpoint whose recorded commit, configuration, data, and model lineage
match the new launcher. Otherwise it is a fresh attempt.

## Promotion

`CPU gate -> 1-step -> 5-step -> resume -> Dev -> Final-200` promotes an
experiment, not a branch. Each gate result links to a specific Git commit.
Failed gates are preserved as evidence but never relabeled as a newer method
version.
