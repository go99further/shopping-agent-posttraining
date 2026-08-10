# Shopping Agent Post-Training

An auditable post-training toolkit for tool-using shopping agents. The project
contains the project-owned pieces of a reproducible pipeline:

```text
teacher trajectories -> action-only SFT -> online GRPO -> frozen evaluation
```

It focuses on the hard parts of long-horizon shopping: legal tool use, variant
selection, public-state process verification, bounded reward sampling, failure
replay, and promotion gates for GPU runs.

## What is included

- A veRL 0.8 integration layer for ShopSimulator-style tool environments.
- Deterministic public-state checks for invalid actions, repeated no-progress
  loops, evidence collection, option selection, and purchase readiness.
- GRPO, GraphGPO and GiGPO-lite components, plus guarded canary and unattended
  promotion controllers.
- SFT data collection and benchmark/evaluation contracts.
- Unit tests and immutable summaries from one 200-task deterministic benchmark.

## Reported benchmark snapshot

All stages used the same frozen 200-task evaluation set and one deterministic
rollout per task. These numbers are experiment artifacts, not claims of
statistical significance.

| Stage | Strict success | Purchase success | Mean reward | Guard rejections |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3.5-2B | 0.0% | 0.0% | -0.1105 | 752 |
| LoRA SFT | 60.5% | 60.5% | 0.4729 | 52 |
| GRPO (step 100) | 62.0% | 62.5% | 0.5158 | 38 |

The machine-readable summaries are in [`results/`](results/).

## Scope and reproducibility

This public repository deliberately excludes ShopSimulator source snapshots,
product catalogs, private task facts, teacher trajectories, model weights,
runtime logs, credentials, and machine-specific launch receipts. Obtain the
environment and data from their respective upstream owners, then configure
their paths through the project configuration. The original evaluation protocol
is documented in [`docs/evaluation.md`](docs/evaluation.md).

GRPO runs require a Linux CUDA environment. The validated stack was Python
3.10, PyTorch 2.10.0+cu128, veRL 0.8.0 and vLLM 0.17.0 on NVIDIA A100 GPUs.
CPU-only development checks can be installed with:

```bash
python -m venv .venv
.venv/bin/pip install -e . pytest
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_action_validation.py \
  tests/test_process_verifier.py tests/test_unattended_promotion_controller.py
```

## Repository layout

```text
src/shopping_grpo/        project-owned Agent, training and evaluation modules
scripts/                  launch validation, collection and promotion scripts
configs/                  experiment and Agent-loop contracts
patches/                  narrow, version-pinned veRL compatibility patches
tests/                    public unit tests
results/                  aggregate benchmark summaries only
docs/                     methods and evaluation protocol
```

## Attribution

This project integrates with [veRL](https://github.com/volcengine/verl) and was
evaluated against the upstream [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator)
environment. Neither upstream source code nor its data are redistributed here.
