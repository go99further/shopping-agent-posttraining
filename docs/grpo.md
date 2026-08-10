# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
  process_verifier.py   deterministic public-state step checks
  gigpo.py              optional public-anchor micro advantages
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train set: `data/grpo/train.parquet` (1,000 tasks)
- Validation set: `data/grpo/validation.parquet` (50 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v3

Hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh --dry-run
```

Train:

```bash
bash scripts/grpo.sh
```

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Maximum training steps | 500 |
| Save / validation frequency | 50 / 50 |
| KL reward / KL loss | disabled / disabled |

Dynamic sampling can generate at most three batches to find a useful update and
permits at most ten consecutive skipped updates. These bounds prevent an
all-equal reward batch from causing an unbounded resampling loop.

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
Advanced overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Three-layer reward and evaluation boundary

The project keeps three signals separate instead of collapsing them into one
opaque reward:

1. **Terminal layer:** frozen Environment v2.1 Reward v3 is the macro objective
   and is never modified by training-side code.
2. **Process layer:** the optional deterministic public-state verifier records
   action legality, state change, new evidence, candidate opening, option
   progress, mechanical variant purchase readiness, repeated no-progress
   decisions and premature purchase. GiGPO-lite can use clipped potential
   deltas for token-level credit.
3. **Judge layer:** the trajectory LLM Judge is evaluation-only. Its online
   training weight is fixed to zero because online use would add latency,
   nondeterminism, leakage risk and an additional reward-hacking surface.

The process layer reads only actor-visible observations and public tool calls.
It never reads the hidden ShopSimulator goal, target ASIN or Reward v3 detail.
Infrastructure-invalid trajectories produce diagnostics but no process reward.
Its implementation is
[`process_verifier.py`](../src/shopping_grpo/training/grpo/process_verifier.py).

Process credit is disabled by default. A dry-run configuration check can enable
it explicitly:

```bash
SHOPPING_GIGPO_PROCESS_ENABLE=true bash scripts/grpo.sh --dry-run -- \
  algorithm.adv_estimator=shopping_gigpo \
  algorithm.shopping_gigpo.process_verifier_enabled=true
```

`potential_reward_clip` mirrors the rollout-side
`SHOPPING_GIGPO_POTENTIAL_CLIP`; preflight requires them to match, and the
advantage estimator rejects turn rewards outside that bound. The recommended
experiment is an ablation, not an assumed replacement for GRPO: compare frozen
Reward v3 only, verifier diagnostics only, and verifier micro-advantage at
several small weights using the same training tasks and paired evaluation.

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The reported comparison uses step 100. Select checkpoints using validation
metrics rather than assuming that the final training step is best.
