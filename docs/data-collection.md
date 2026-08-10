# Data collection

## Goal

The SFT stage needs complete examples of a shopping agent using tools correctly:
searching, opening products, inspecting evidence, choosing options and ending
with a valid purchase. The repository contains the accepted action-only
trajectories, not historical failed collection attempts.

## How the dataset was produced

The final collection used ShopSimulator Environment v2.1, Reward v3 and
`deepseek-v4-flash` as the teacher. Seven batches produced 604 raw trajectories.
Every trajectory executed its actions in ShopSimulator during collection. The
saved result was accepted only when Environment v2.1 returned a valid Reward v3
gold purchase; no second model judged whether the trajectory succeeded.

Collection audit:

| Item | Value |
|---|---:|
| Raw trajectories | 604 |
| Unique task IDs | 604 |
| Accepted gold trajectories | 428 |
| Acceptance rate | 70.9% |
| Mean raw reward | 0.6121 |
| Mean steps | 11.3 |
| Guard violations | 0 |
| HTTP 400 responses | 0 |
| Collection errors | 4 |

The 428 accepted trajectories were split into 379 training and 49 validation
rows. Assistant reasoning was removed; the SFT target contains only the
observable action protocol. This keeps the training contract aligned with what
the environment can verify.

## Frozen deliverables

| File | Rows | SHA-256 |
|---|---:|---|
| `data/sft/train.jsonl` | 379 | `8cd1f72130b3c781d5ffe08fe3e399b2a9e45d204e3f3bd0d8e677d1b51c8ec5` |
| `data/sft/validation.jsonl` | 49 | `f8ae506d0fa9d1526342a9f717da24922c8a55776d076a296698abac4fde05b3` |

The aggregate raw collection had SHA-256
`b1db9e41673d285da7164e8352fa0a702f537157792fa137c94f7cf200435fa1`;
the accepted aggregate had SHA-256
`aab4d81f134dfcd40e67611f5a413142e4825d5cb6ea60b697536aec2c88fab7`.
Raw teacher responses are intentionally not part of the beginner repository.

## Run a new collection

Start ShopSimulator, configure an OpenAI-compatible Teacher endpoint, and run:

```bash
export OPENAI_BASE_URL=https://your-provider.example/v1
export OPENAI_API_KEY=your-key

python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --model deepseek-v4-flash \
  --target-accepted 428 \
  --workers 4
```

`raw.jsonl` is the resumable source of truth. Running the same command again
skips completed task attempts and rebuilds all derived files:

```text
outputs/sft-collection/
  raw.jsonl           complete Teacher responses and environment results
  accepted.jsonl      strict Reward v3 gold trajectories
  rejected.jsonl      task IDs and deterministic rejection reasons
  reject_stats.json   aggregate acceptance audit
  sft.jsonl           sanitized training rows before splitting
  train.jsonl         task-disjoint training split
  validation.jsonl    task-disjoint validation split
  metadata.json       row counts, configuration and SHA-256 hashes
```

The command removes all task IDs listed in `data/evaluation/tasks.jsonl` before
collection and checks again while building artifacts. It also keeps at most one
accepted trajectory per task. To rebuild the derived files without contacting
the Teacher or environment, run:

```bash
python scripts/collect_sft_data.py \
  --build-only \
  --output-dir outputs/sft-collection
```

Only copy `train.jsonl`, `validation.jsonl` and their metadata into `data/sft/`
after reviewing the collection audit. Raw Teacher responses remain in
`outputs/` and should not be committed.

## What a training row contains

Each JSONL row is a chat trajectory with:

- the shopping instruction;
- assistant tool calls;
- ShopSimulator tool observations;
- the final terminal action;
- metadata tying the row to Environment v2.1 and Reward v3.

During SFT, user and tool tokens are masked. Loss is computed only on assistant
actions. See [SFT](sft.md) for the exact training recipe.
