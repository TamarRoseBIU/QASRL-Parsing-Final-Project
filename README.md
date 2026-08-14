# QA-SRL Parser Training — Qwen3-30B-A3B-Instruct

Training a **Qwen3-30B-A3B-Instruct-2507** model to perform QA-SRL (question-answer
driven semantic role labeling): given a `(sentence, predicate)` pair, generate all
question–answer pairs describing that predicate's arguments.

**Goal:** find out whether reinforcement learning on top of supervised fine-tuning
improves a QA-SRL parser, and which RL objective works better.

**Finding: it does, and GRPO beats DPO — SFT < DPO < GRPO.**

## Approach

Three training stages plus evaluation. Both RL tracks branch independently off the same
supervised checkpoint, so their gains are directly comparable.

```
        SFT (Cross-Entropy, LoRA)          ← base capability
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
    GRPO                     DPO            ← two independent improvement tracks,
 (F_β reward)          (on-policy pairs)      each warm-started from the SFT adapter
      └───────────┬───────────┘
                  ▼
            Evaluation                       ← greedy inference + Unlabelled/Labelled F1
```

- **SFT** — LoRA cross-entropy fine-tuning, the base capability the RL stages build on.
- **GRPO** — group-relative policy optimization against a recall-weighted F_β reward,
  motivated by the SFT model's tendency to under-generate adjunct roles. The winner.
- **DPO** — direct preference optimization on preference pairs sampled from the SFT
  model itself.
- **Evaluation** — greedy inference, then Unlabelled Argument F1 against a fixed gold
  set.

## Results

Unlabelled Argument F1 on the held-out `passive_red` test split:

| Stage | F1 | vs SFT |
|-------|----|--------|
| SFT (cross-entropy) | 78.90 | — |
| DPO (on-policy pairs) | 79.59 ± 0.26 | +0.7 |
| **GRPO** (F_β=2 reward) | **~80.6** | **+1.7** |

GRPO is the winning system; DPO is the second, independently warm-started track. The
GRPO figure is a split-half estimate that corrects for selecting the checkpoint on the
same split it is reported on.

**→ [Detailed experiment documentation](evaluation/results/README.md)** — every
experiment, its configuration and hyperparameters, the checkpoint it was scored at, and
the full results table, including the ablations and the two non-winning variants.

## Repository layout

```
repo_clean/
├── data/
│   ├── README.md             ← how to access / build every dataset
│   └── download_data.py      ← fetch the base train/dev/test splits
│
├── training/
│   ├── run_full_pipeline.py  ← runs SFT → RL → inference → evaluation end to end
│   ├── shared/               ← modules imported by more than one stage
│   ├── sft/                  ← Stage 1: supervised fine-tuning (cross-entropy, LoRA)
│   ├── grpo/                 ← Stage 2a: GRPO with recall-weighted F_β reward (WINNER)
│   └── dpo/                  ← Stage 2b: DPO on on-policy preference pairs
│       ├── build_dataset/        (reproduce the preference pairs)
│       └── existing_dataset/     (the generated preference dataset, ready to train on)
│
└── evaluation/               ← inference + scoring subproject
    ├── config.yaml
    ├── results/README.md     ← detailed experiment documentation
    ├── scripts/              ← inference, slot filling, scoring, summarization
    └── data/                 ← gold, model inputs, and the prediction CSVs behind
                                the reported numbers
```

This repository contains **only the final, best-performing implementation of each
stage**, plus the superseded DPO arm kept for the record. Intermediate ablations and
diagnostic scripts were left out.

## How runs are configured

There are **no shell launchers**. Each stage is a Python entry point plus a
`config.yaml` capturing everything the old SLURM scripts held — conda env, SLURM
resources, entry point, `PYTHONPATH`, hyperparameters, and a ready-to-copy `run:`
command. Read the stage's `config.yaml`, then run the Python entry point directly
(optionally wrapped in your own `sbatch`).

| Stage | Configuration mechanism |
|-------|-------------------------|
| SFT | in-script constants (top of the `.py`) |
| GRPO | environment variables (defaults in the `.py`) |
| DPO | environment variables + CLI args |
| Evaluation | CLI args, two conda envs |

## Environments

| Env | Purpose |
|-----|---------|
| `train_qwen3` | all training + GPU inference |
| `eval` | CPU-only F1 scoring |

GPU jobs run under SLURM, 1 GPU each. Set `<GPU_PARTITION>` / `<SLURM_ACCOUNT>` in the
`config.yaml` files to your own cluster's partition and account.

## Model storage (not in this repo)

Trained adapters, checkpoints, and run logs are read from / written to a storage root
that is **intentionally not part of this repository** (multi-GB model state):

```
$QASRL_BASE_DIR/
    ├── models_save_baseline/<STAGE>/<RUN_NAME>/      final LoRA adapters
    ├── trainer_runs_baseline/<STAGE>/<RUN_NAME>/     checkpoint-*/
    └── logs_baseline/<STAGE>/<RUN_NAME>/
```

`QASRL_BASE_DIR` defaults to `<repo>/runs`, so a fresh clone runs without editing any
file. Point it at a filesystem with room for multi-GB adapters:

```bash
export QASRL_BASE_DIR=/path/to/your/model-storage
```

**The SFT LoRA adapter is not distributable** and is not included here. GRPO and DPO
both warm-start from it. Either run the SFT stage first to produce it, or point
`QASRL_SFT_ADAPTER` at your own:

```bash
export QASRL_SFT_ADAPTER=/path/to/your/sft-adapter
```

---

## Quickstart

Get the base data once (SFT/GRPO also fetch it at runtime; this makes it explicit):

```bash
python data/download_data.py           # -> data/raw/{train,dev,test}.json
```

### Run the whole pipeline

```bash
cd training
python run_full_pipeline.py --sft_data DEV --rl_method GRPO
```

This chains SFT → RL → checkpoint selection → inference → scoring, passing each stage's
adapter to the next and stopping with a clear message if any stage fails. Use
`--rl_method DPO` for the DPO track and `--sft_data TRAIN` for the full-`train`-split
baseline. See `--help` for checkpoint-selection and interpreter options.

### Or run stages individually

Each stage's exact command, with the hyperparameters that reproduce the reported run,
is in its `config.yaml`:

```bash
python training/sft/Stage_CE_Instruct_DEV.py         # see training/sft/config.yaml
python training/grpo/Stage_GRPO_Instruct_DEV.py      # see training/grpo/config.yaml
python training/dpo/Stage_DPO_Instruct_DEV.py        # see training/dpo/config.yaml
```

The DPO preference dataset already ships ready to train on; rebuilding it is optional.

### Evaluate an adapter

Three steps from `evaluation/` — GPU inference in `train_qwen3`, then slot fill and
scoring in `eval`. The exact commands are in `evaluation/config.yaml`; to re-score a
prediction CSV that already ships, only the last step is needed:

```bash
cd evaluation
python scripts/evaluate_dataset.py \
    ./data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<PREDICTIONS>.csv \
    ./data/gold/gold_updated_passive_filled_slots.csv
```

Labelled Argument F1 additionally needs the bundled Scala `FillQasrlSlots` slot-filler
instead of `add_dummy_slots.py`.

---

See [`data/README.md`](data/README.md) for dataset details and
[`evaluation/results/README.md`](evaluation/results/README.md) for the full experiment
record.
