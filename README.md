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
 (F_β reward)          (on-policy pairs)      starting from the same SFT weights
      └───────────┬───────────┘
                  ▼
            Evaluation                       ← greedy inference + Unlabelled/Labelled F1
```

- **SFT** — LoRA cross-entropy fine-tuning, the base capability the RL stages build on.
- **GRPO** — group-relative policy optimization against a recall-weighted F_β reward,
  motivated by the SFT model's tendency to miss adjunct arguments, above all the *why*
  and *where* questions. The winner.
- **DPO** — direct preference optimization on preference pairs sampled from the SFT
  model itself.
- **Evaluation** — greedy inference, then Unlabelled Argument F1 against a fixed gold
  set.

## Results

Unlabelled Argument F1 on the held-out **`passive_red`** test split: 2,450 predicates in
999 Wikinews and Wikipedia sentences ([details](data/README.md)):

| Stage                   | F1           | vs SFT   |
| ----------------------- | ------------ | -------- |
| SFT (cross-entropy)     | 78.90        | —        |
| DPO (on-policy pairs)   | 79.59 ± 0.26 | +0.7     |
| **GRPO** (F_β=2 reward) | **~80.6**    | **+1.7** |

Why the two figures are written differently: DPO's is a **mean over 3 seeds**, GRPO's a
**split-half estimate** — its best checkpoint is mid-run, so the checkpoint is chosen on
one half of the test split and scored on the other, which avoids crediting GRPO for a
lucky checkpoint. Both are measured against the same SFT baseline.

That baseline, 78.90, is the checkpoint the reported GRPO and DPO runs started from. A
later SFT run, selected on a held-out slice of `dev` instead of on `test`, reached 79.42;
it is a validation check rather than part of the reported chain
([why](evaluation/results/README.md#3-stage-1--sft-cross-entropy-lora)).

**→ [Detailed experiment documentation](evaluation/results/README.md)** — every
experiment, its configuration and hyperparameters, the checkpoint it was scored at, and
the full results table, including the ablations and the two non-winning variants.

## Environments

Two **conda** environments. Training pins a newer `transformers`/`trl` than the scorer,
and the scorer needs neither a GPU nor the model, so they are kept separate.

| Env           | Purpose                      | Python | Packages                 |
| ------------- | ---------------------------- | ------ | ------------------------ |
| `train_qwen3` | all training + GPU inference | 3.12   | `requirements-train.txt` |
| `eval`        | CPU-only F1 scoring          | 3.10   | `requirements-eval.txt`  |

```bash
# training + GPU inference
conda create -n train_qwen3 python=3.12 -y
conda activate train_qwen3
pip install -r requirements-train.txt

# CPU-only scoring
conda create -n eval python=3.10 -y
conda activate eval
pip install -r requirements-eval.txt
```

Keep the two names as they are: `run_full_pipeline.py` looks each interpreter up by env
name (override with `--python_train` / `--python_eval`). The pinned versions are the ones
the reported runs were produced with; if your CUDA version differs, install the matching
`torch` wheel first and then re-run the `pip install`.

---

## Quickstart

Re-scoring the prediction CSVs that ship needs no GPU — just the `eval` environment. The
training stages need one.

Every command below starts by activating a conda environment. If you have not created
them yet, do that first — see [Environments](#environments) above (`conda create`, then
`pip install -r requirements-train.txt` / `requirements-eval.txt`).

The base model needs no setup: every stage loads `Qwen/Qwen3-30B-A3B-Instruct-2507` from
Hugging Face on first use (~69 GB, cached under `HF_HOME`).

Get the base data once (SFT/GRPO also fetch it at runtime; this makes it explicit):

```bash
conda activate train_qwen3             # either env works: this script only needs requests
python data/download_data.py           # -> data/raw/{train,dev,test}.json
```

### Run the whole pipeline

Start from the SFT checkpoint the reported experiments used — the adapter that ships in
this repo — rather than training a new one:

```bash
conda activate train_qwen3
cd training
python run_full_pipeline.py --sft_data DEV --rl_method GRPO \
    --sft_adapter sft/adapters/sft_dev_selected_on_test
```

That runs RL → checkpoint selection → inference → scoring, passing each stage's adapter to
the next and stopping with a clear message if any stage fails. Use `--rl_method DPO` for
the DPO track.

`sft/adapters/sft_dev_selected_on_test` is the exact SFT checkpoint (78.90 F1) that the
reported GRPO and DPO runs were initialized from, so `--sft_adapter` is how you start from
the same place they did. **It does not guarantee the same final number:** RL training is
stochastic — sampling, rollout order and seed all move the result — and the best
checkpoint sits mid-run, so a rerun lands near the reported figure rather than on it.

To verify the reported numbers exactly, score the prediction CSVs that ship
([Evaluate an adapter](#evaluate-an-adapter) below); those re-derive to the published F1
down to the TP/FP/FN counts.

Omit `--sft_adapter` to train SFT from scratch first (~2,400 examples, 5 epochs). That
follows the same protocol, but produces its own checkpoint rather than the one behind the
reported results. `--sft_data TRAIN` swaps in the full-`train`-split baseline.

**Check your setup first:** add `--dry_run` and it prints every command it would run,
including which interpreter each step gets, without executing anything. Worth doing before
spending GPU hours.

It runs in `train_qwen3` because it reads the per-stage `config.yaml` files (PyYAML), but
it dispatches each step to the right environment itself, finding each interpreter by env
name in the usual conda locations — so the `eval` environment has to exist under that name
too. If it cannot find one it says so and falls back to the current interpreter; point it
at the right one with `--python_train` / `--python_eval` (or `QASRL_PYTHON_EVAL`).

### Or run stages individually

Each stage runs from its own directory, and the two RL stages need `../shared` on
`PYTHONPATH` (they import the reward function from there). The hyperparameters that
reproduce the reported run are in each stage's `config.yaml`. From the repo root:

```bash
conda activate train_qwen3
(cd training/sft  && python Stage_CE_Instruct_DEV.py)
(cd training/grpo && PYTHONPATH=../shared python Stage_GRPO_Instruct_DEV.py)
(cd training/dpo  && PYTHONPATH=../shared python Stage_DPO_Instruct_DEV.py)
```

GRPO and DPO warm-start from the committed SFT adapter, so the SFT stage is optional —
run it only to retrain the baseline yourself. The DPO preference dataset also ships ready
to train on; rebuilding it is optional too.

### Evaluate an adapter

Scoring a prediction CSV that already ships takes one CPU command:

```bash
conda activate eval
cd evaluation
python scripts/evaluate_dataset.py \
    ./data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<PREDICTIONS>.csv \
    ./data/gold/gold_updated_passive_filled_slots.csv
```

Scoring an adapter of your own takes three (inference → slot fill → score). See
**[`evaluation/README.md`](evaluation/README.md)** for those commands and for what the
metrics mean.

## Trained adapters

**Two SFT LoRA adapters ship with this repo** (37 MB each), so GRPO and DPO are runnable
without retraining SFT first:

| `training/sft/adapters/` | F1 | What it is |
| ------------------------ | :--: | ---------- |
| `sft_dev_selected_on_test` | 78.90 | the reported SFT baseline, and the checkpoint the reported GRPO/DPO runs warm-start from — **used by default** |
| `sft_dev_val_heldout`      | 79.42 | the later `dev_val` held-out sanity check, kept for reference; not part of the reported RL chain |

The RL stages resolve their warm start in this order: `$QASRL_SFT_ADAPTER` if set, then
the committed adapter above, then a local SFT run's output. So `training/grpo/` and
`training/dpo/` run as-is on a fresh clone, and you only need the env var to use an
adapter of your own:

```bash
export QASRL_SFT_ADAPTER=/path/to/your/sft-adapter
```

Anything *you* train — adapters, checkpoints, logs — is written under a separate storage
root:

```
$QASRL_BASE_DIR/
    ├── models_save_baseline/<STAGE>/<RUN_NAME>/      final LoRA adapter (~40 MB)
    ├── trainer_runs_baseline/<STAGE>/<RUN_NAME>/     checkpoint-*/ (~700 MB per run)
    └── logs_baseline/<STAGE>/<RUN_NAME>/
```

`QASRL_BASE_DIR` defaults to `<repo>/runs`, so a fresh clone runs without editing any
file. The checkpoints are the bulky part — point it at a filesystem with a few GB free:

```bash
export QASRL_BASE_DIR=/path/to/your/model-storage
```

## Repository layout

```
QASRL-Parsing-Final-Project/
├── LICENSE                    ← MIT
├── requirements-train.txt    ← the train_qwen3 env (training + GPU inference)
├── requirements-eval.txt     ← the eval env (CPU scoring)
│
├── data/
│   ├── README.md             ← every dataset: where it comes from, how to rebuild it
│   └── download_data.py      ← fetch the base train/dev/test splits
│
├── training/
│   ├── run_full_pipeline.py  ← SFT → RL → checkpoint selection → inference → scoring
│   ├── shared/               ← imported by more than one stage: the F_β reward,
│   │                           inference helpers, the DPO data builder
│   ├── sft/                  ← Stage 1: cross-entropy LoRA fine-tuning
│   │   ├── adapters/         ← the two trained SFT adapters (37 MB each), one of which
│   │   │                       GRPO and DPO warm-start from
│   │   └── config.yaml
│   ├── grpo/                 ← Stage 2a: GRPO on a recall-weighted F_β reward (WINNER)
│   └── dpo/                  ← Stage 2b: DPO on on-policy preference pairs
│       ├── build_dataset/    ← rebuild the pairs (mining + pair construction), plus
│       │                       the superseded synthetic arm, kept for the record
│       ├── existing_dataset/ ← the pairs as used, ready to train on
│       └── eval_on_val.py    ← rank DPO checkpoints on the held-out dev slice
│
└── evaluation/               ← inference + scoring
    ├── README.md             ← what slots are, what the three metrics mean, how to score
    ├── config.yaml           ← the exact commands and paths for each step
    ├── scripts/              ← inference, slot filling, the scorer, summarization
    ├── results/              ← one report per evaluation ever run, the summary tables,
    │                           and README.md: the full experiment record
    ├── data/
    │   ├── gold/             ← the scoring reference (and its two derivation stages)
    │   ├── model_input/      ← the prompts fed to inference
    │   ├── model_output/     ← (empty) where inference writes raw predictions
    │   ├── model_output_filled_slots/  ← the 5 prediction CSVs behind the reported
    │   │                                 numbers, plus your own slot-filled output
    │   └── sentences/        ← tokenized sentences, needed by the Scala slot-filler
    │
    └── the Scala slot-filler, for Labelled F1 only:
        ├── src/main/scala/qasrl/slots/FillQasrlSlots.scala
        ├── build.sbt, project/     ← sbt build; the qasrl library comes from Maven
        └── datasets/wiktionary/    ← verb inflections it loads at runtime
```

This repository contains **only the final, best-performing implementation of each
stage**, plus the superseded DPO arm kept for the record. Intermediate ablations and
diagnostic scripts were left out; the experiments behind them are documented in
[`evaluation/results/README.md`](evaluation/results/README.md).

## How runs are configured

Every stage is a Python entry point plus a `config.yaml` next to it, holding the conda
env, the `PYTHONPATH`, the hyperparameters that reproduce the reported run, and a
ready-to-copy command. There are no shell launchers to adapt. Hyperparameters live as
in-script constants for SFT and as environment variables for GRPO and DPO; each
`config.yaml` lists the values used.

---

See [`data/README.md`](data/README.md) for dataset details,
[`evaluation/README.md`](evaluation/README.md) for the metrics and scoring pipeline, and
[`evaluation/results/README.md`](evaluation/results/README.md) for the full experiment
record.
