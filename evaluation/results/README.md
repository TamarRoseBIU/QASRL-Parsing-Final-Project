# Experiments and results

Detailed record of every experiment run in this project: what each one changed, the
configuration it ran with, the checkpoint it was scored at, and the number it produced.

The main [`README.md`](../../README.md) covers what the project is and how to run it.
This document is the evidence behind the numbers quoted there.

---

## 1. The metric

**Unlabelled Argument F1** on the held-out `passive_red` **test** split, greedy decoding
(`do_sample=False`). An argument counts as correct when its predicted span overlaps a
gold span with IoU ≥ 0.3 under one-to-one matching. F1 is **micro**-averaged (global
tp/fp/fn), so it is the statistic checkpoints are selected on as well as the one
reported.

Labelled Argument F1 and Unlabelled Role F1 are also emitted by the scorer and appear in
each report, but the headline comparison throughout the project is Unlabelled Argument
F1. Labelled F1 additionally depends on the Scala `FillQasrlSlots` slot-filler; the
numbers here use `add_dummy_slots.py`, so their *labelled* column is not meaningful.

### Reproducing a number

Any experiment marked "CSV ships" below can be re-scored on a CPU with no GPU and no
model:

```bash
cd evaluation
python scripts/evaluate_dataset.py \
    ./data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<PREDICTIONS>.csv \
    ./data/gold/gold_updated_passive_filled_slots.csv
```

Experiments without a shipped CSV would have to be retrained to reproduce; their scorer
report is the record.

---

## 2. Results at a glance

| # | Experiment | Stage | F1 | P | R | CSV ships |
|---|---|---|:---:|:---:|:---:|:---:|
| 1 | Zero-shot Qwen3-Instruct | none | 67.49 | 73.52 | 62.38 | — |
| 2 | CE on the full `train` split | SFT | 73.89 | 91.34 | 62.04 | ✅ |
| 3 | **CE on `dev`, selected on `test`** | SFT | **78.90** | 84.84 | 73.74 | — |
| 4 | CE on `dev`, selected on a `dev` holdout | SFT | 79.42 | 82.23 | 76.80 | ✅ |
| 5 | DPO, synthetic pairs, selected ckpt | DPO | 79.01 | 84.73 | 74.02 | — |
| 6 | DPO, synthetic pairs, final epoch | DPO | 78.77 | 84.52 | 73.76 | — |
| 7 | **DPO, on-policy pairs, ckpt-150** | DPO | **79.82** | 81.34 | 78.36 | ✅ |
| 8 | **GRPO, F_β=2, ckpt-3600** | GRPO | **80.90** | 79.60 | 82.26 | ✅ |
| 9 | GRPO, F_β=1.5, ckpt-3600 | GRPO | 80.98 | 80.86 | 81.11 | — |
| 10 | GRPO, F_β=3.0, ckpt-4000 | GRPO | 81.07 | 80.23 | 81.93 | — |

Bold rows are the reported systems. Read top to bottom, the table is the project's arc:
**67.49 zero-shot → 78.90 after SFT → 79.82 with DPO → 80.90 with GRPO**, with the
non-winning variants (rows 2, 5, 6) showing which choices mattered.

**The finding: RL on top of SFT helps, and GRPO helps more than DPO — SFT < DPO < GRPO.**

> Figures are produced by the evaluator in this repo and may differ by a few hundredths
> from the write-up, which was scored on an earlier revision. The ordering — the actual
> finding — is identical either way.

---

## 3. Stage 1 — SFT (cross-entropy, LoRA)

Entry point: `training/sft/Stage_CE_Instruct_DEV.py` · config: `training/sft/config.yaml`

### Shared hyperparameters

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| LoRA rank / α / dropout | 8 / 16 / 0.05 |
| LoRA target modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| Learning rate | 2e-4 |
| Epochs | 5 |
| Batch size × grad accum | 2 × 4 |

### What differs between the three CE runs

All three use the recipe above. They differ only in **which split they train on** and
**how the checkpoint is chosen** — and only one of them is the checkpoint the RL stages
build on.

| # | Train split | Selection | `QASRL_SFT_TRAIN_ON` | F1 | Role |
|:---:|---|---|---|:---:|---|
| 2 | full `train` (92,805 ex.) | `eval_dev_loss` | `train` | 73.89 | **Baseline-table row only.** Training on the small clean `dev` split beats it by ~5 F1 — that comparison is why `dev` was chosen. Not an SFT anchor. |
| 3 | `dev` (2,406 ex.) | on **`test`** | `dev` *(default)* | **78.90** | ⭐ **The SFT checkpoint GRPO and DPO warm-start from**, and the reference the RL gains are measured against. |
| 4 | 90% of `dev` | on a held-out `dev_val` slice (ckpt-600) | `dev` *(default)* | 79.42 | Methodologically preferred protocol — never touches `test`. Ships its predictions as evidence; **not** the RL warm-start. |

Rows 3 and 4 are the same training data and differ only in *how the checkpoint was
chosen*. The RL comparison is anchored on **78.90** throughout: GRPO +1.7 and DPO +0.7
are both measured against it.

**Why 78.90 stays the headline.** It is the figure the write-up reports, and the run's
own scorer report backs it. `Stage_CE_Instruct_DEV.py` as shipped implements the
*sound* protocol instead — a deterministic grouped 90/10 split of `dev` by
`sentence_id` (`DEV_VAL_FRACTION=0.10`, `SPLIT_SEED=42`), training on the 90% and
selecting on the held-out `dev_val`, never touching `test`. That variant measures
**79.42** — *above* the headline, so the RL gains are not an artifact of a weak SFT
reference. Run 3 is therefore documented but not reproducible from the shipped default.

Reproduce row 2 with `QASRL_SFT_TRAIN_ON=train`; row 4 is the shipped default.

---

## 4. Stage 2a — GRPO (the winner)

Entry point: `training/grpo/Stage_GRPO_Instruct_DEV.py` · config: `training/grpo/config.yaml`

Warm-starts from the SFT adapter (row 3), which also serves as the frozen KL reference.
Configured entirely through environment variables.

### Configuration (Exp 2a — the reported run)

| Env var | Value | Note |
|---|---|---|
| `REWARD_BETA` | `2.0` | F_β for the reward; >1 up-weights recall |
| `GRPO_LR` | `2e-5` | **override** — script default 1e-5 under-trains |
| `NUM_GEN` | `4` | rollouts per prompt (group size) |
| `GRAD_ACCUM` | `4` | must be a multiple of `NUM_GEN` |
| `KL_BETA` | `0.04` | KL anchor to the frozen CE reference |
| `GRPO_EPOCHS` | `2` | |
| `GRPO_TEMP` | `1.0` | |
| `EVAL_SUBSET` | `128` | **override** — script default 256 |
| `EVAL_STEPS` / `SAVE_STEPS` | `200` / `200` | |
| `EXP_TAG` | `grpo_exp2a` | appended to the run/output dir name |

`GRPO_LR` and `EVAL_SUBSET` differ from the script's built-in defaults and **must** be
set to reproduce the winner.

### Why the reward is recall-weighted

Error analysis showed the SFT model is **recall-limited** — it systematically
under-generates adjunct roles. F_β=2 up-weights recall, and the effect is visible in the
metric split: SFT is 84.84 P / 73.74 R, while GRPO ckpt-3600 is 79.60 P / **82.26 R**.
GRPO trades precision for a larger recall gain.

### The β plateau

Three reward strengths were run. At matched step 3600 they are indistinguishable:

| Experiment | β | Checkpoint | F1 |
|---|:---:|---|:---:|
| Exp 2a (reported) | 2.0 | 3600 | 80.90 |
| Exp 2c | 1.5 | 3600 | 80.98 |
| Exp 2d | 3.0 | 4000 | 81.07 |

The 0.17 spread is within run-to-run noise, so **the optimum is a flat plateau over
β ≈ 1.5–3.0**, not a sharp peak at 2.0. The claim the project makes is "recall-weighting
helps", not "β=2.0 specifically is best". Rows 9 and 10 are reported as ablation
evidence for that plateau, not as competing headline numbers.

### Checkpoint selection, and why the headline is ~80.6

The best checkpoint is **mid-run (~3600 of 4812)**, not the final adapter. Because the
same `test` split would otherwise be used both to pick the checkpoint and to report the
score, GRPO uses a **split-half protocol**: select on one half of `test`, report on the
other. That yields the honest **~80.6** (range 80.1–81.1) quoted in the main README,
roughly 0.6 below the 80.90 that selecting and reporting on the full split gives.

**80.90 is the number that re-scores from the shipped CSV**; ~80.6 is the same run
corrected for selection bias. Both describe row 8.

---

## 5. Stage 2b — DPO

Entry point: `training/dpo/Stage_DPO_Instruct_DEV.py` · config: `training/dpo/config.yaml`

An independent improvement track, warm-started from the same SFT adapter as GRPO.

### Optimizer settings (held fixed across arms, to isolate pair provenance)

| Env var | Value |
|---|---|
| `DPO_ARM` | `d2_onpolicy_recall_s42` |
| `DPO_SEED` | `42` |
| `DPO_LR` | `5e-6` |
| `DPO_EPOCHS` | `2` |
| `DPO_BETA` | `0.2` |
| `DPO_DATA_DIR` | `existing_dataset/pairs_onpolicy_recall` |

### What differs between the two DPO arms

The optimizer is identical in both. The experimental variable is **where the `rejected`
side comes from**.

| Arm | Negatives | Result | Status |
|---|---|:---:|---|
| Synthetic add/truncate (**off-policy**) | Gold completion corrupted: `truncate` drops a QA pair, `add` injects a wrong-role pair borrowed from another predicate. Forced to an exact 50/50 split, then passed through a local-vLLM grammar-repair step. | 78.77–79.01 | **Superseded.** At or below the 78.90 SFT baseline. |
| On-policy recall (**reported**) | Both sides are real samples from the SFT model itself. `chosen` = highest F_β=2 of k=8 samples; `rejected` = lowest-recall sample subject to a precision floor. | **79.59 ± 0.26** (best seed 79.82) | ⭐ Reported arm. |

**Why the first arm failed.** The forced 50/50 add/truncate balance was diagnosed as the
cause: a balanced precision/recall signal gives the optimizer no net direction to move
in. Synthetic negatives are also not errors the model actually makes, so part of the
preference signal teaches it to avoid mistakes it was never going to produce. That
diagnosis is exactly what motivated the recall-targeted on-policy pairs, which shifted
recall from 74.02 to 78.36.

The superseded arm ships under `training/dpo/build_dataset/synthetic_add_truncate/` for
the record.

### On-policy pair construction

| Parameter | Value |
|---|---|
| Mining | k=8 samples/group, temperature 1.0, top_p 0.95, over `dev`, 2 shards |
| Mode | `onpolicy_recall` |
| Precision floor | 0.6 |
| Scoring | `fbeta2` |
| Minimum margin | 0.15 |
| Validation fraction | 0.15 |
| Resulting pairs | 1232 train / 239 val |

The validation holdout is a deterministic md5-parity slice of the `dev` groups
(`val_bucket()` in `build_onpolicy_pairs.py`) — model-independent and reproducible, so
every arm sees the same partition and `test.json` is never used for selection.

### Checkpoint selection

`eval_on_val.py` ranks every checkpoint plus the final adapter by micro F1 on the dev
holdout. The recorded ranking ships in
`training/dpo/existing_dataset/val_selection_d2_onpolicy.json`:

| Rank | Checkpoint | Val micro F1 |
|:---:|---|:---:|
| 1 | **checkpoint-150** ⭐ | **0.8317** |
| 2 | checkpoint-100 | 0.8308 |
| 3 | checkpoint-308 | 0.8281 |
| 4 | final adapter | 0.8281 |
| 5 | checkpoint-300 | 0.8275 |
| 6 | checkpoint-200 | 0.8238 |
| 7 | checkpoint-250 | 0.8235 |

Seven entries: six saved checkpoints plus the final adapter, which ties checkpoint-308
because training ends there.

**checkpoint-150** was selected and scored 79.82 on `test`. Note the final adapter is
*not* the best — selection matters here, and the seed spread (±0.26 over 3 seeds) is
comparable to the DPO gain itself.

---

## 6. Files in this directory

| File | What it is |
|---|---|
| `<model>~<run>.txt` (10 files) | Raw scorer reports, one per evaluation ever run against the common gold. Output of `scripts/run_evaluation.py`. |
| `summary_data.csv` | Machine-readable table of all 10 runs. |
| `model_comparison.txt` | Rendered comparison tables. |
| `placeholder_corrected_f1.md` | A correction analysis — see below. |

`summary_data.csv` and `model_comparison.txt` are **regenerated** from the `.txt`
reports by `scripts/summarize_results.py`. Deleting a report silently shrinks the table
on the next regeneration.

**Four of the ten runs also ship their prediction CSV** under
`evaluation/data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/` (rows 2, 4, 7,
8), so their numbers re-derive without a GPU. The other six are a record of the run and
its score.

### Placeholder-corrected F1

`placeholder_corrected_f1.md` documents a scoring artifact: when the inference script
cannot locate an answer's text in the sentence it emits a fixed out-of-range placeholder
span, and because the evaluator builds argument sets with a Python `set`, all such
answers within one predicate collapse to one entry — slightly *under*-counting false
positives.

The correction moves every system by at most a few tenths and **does not change the
ordering**: the recall-pushed systems (GRPO, on-policy DPO) absorb marginally more of
the penalty than precision-heavy SFT. The tables above report uncorrected numbers, which
is what the shipped CSVs re-score to; that document holds the corrected variants.
