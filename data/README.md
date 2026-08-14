# Datasets

This project uses three kinds of data:

1. **Base QA-SRL data** (`train` / `dev` / `test`) — downloaded from a URL at runtime.
2. **DPO preference pairs** — built by this repo; the winning set ships under
   `training/dpo/existing_dataset/`.
3. **Evaluation gold/inputs** — ship inside `evaluation/data/`.

There is intentionally **no copy of the multi-GB base splits, caches, or superseded
preference-pair variants** in this repo — the base data is fetched from its canonical
URL, and only the DPO artifacts the winning pipeline actually consumes are included.

---

## 1. Base QA-SRL data (SFT + GRPO)

The SFT and GRPO scripts do **not** read a local data file — they download the splits
directly from the project's canonical host and cache them next to the script:

```
https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/train.json
https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/dev.json
https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/test.json
```

To materialize them locally (offline use / explicit provenance), run the download
helper — it fetches the same three splits:

```bash
python download_data.py                 # -> ./raw/{train,dev,test}.json
python download_data.py --splits dev test
```

- `Stage_CE_Instruct_DEV.py` and `Stage_GRPO_Instruct_DEV.py` build the split URL
  as `_URL + "<split>.json"` and load it via `datasets`/`requests`. No action needed —
  the first run fetches (and locally caches) the data.
- **Split usage (important):** the model is trained on the more exhaustive **`dev`** split, 
  not the 92k **`train`** split. `test` is the held-out metric. Because `dev` is also
  GRPO's training set, checkpoints are never selected on data the stage trained on:
  GRPO uses a split-half protocol over `test` (select on one half, report the other,
  which is why the headline is quoted as a split-half estimate), and DPO uses a
  model-independent held-out `dev` slice.


---

## 2. DPO preference dataset (on-policy recall pairs)

### 2a. Format

Each line of `dpo_train.jsonl` / `dpo_val.jsonl` is one preference example in the
schema TRL's `DPOTrainer` reads directly (no re-templating at train time):

```json
{
  "prompt":   "<|im_start|>system\n…QA-SRL instructions…<|im_start|>user\n…sentence + predicate…",
  "chosen":   "Q1? ans1a <A> ans1b <QA> Q2? ans2 <QA> …",
  "rejected": "Q1? ans1 <QA> …",
  "group_id": "363d47da756b549073f14a8ac1d4832e",
  "predicate": "posted",
  "margin":   0.375
}
```

- `prompt` — the SFT/GRPO chat-templated prompt (system + user turn,
  `enable_thinking=False`), identical across all stages.
- `chosen` — the higher-quality completion (flat `Q? ans <QA> …` string, same format
  the model is trained to emit).
- `rejected` — the contrastive completion.
- `group_id` — stable md5 of the `(sentence, predicate)` group; used for the
  deterministic train/val split.
- `margin` — the F_β score gap between chosen and rejected (diagnostic; not consumed by
  the trainer).

Only `prompt` / `chosen` / `rejected` are required by `DPOTrainer`; the rest are
provenance/bookkeeping.

### 2b. How the pairs are constructed

Two ways of producing the `rejected` side were tried, and the distinction is the main
experimental variable:

- **Off-policy** (the first attempt, superseded — see
  `training/dpo/build_dataset/synthetic_add_truncate/`): negatives are *synthesised* by
  corrupting the gold completion (drop a QA pair, or inject wrong-role ones). They are
  cheap and need no GPU, but they are not errors the model actually makes, so the
  preference signal partly teaches it to avoid mistakes it was never going to produce.
- **On-policy** (what ships and what is reported): both sides are real samples from the
  SFT model, so every pair contrasts two outputs the model genuinely produces. This is
  the arm that yielded 79.59 ± 0.26.

These are the **on-policy recall** pairs. Both sides are sampled from the SFT model
itself, and the negative is chosen to isolate *recall* (not general quality):

- `chosen`  = the highest-**F_β=2** sample among k on-policy samples of the group.
- `rejected`= the **lowest-recall** sample subject to a **precision floor (0.6)**; the
  pair is dropped unless the negative is strictly recall-deficient. This yields a
  "same precision, fewer arguments" contrast (~2.8× recall-dominant), directly
  targeting the model's tendency to under-generate adjunct roles.
- **Validation holdout:** a deterministic md5-parity slice of the `dev` groups
  (`val_bucket()` in `build_onpolicy_pairs.py`) is written to `dpo_val.jsonl`. It is
  model-independent and reproducible, so every arm sees the same train/val partition,
  and `test.json` is not used for selection.

Shipped counts: `dpo_train.jsonl` = 1232 pairs, `dpo_val.jsonl` = 239 pairs.


### 2c. Where it lives

```
training/dpo/existing_dataset/pairs_onpolicy_recall/
    ├── dpo_train.jsonl        ← training pairs
    └── dpo_val.jsonl          ← held-out DEV slice (checkpoint selection)
training/dpo/existing_dataset/val_selection_d2_onpolicy.json  ← recorded ckpt ranking
```

### 2d. Rebuilding it from scratch

Two steps. Both run in the `train_qwen3` conda env; the tokenizer used for prompt
rendering is the SFT adapter dir (`DEFAULT_TOKENIZER_DIR`, resolved from
`$QASRL_SFT_ADAPTER`).

**Step 1 — mine on-policy samples from the SFT checkpoint over `dev`** (GPU; sharded).
The mined shards are the *input* to the pair builder and already ship in
`training/dpo/build_dataset/onpolicy_samples/`. Run once per shard (see the
`build_dataset.mine_samples` block of `training/dpo/config.yaml` for the SLURM/env
details):

```bash
cd training/dpo/build_dataset
# The SFT adapter is NOT distributable and is not in this repo — run training/sft/ to
# produce it, or point this at your own:
CE_CKPT=${QASRL_SFT_ADAPTER:?set QASRL_SFT_ADAPTER to your SFT adapter dir}
for IDX in 0 1; do
  PYTHONPATH=../../shared \
  python mine_onpolicy_samples.py \
      --ckpt "$CE_CKPT" --split dev --k 8 --temperature 1.0 --top_p 0.95 \
      --shard_idx $IDX --n_shards 2 \
      --out onpolicy_samples/dev_k8_shard${IDX}of2.jsonl
done
# → onpolicy_samples/dev_k8_shard{0,1}of2.jsonl
```

Mined-sample schema (one line per group):
`{group_id, sentence, predicate, gold_qas, n_gold, samples}`.

**Step 2 — build the preference pairs** (CPU). `build_onpolicy_pairs.py` imports
helpers from `training/shared/build_dpo_training_data.py`, so put `shared` on
`PYTHONPATH`:

```bash
cd training/dpo/build_dataset
PYTHONPATH=../../shared \
python build_onpolicy_pairs.py \
    --samples onpolicy_samples/dev_k8_shard0of2.jsonl \
              onpolicy_samples/dev_k8_shard1of2.jsonl \
    --mode onpolicy_recall \
    --prec-floor 0.6 \
    --score fbeta2 \
    --min-margin 0.15 \
    --val-frac 0.15 \
    --out-dir ../existing_dataset/pairs_onpolicy_recall
# → dpo_train.jsonl + dpo_val.jsonl in the out-dir
```

The `--mode onpolicy_recall --prec-floor 0.6 --score fbeta2` flags are what make this
the shipped on-policy arm (the builder's default mode is `hybrid`; other modes/floors are the
non-winning arms and are not reproduced here).

---

## 3. Evaluation data

Ships in place under `evaluation/data/`:

| Path | Contents |
|------|----------|
| `data/model_input/` | prompt CSVs fed to the inference scripts (e.g. `passive_red.model_inputs.csv`) |
| `data/gold/` | gold slot-filled CSVs used as the scoring reference (e.g. `gold_updated_passive_filled_slots.csv`) |
| `data/ground_truth/` | source gold annotations |
| `data/model_output/` | recorded raw model predictions per model |
| `data/model_output_filled_slots/` | slot-filled predictions (input to `evaluate_dataset.py`) |
| `data/sentences/` | tokenized/detokenized sentence data |
| `../results/` | recorded scorer output (`summary_data.csv`, `model_comparison.txt`) |

The scoring path is: adapter → `run_qwen3_instruct_inference.py` → raw CSV in
`model_output/` → dummy-slot fill (unlabelled) or Scala `FillQasrlSlots` (labelled) →
`evaluate_dataset.py` vs the gold CSV.

**How the canonical gold was derived** (`gold_updated_passive_filled_slots.csv` is the
single gold all reported numbers are scored against):
`passive_red_test_gold_detokenized_normalized.csv` (337 questions had `"` where an
apostrophe belonged) → quote repair → `passive_red_test_gold_quotefixed_intermediate.csv`
(the 6 core columns, quotes fixed) → Scala `FillQasrlSlots` (adds the 9 slot columns) →
`gold_updated_passive_filled_slots.csv`.

### The three headline prediction files

`data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/` ships the prediction CSVs
behind the three numbers the write-up reports. Each was re-scored against
`gold_updated_passive_filled_slots.csv` and reproduces its recorded figure exactly,
down to the TP/FP/FN counts:

| File | Stage | P | R | **Unlab Arg F1** | TP / FP / FN |
|------|-------|---|---|------|--------------|
| `passive_red_output_SFT_dev_heldout_filled_slots.csv` | SFT, clean `dev_val` selection | 82.23 | 76.80 | **79.42** | 6695 / 1447 / 2023 |
| `passive_red_output_GRPO_beta2_ckpt3600_filled_slots.csv` | GRPO, β=2 @ ckpt-3600 | 79.60 | 82.26 | **80.90** | 7171 / 1838 / 1547 |
| `passive_red_output_DPO_D2_onpolicy_ckpt150_filled_slots.csv` | DPO, on-policy @ ckpt-150 (best seed) | 81.34 | 78.36 | **79.82** | 6831 / 1567 / 1887 |

Reproduce any row with:

```bash
cd evaluation
python scripts/evaluate_dataset.py \
    data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<file>.csv \
    data/gold/gold_updated_passive_filled_slots.csv
```

> **Read the *Unlabelled* rows only.** These three were slot-filled by
> `scripts/add_dummy_slots.py`, which writes `_` into every question slot (verified:
> all rows all-`_`). That is the intended path for the unlabelled metric — the argument
> spans are real — but it means the **Labelled** Argument figures the scorer prints for
> these files are an artifact of the placeholder slots and are not reportable. Labelled
> F1 requires the Scala `FillQasrlSlots` path (see §3 above); the labelled numbers in the
> write-up come from files produced that way.

### GRPO sits on a plateau, not a peak

Three of the shipped reports are **reward-ablation runs**, not competing headline claims —
they exist to show the GRPO result is robust to the reward's β rather than a lucky setting:

| β | checkpoint | Unlab Arg F1 | report |
|---|---|---|---|
| 1.0 (control) | 4812 | 79.66 | not shipped |
| 1.5 | 3600 | 80.98 | `…~posthoc_exp2c_b1p5_ckpt3600_…` |
| **2.0 (headline)** | **3600** | **80.90** | `…~passive_red_output_GRPO_beta2_ckpt3600_…` |
| 3.0 | 3600 | 80.99 | *(matched-step figure; the shipped report is ckpt-4000 at 81.07)* |

At matched step 3600 the three settings span **80.90–80.99 — a spread of 0.09**, i.e. noise.
The gain comes from using **β > 1 at all** (79.66 → ~81, about +1.3), not from tuning β to a
particular value. So the β=2 headline is a representative point on a flat optimum.

`summary_data.csv` (machine-readable) and `model_comparison.txt` (the same data rendered)
are the accumulated scorer output across the whole project.

Both are *derived* files. The per-run `<model>~<run>.txt` reports in the same directory
are the raw scorer output they are built from, and they ship alongside so the summary is
regenerable:

```bash
cd evaluation
# score one run: writes results/<model>~<run>.txt, then refreshes both summary files
python scripts/run_evaluation.py <predictions.csv> <gold.csv>
python scripts/summarize_results.py          # rebuild the summaries alone
```

> `summarize_results.py` rebuilds the summaries **from whatever `.txt` reports it finds**,
> Do not run it against an empty or partial `results/` — it overwrites rather than merges.
> The `Method` column in `model_comparison.txt` is hand-curated for the
> post-hoc runs (the parser emits `N/A` for tags it does not recognise, such as `DPO`);
> re-running the generator resets those labels and they must be re-applied.

**Metric stability.** The Unlabelled figures are deterministic — re-scoring reproduces
them exactly, counts included. The **Labelled** figures drift by ~±0.02 F1 between
identical runs (observed 39.66 / 39.67 / 39.68 on one file), so treat their last decimal
as noise. This is a second reason not to read the labelled column on the dummy-slot files
described above.

### Wiktionary inflection data (`evaluation/datasets/wiktionary/`)

The **labelled** F1 path only: `FillQasrlSlots.scala` loads `en_verb_inflections.txt`
from this directory at runtime to inflect verbs. The `.txt` files here are the shipped
artifact and are all a runner needs — nothing regenerates them as part of the pipeline.

> The `extract_english_*.py` scripts in that directory are **upstream Python 2** tooling
> that originally produced these files from a raw Wiktionary dump (not included). They
> are kept for provenance only, are not part of the runnable path, and will not execute
> under Python 3.
