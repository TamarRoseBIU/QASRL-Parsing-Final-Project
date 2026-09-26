# Datasets

This project uses three kinds of data:

1. **Base QA-SRL data** (`train` / `dev` / `test`) — downloaded from a URL at runtime.
2. **DPO preference pairs** — built by this repo; the winning set ships under
   `training/dpo/existing_dataset/`.
3. **Evaluation gold/inputs** — ship inside `evaluation/data/`.

There is intentionally **no copy of the base splits (~220 MB), their caches, or the
superseded preference-pair variants** in this repo — the base data is fetched from its canonical
URL, and only the DPO artifacts the winning pipeline actually consumes are included.

---

## 1. Base QA-SRL data (SFT + GRPO)

`passive_red` is a QA-SRL corpus of Wikinews and Wikipedia sentences. One example is a
`(sentence, predicate)` group with its gold question–answer pairs:

| Split | Size | Groups | Used for |
| ----- | ---- | ------ | -------- |
| `train` | 204 MB | 92,805 | **not** used for training here — see below |
| `dev`   | 7.9 MB | 2,406  | what every stage trains on |
| `test`  | 7.3 MB | 2,450  | the held-out split all reported numbers are measured on |

**Why train on `dev` rather than `train`?** `train` is large but noisily annotated, while
`dev` is densely and carefully annotated. The same recipe scores **78.90** trained on
`dev` against **73.89** trained on the 92k `train` split, so `dev` is the deliberate
choice — not an oversight.

**How checkpoints are chosen, given that `dev` is the training data.** A stage cannot
select its checkpoint on data it trained on, so GRPO selects on one half of `test` and
reports on the other (this is the "split-half estimate" behind the ~80.6 headline), and
DPO selects on a held-out slice of `dev` carved out before training.

The stages download the splits themselves on first use and cache them next to the script,
so no setup is needed:

```
https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/{train,dev,test}.json
```

To fetch them explicitly instead — for offline use, or just to see them:

```bash
conda activate train_qwen3              # either env works: only requests is needed
python download_data.py                 # -> ./raw/{train,dev,test}.json
python download_data.py --splits dev test
```

That host is the only place data is fetched from; every other dataset this project uses
ships in the repo.


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

DPO needs a better and a worse answer for each group. Where the *worse* one comes from
was the experiment:

- **First attempt (superseded, kept under
  `training/dpo/build_dataset/synthetic_add_truncate/`):** corrupt the gold answer — drop
  a QA pair, or paste in a wrong one. Cheap and needs no GPU, but these are not mistakes
  the model actually makes, so part of the signal teaches it to avoid errors it would
  never have produced.
- **What ships:** both sides are real samples from the SFT model, so each pair contrasts
  two outputs it genuinely produces. This is the arm that reached 79.59 ± 0.26.

In the shipped pairs, for each group the model is sampled k=8 times and every sample is
scored against gold:

- `chosen` — the sample with the best F_β=2 score (β=2 weights recall over precision).
- `rejected` — the sample with the *lowest recall*, but only if its precision is still
  above **0.6**. That floor is what makes the pair say "you missed arguments" rather than
  "you wrote nonsense": across the shipped set the two sides differ by +0.38 in recall and
  only +0.14 in precision, and the rejected side names about one argument fewer than gold.
  Pairs that fail the floor, or whose two sides are too close (`--min-margin 0.15`), are
  dropped.
- **The validation slice** (`dpo_val.jsonl`) is picked by hashing each group's id, so it
  is the same slice on every run and independent of the model. `test` is never used to
  choose a DPO checkpoint.

Shipped counts: 1,232 training pairs and 239 validation pairs.


### 2c. Where it lives

```
training/dpo/existing_dataset/pairs_onpolicy_recall/
    ├── dpo_train.jsonl        ← training pairs
    └── dpo_val.jsonl          ← held-out DEV slice (checkpoint selection)
training/dpo/existing_dataset/val_selection_d2_onpolicy.json  ← recorded ckpt ranking
```

### 2d. Rebuilding it from scratch

Two steps. Both run in the `train_qwen3` conda env (see
[Environments](../README.md#environments) for how to create it); the tokenizer used for prompt
rendering is the SFT adapter dir (`DEFAULT_TOKENIZER_DIR`, resolved from
`$QASRL_SFT_ADAPTER`).

**Step 1 — sample from the SFT model over `dev`** (GPU). For each
`(sentence, predicate)` group it draws k=8 samples and scores each one against gold.
Its output is the input to step 2, and it already ships in
`training/dpo/build_dataset/onpolicy_samples/`, so this step is only needed to rebuild
from scratch. One command:

```bash
conda activate train_qwen3
cd training/dpo/build_dataset
PYTHONPATH=../../shared \
python mine_onpolicy_samples.py \
    --ckpt ../../sft/adapters/sft_dev_selected_on_test \
    --split dev --k 8 --temperature 1.0 --top_p 0.95 \
    --out onpolicy_samples/dev_k8.jsonl
```

Pass a different `--ckpt` to sample from an SFT adapter of your own. See the
`build_dataset.mine_samples` block of `training/dpo/config.yaml` for the same settings in
config form.

> **Splitting the work across jobs (optional).** One pass over all ~2,400 `dev` groups
> takes longer than the 4-hour GPU slot this project ran under, so the shipped samples
> were produced in **two halves, run as two jobs**: `--n_shards 2 --shard_idx 0` and
> `--n_shards 2 --shard_idx 1`. Each half is a fixed, reproducible slice of `dev` (every
> other group, by position), which is why the two files that ship are named
> `dev_k8_shard0of2.jsonl` and `dev_k8_shard1of2.jsonl`. Step 2's `--samples` takes any
> number of these files, so one file or two makes no difference to the result. With the
> defaults (`--n_shards 1`), the single command above does the whole split in one go.

Mined-sample schema (one line per group):
`{group_id, sentence, predicate, gold_qas, n_gold, samples}`.

**Step 2 — build the preference pairs** (CPU). Takes the samples from step 1 and turns
each group into one `chosen` / `rejected` pair. List every file step 1 produced after
`--samples` — one if you ran it once, the two shipped halves as below. It imports helpers
from `training/shared/build_dpo_training_data.py`, so put `shared` on `PYTHONPATH`:

```bash
conda activate train_qwen3
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
| `data/model_output/` | **empty** — where inference writes raw predictions |
| `data/model_output_filled_slots/` | slot-filled predictions (input to `evaluate_dataset.py`); the 5 shipped CSVs live here |
| `data/sentences/` | tokenized/detokenized sentence data; the Scala slot-filler reads `passive_red_sentences.csv` |
| `../results/` | recorded scorer output (`summary_data.csv`, `model_comparison.txt`) |

The scoring path is: adapter → `run_qwen3_instruct_inference.py` → raw CSV in
`model_output/` → dummy-slot fill (unlabelled) or Scala `FillQasrlSlots` (labelled) →
`evaluate_dataset.py` vs the gold CSV. See
[`evaluation/README.md`](../evaluation/README.md) for what slots are and what each metric
measures.

**How the canonical gold was derived** (`gold_updated_passive_filled_slots.csv` is the
single gold all reported numbers are scored against):
`passive_red_test_gold_detokenized_normalized.csv` (337 questions had `"` where an
apostrophe belonged) → quote repair → `passive_red_test_gold_quotefixed_intermediate.csv`
(the 6 core columns, quotes fixed) → Scala `FillQasrlSlots` (adds the 9 slot columns) →
`gold_updated_passive_filled_slots.csv`.

### The shipped prediction files

`data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/` ships these prediction
CSVs. Each was re-scored against `gold_updated_passive_filled_slots.csv` and reproduces
its recorded figure exactly, down to the TP/FP/FN counts. **Two SFT files ship, and the
distinction matters:** `..._dev_selected_on_test` is the reported 78.90 baseline and the
checkpoint the GRPO/DPO experiments were initialized from; `..._dev_heldout` is the later
`dev_val` sanity check, not part of the reported RL chain (see
[the experiment notes](../evaluation/results/README.md#3-stage-1--sft-cross-entropy-lora)).

| File | Stage | P | R | **Unlab Arg F1** | TP / FP / FN |
|------|-------|---|---|------|--------------|
| `passive_red_output_CE_train_split_filled_slots.csv` | SFT on the full `train` split (baseline-table row) | 91.34 | 62.04 | **73.89** | 5409 / 513 / 3309 |
| `passive_red_output_SFT_dev_selected_on_test_filled_slots.csv` | **SFT baseline** — `dev`, selected on `test`; the GRPO/DPO warm start | 84.84 | 73.74 | **78.90** | 6429 / 1149 / 2289 |
| `passive_red_output_SFT_dev_heldout_filled_slots.csv` | SFT `dev_val` sanity check (not the RL warm start) | 82.23 | 76.80 | **79.42** | 6695 / 1447 / 2023 |
| `passive_red_output_GRPO_beta2_ckpt3600_filled_slots.csv` | GRPO, β=2 @ ckpt-3600 | 79.60 | 82.26 | **80.90** | 7171 / 1838 / 1547 |
| `passive_red_output_DPO_D2_onpolicy_ckpt150_filled_slots.csv` | DPO, on-policy @ ckpt-150 (best seed) | 81.34 | 78.36 | **79.82** | 6831 / 1567 / 1887 |

Reproduce any row with:

```bash
conda activate eval
cd evaluation
python scripts/evaluate_dataset.py \
    data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<file>.csv \
    data/gold/gold_updated_passive_filled_slots.csv
```

> **Which files carry real slots.** Verified per file:
>
> | File | Slots | Labelled figure |
> |------|-------|-----------------|
> | `..._CE_train_split` | real (Scala `FillQasrlSlots`) | meaningful |
> | `..._SFT_dev_selected_on_test` | real (Scala `FillQasrlSlots`) | meaningful (53.72) |
> | `..._SFT_dev_heldout` | dummy — every slot `_` | **not reportable** |
> | `..._GRPO_beta2_ckpt3600` | dummy — every slot `_` | **not reportable** |
> | `..._DPO_D2_onpolicy_ckpt150` | dummy — every slot `_` | **not reportable** |
>
> `scripts/add_dummy_slots.py` writes `_` into every question slot. That is the intended
> path for the unlabelled metric — the argument spans are real — but the **Labelled**
> Argument figures the scorer prints for those three files are an artifact of the
> placeholder slots. Labelled numbers also drift ~±0.02 between runs, so on the two
> real-slot files treat the last decimal as noise.

The scores themselves, including which reward settings were ablated and why, are in
[`evaluation/results/README.md`](../evaluation/results/README.md). How to run the scorer
is in [`evaluation/README.md`](../evaluation/README.md).

### Wiktionary inflection data (`evaluation/datasets/wiktionary/`)

Used by the **labelled** F1 path only: the Scala `FillQasrlSlots` loads
`en_verb_inflections.txt` from here at runtime to inflect verbs. These `.txt` files are
the finished artifact — nothing in the pipeline regenerates them, and a runner needs
nothing else.
