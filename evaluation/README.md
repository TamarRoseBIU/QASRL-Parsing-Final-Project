# Evaluation

How a QA-SRL prediction is scored: what the model outputs, what "slots" are, what the
three metrics mean, and the exact commands.

The numbers themselves live in [`results/README.md`](results/README.md).

## What gets scored

Inference writes one row per predicted question–answer pair:

```
qasrl_id,verb_idx,verb,question,answer_range,answer
Wiki1k:wikinews:675013:1:2,3,indicated,what did something indicate?,4:10,both men were in stable condition
```

`qasrl_id` + `verb_idx` identify the sentence and the predicate inside it. `answer_range`
is a **token span**, `start:end`, into that sentence's tokens. Gold has exactly the same
shape, so scoring is a comparison of two such tables.

## Slots

A QA-SRL question is not just a string — it is a filled template. Nine columns spell the
template out:

| Column | Example (`what did something indicate?`) |
| ------ | --------------------------------------- |
| `wh`   | `what` |
| `aux`  | `did` |
| `subj` | `something` |
| `obj`, `obj2`, `prep`, `verb_prefix` | `_` when the slot is empty |
| `is_passive`, `is_negated` | `False` / `False` |

They exist because two differently-worded questions can ask the same thing. The scorer
treats questions as equivalent when the strings match outright **or** when `wh`, `subj`,
`obj`, `is_passive` and `is_negated` all agree (`scripts/paraphrases.py`). Without slots
it can only catch the exact-string case.

"Slot filling" is the step that adds those nine columns to a raw prediction CSV. Two
fillers exist:

| Filler | Slots it writes | Use it for |
| ------ | --------------- | ---------- |
| `scripts/add_dummy_slots.py` | `_` / `False` everywhere | Unlabelled metrics — the cheap path, CPU only |
| Scala `FillQasrlSlots` | real, parsed from the question | Labelled Argument F1 |

**The dummy path is exact for the unlabelled metrics, not a shortcut with a cost.** Scored
both ways on the same predictions, Unlabelled Argument is identical to the count
(84.84 / 73.74 / **78.90**, TP/FP/FN 6429 / 1149 / 2289) and so is Unlabelled Role.
Only Labelled changes, and it *drops* — **53.71 → 38.64** — because with every slot `_`
only the exact-string branch can fire, so real paraphrases stop counting. That is why the
labelled column is meaningless on a dummy-filled file rather than merely approximate.

## The three metrics

The scorer prints all three; **Unlabelled Argument F1 is the headline** used throughout
this project.

| Metric | What it counts |
| ------ | -------------- |
| **Unlabelled Argument** | Did the model find the right answer spans? Predicted and gold spans are matched by **IoU ≥ 0.3** under **one-to-one** assignment (max-weight bipartite matching), ignoring questions entirely. |
| **Labelled Argument** | Same matching, but a matched pair is downgraded to a miss unless the two questions are equivalent under the paraphrase rule above. Needs real slots. |
| **Unlabelled Role** | How many distinct gold questions got any argument matched. Its false positives are hardcoded to 0, so its **precision always prints 100%** — read the recall. |

All three are **micro**-averaged: one global tp/fp/fn over the whole split, not a mean of
per-sentence scores. Decoding is greedy (`do_sample=False`), so runs are deterministic.

## Scoring an adapter (unlabelled)

Three steps. Inference needs a GPU and the `train_qwen3` env; the rest is CPU in `eval`.

```bash
# 1. inference  (conda activate train_qwen3)
python scripts/run_qwen3_instruct_inference.py \
    --model_path <ADAPTER_DIR> \
    --input  ./data/model_input/passive_red.model_inputs.csv \
    --output ./data/model_output/Qwen3-30B-A3B-Instruct-2507/<TAG>.csv

# 2. slot fill + 3. score  (conda activate eval)
python scripts/add_dummy_slots.py \
    ./data/model_output/Qwen3-30B-A3B-Instruct-2507/<TAG>.csv \
    ./data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<TAG>_filled_slots.csv
python scripts/evaluate_dataset.py \
    ./data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<TAG>_filled_slots.csv \
    ./data/gold/gold_updated_passive_filled_slots.csv
```

Run all of it from this `evaluation/` directory. `scripts/run_evaluation.py <pred> <gold>`
does step 3 and also files the report under `results/`; `scripts/summarize_results.py`
rebuilds `results/summary_data.csv` and `results/model_comparison.txt` from those reports.

To re-score a prediction CSV that already ships, only the last command is needed — see
[`results/README.md`](results/README.md) for which ones ship.

## Scoring Labelled Argument F1 (optional)

Replace step 2 with the bundled Scala slot-filler. It needs a **JVM and sbt** (verified with
OpenJDK 17 and sbt 1.9.3; sbt is not bundled), and it resolves `org.julianmichael::qasrl`
from Maven Central on the first build:

```bash
cd evaluation        # required: the code reads datasets/wiktionary relative to the cwd
sbt "runMain qasrl.slots.FillQasrlSlots \
    data/model_output/Qwen3-30B-A3B-Instruct-2507/<TAG>.csv \
    data/sentences/passive_red_sentences.csv \
    data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/<TAG>_filled_slots.csv"
```

Then score the output with `scripts/evaluate_dataset.py` exactly as above. Labelled
figures drift by ~±0.02 between identical runs, so treat the last decimal as noise.

`FillQasrlSlots.scala` (in `src/main/scala/qasrl/slots/`) is the only Scala code in this
repo. The question-parsing library it builds on is
[julianmichael/qasrl](https://github.com/julianmichael/qasrl) (MIT), pulled from Maven
Central as `org.julianmichael::qasrl:0.1.0` — no vendored copy is needed. The verb
inflections it loads at runtime come from `datasets/wiktionary/`.
