# Superseded DPO arm — synthetic add/truncate pairs (off-policy)

**This is not the reported arm.** It is the project's initial hypothesis, kept for the
record. The reported DPO result (**79.59 ± 0.26**, best seed 79.82) came from the
on-policy pairs in `../../existing_dataset/pairs_onpolicy_recall/`.

The difference is *where the negatives come from*. Here they are **off-policy**:
synthesised by corrupting the gold completion, so they are not errors the model actually
makes. The shipped arm is **on-policy** — both sides are sampled from the SFT model
itself, so each pair contrasts two outputs the model genuinely produces.

## What it does

Negatives are built by corrupting the gold completion of each `(sentence, predicate)`
group, one process per group, forced to an exact **50/50** split dataset-wide:

- **`truncate`** — drop one QA pair (or one answer from a multi-answer pair).
- **`add`** — inject 1–3 wrong-role QA pairs borrowed from a *different predicate*.
  The question's verb slot is re-written for the current predicate, but **the answer
  span is copied verbatim from the donor** — that is what makes it wrong-role, and it
  is deliberate. A local-vLLM pass then repairs the grammar of those adapted questions.

Shipped dataset: `dpo_pairs.grammar_fixed.manual_fixed.json` — 2406 records, exactly
1203 `truncate` + 1203 `add`.

## Why it was superseded

Measured against the on-policy arm and the SFT baseline:

| Arm | Unlab Arg F1 |
|---|---|
| this arm, unselected | 78.47 |
| this arm, original run | 79.01 |
| this arm, control re-run | 78.84 |
| **on-policy pairs (the reported arm)** | **79.59 ± 0.26** |
| SFT reference | 78.90 |

At or below the SFT baseline. The forced 50/50 balance was diagnosed as the cause — a
signal balanced between precision- and recall-pushing negatives has no net direction to
move in — which is what motivated the recall-targeted on-policy pairs that replaced it.

## Pipeline

```bash
python build_dpo_data.py                  # -> dpo_pairs.json
sbatch run_grammar_fix.sh                 # local vLLM on GPU -> dpo_pairs.grammar_fixed.json
python manual_fix_grammar.py              # -> dpo_pairs.grammar_fixed.manual_fixed.json
```

Then render to the trainer's schema with `../../../shared/build_dpo_training_data.py`.
Its `--eval-input` default points at `dpo_pairs_test.grammar_fixed.json`, the test-split
counterpart, which is **not shipped** (15 MB, and this arm reports no number). Run with
`--eval-input ''` to carve the eval split off the train file instead.

## Answer-offset correction (differs from the original)

`answer_start_chars` / `answer_end_chars` here are **provenance only** — nothing reads
them. `build_dpo_training_data.render_completion()` renders question + answer *text*, so
these offsets never reach training and affect no reported metric.