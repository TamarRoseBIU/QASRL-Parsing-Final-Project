# Placeholder-corrected Unlabelled Argument F1

**Date:** 2026-07-25
**Metric:** Unlabelled Argument F1 on `passive_red` test, greedy decoding.

## The bug this corrects

When the Qwen inference script (`scripts/run_qwen3_instruct_inference.py`) produces an
answer whose text it **cannot locate** as a token span in the sentence, it substitutes
a fixed out-of-range placeholder span **`999:1998`** (`INVALID = "999:1998"`, line 635)
so the answer count is preserved.

The evaluator (`scripts/evaluate.py`) then builds the argument sets with

```python
sys_args = set(arg for role in sys_roles for arg in role.arguments)
```

Because this is a **`set`** and every unlocatable answer decodes to the *same* tuple
`(999, 1998)`, **all placeholder answers within one predicate collapse to a single
element** → they are scored as **one** false positive no matter how many unlocatable
answers the model actually emitted. So a predicate that hallucinated 5 unlocatable
spans is penalized as if it made a single mistake.

`(999, 1998)` never overlaps a real gold span (gold spans use real, low token indices),
so a placeholder is always an unmatched false positive — never a TP, and it never
affects recall or FN.

## The correction

Give every placeholder answer a **unique, disjoint** span so each counts as its own
false positive (instead of collapsing). Equivalently, in closed form:

```
FP_corrected = FP_baseline + (placeholder_spans − predicates_with_a_placeholder)
TP, FN unchanged   →   recall unchanged; only precision (and thus F1) drops.
```

Both methods were computed and agree exactly (the empirical uniquify-and-rescore run
matches the formula; TP/FN verified unchanged).

## Placeholder counts

`placeholder_spans` = total `999:1998` occurrences across all answers;
`predicates_with_a_placeholder` = distinct `(qasrl_id, verb_idx)` having ≥1 placeholder
(= number of placeholder-FPs the buggy set-based metric actually counts).

| model | file | placeholder spans | predicates w/ placeholder | extra FP added |
|-------|------|:-:|:-:|:-:|
| SFT instruct-dev | `passive_red_CE_Instruct_dev_val_TEST_output_filled_slots.csv` | 29 | 28 | 1 |
| DPO (on-policy, ckpt150) | `posthoc_d2_onpolicy_ckpt150_output_filled_slots.csv` | 89 | 44 | 45 |
| GRPO exp2d (β=3, ckpt4000) | `posthoc_exp2d_b3_ckpt4000_output_filled_slots.csv` | 64 | 38 | 26 |
| GRPO exp2c (β=1.5, ckpt3600) | `posthoc_exp2c_b1p5_ckpt3600_output_filled_slots.csv` | 38 | 35 | 3 |

Notes:
- **DPO (on-policy)** is the most affected: it is the recall-pushed arm, so it hallucinates the
  most unlocatable spans, and they cluster (89 spans over only 44 predicates → many
  predicates with several placeholders each), which is exactly the case the set-collapse
  bug hides.

## Corrected results

TP / FP are the actual matched-argument counts; baseline reproduces the numbers in
`model_comparison.txt` exactly.

| model | TP | FN | FP base → corr | P base → corr | R (unch.) | **F1 base → corr** | ΔF1 |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GRPO exp2d (β=3, ckpt4000) — *highest F1* | 7143 | 1575 | 1760 → 1786 | 80.23 → 80.00 | 81.93 | **81.07 → 80.95** | −0.12 |
| GRPO exp2c (β=1.5, ckpt3600) | 7071 | 1647 | 1674 → 1677 | 80.86 → 80.83 | 81.11 | **80.98 → 80.97** | −0.01 |
| DPO (on-policy, ckpt150) | 6831 | 1887 | 1567 → 1612 | 81.34 → 80.91 | 78.36 | **79.82 → 79.61** | −0.21 |
| SFT instruct-dev | 6429 | 2289 | 1149 → 1150 | 84.84 → 84.83 | 73.74 | **78.90 → 78.89** | −0.00 |

## Takeaways

- The bug **inflates precision** by under-counting hallucinated-but-unlocatable answers.
  The effect is small here (≤ 0.21 F1) because placeholders are rare (0.4–1.3% of answers).
- **Ranking is essentially unchanged.** After correction the two GRPO checkpoints are a
  dead heat (80.95 vs 80.97 — within noise; exp2c nominally edges ahead only because
  exp2d had more placeholders), and GRPO still clears DPO (79.61) and SFT (78.89).
- The correction is **precision-only** (recall/FN are untouched), so recall-heavy models
  (DPO on-policy, GRPO) absorb slightly more of the penalty than the precision-heavy SFT.
