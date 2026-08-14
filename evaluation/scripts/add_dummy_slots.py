#!/usr/bin/env python3
"""
Add dummy slot columns to a raw Qwen3 QA-SRL prediction CSV so it can be scored for
UNLABELLED argument F1 without running the Scala FillQasrlSlots pipeline.

The transformation applied: keep the predicted question/answer columns and fill every
grammatical-slot column with a dummy value ("_" / False). The unlabelled metric ignores
the slot fields, so the dummy fill produces the same Unlabelled Argument / Role F1 as
the full slot-filled pipeline (verified to match exactly), while the Labelled metric is
meaningless on the output (run the real FillQasrlSlots pipeline when a valid Labelled number
is needed).

Usage (conda `eval` env):
  python add_dummy_slots.py <RAW_PRED_CSV> <OUT_FILLED_CSV>
"""
import sys
import pandas as pd


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python add_dummy_slots.py <RAW_PRED_CSV> <OUT_FILLED_CSV>")
        return 1
    raw_path, fs_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(raw_path)
    keep = ['qasrl_id', 'verb_idx', 'verb', 'question', 'answer_range', 'answer']
    df = df[keep].copy()
    for c in ['wh', 'subj', 'obj', 'obj2', 'aux', 'prep', 'verb_prefix']:
        df[c] = "_"
    for c in ['is_passive', 'is_negated']:
        df[c] = False
    df.to_csv(fs_path, index=False)
    print(f"wrote {fs_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
