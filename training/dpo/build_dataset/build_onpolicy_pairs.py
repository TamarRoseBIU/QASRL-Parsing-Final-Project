"""
Build DPO preference pairs from mined on-policy samples (CPU only).

Consumes the JSONL dumps from mine_onpolicy_samples.py and emits
dpo_train.jsonl / dpo_val.jsonl in the exact schema Stage_DPO_Instruct_DEV.py
already reads (prompt / chosen / rejected), so the trainer needs no change.

WHY THREE MODES:

  onpolicy : chosen = best-scoring sample, rejected = worst-scoring sample.
             The purest test of "make DPO on-policy".

  hybrid   : chosen = gold, rejected = worst-scoring sample.
             Keeps the positive high-quality (as the current pipeline does) but
             replaces the SYNTHETIC negative with a REAL model failure. Since the
             model's errors are overwhelmingly recall errors (under-generation),
             the worst-of-k sample is usually an under-generation, so this yields
             the recall-skewed signal this stage needs WITHOUT hand-tuning an
             add/truncate ratio. Expected to be the strongest arm.

  hybrid_hard : as hybrid, but rejected = the worst sample that still parses,
             and groups whose worst sample is degenerate (unparseable/empty) are
             dropped. Guards against the negative being trivially bad — a pair the
             model already assigns near-zero probability carries little gradient.

  onpolicy_recall : THE SHIPPED ARM. chosen = highest-F_beta=2 sample,
             rejected = LOWEST-RECALL sample subject to a precision floor, and the
             pair is dropped unless the negative is strictly recall-deficient.

VALIDATION HOLDOUT: a deterministic md5-parity slice of dev groups is held
out of the training pairs and written to dpo_val.jsonl. DPO checkpoints are selected
on it. This is model-independent and reproducible, and keeps test.json entirely out
of the selection loop. Note the SFT stage did see these groups, so the holdout is a
valid signal for ranking DPO checkpoints, not a clean generalization estimate.

"""

import argparse
import hashlib
import json
import statistics as st
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from build_dpo_training_data import (
    DEFAULT_TOKENIZER_DIR,
    make_prompt,
    render_completion,
)


def val_bucket(group_id: str, val_frac: float) -> bool:
    """Deterministic holdout: hash the group id to [0,1) and compare to val_frac.
    Model-independent and stable across reruns / pairing modes, so every arm sees
    the SAME train/val partition and arms stay comparable."""
    h = int(hashlib.md5(("val::" + group_id).encode("utf-8")).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < val_frac


def build_pair(rec: dict, mode: str, score_key: str, min_margin: float,
               prec_floor: float = 0.6):
    """Returns (chosen_text, rejected_text, diag) or (None, None, reason)."""
    samples = rec["samples"]
    if not samples:
        return None, None, "no_samples"

    parsed = [s for s in samples if s["parse_ok"]]

    if mode == "onpolicy_recall":
        if not parsed:
            return None, None, "all_unparseable"
        best = max(parsed, key=lambda s: s[score_key])
        floored = [s for s in parsed if s["precision"] >= prec_floor]
        if not floored:
            # Every sample is below the precision floor: the only available
            # negatives are globally degenerate, which is the exact contrast this
            # mode exists to avoid. Drop rather than dilute the signal.
            return None, None, "no_sample_above_prec_floor"
        worst = min(floored, key=lambda s: s["recall"])
        if worst is best:
            return None, None, "degenerate_pair"
        if worst["recall"] >= best["recall"]:
            # Negative is not actually recall-deficient -> wrong direction.
            return None, None, "negative_not_recall_deficient"
        chosen_text, rejected_text = best["text"], worst["text"]
        margin = best[score_key] - worst[score_key]
        if not chosen_text or not rejected_text:
            return None, None, "empty_side"
        if chosen_text.strip() == rejected_text.strip():
            return None, None, "identical"
        if margin < min_margin:
            return None, None, "below_margin"
        return chosen_text, rejected_text, {
            "margin": margin,
            "worst_n_qa": worst["n_qa"],
            "best_n_qa": best["n_qa"],
            "n_gold": rec["n_gold"],
            "worst_recall": worst["recall"],
            "worst_precision": worst["precision"],
            "d_recall": best["recall"] - worst["recall"],
            "d_precision": best["precision"] - worst["precision"],
        }

    if mode == "hybrid_recall":
        # chosen = GOLD (perfect P and R), rejected = the SAME precision-floored
        # min-recall on-policy sample as onpolicy_recall. This crosses a real
        # recall-deficient negative with a clean gold positive, isolating
        # "real under-generation negative" on the recall axis where the SFT->GRPO
        # gap lives, without the sub-greedy sampled positive that caps
        # onpolicy_recall (best-of-8 F1 0.808 < greedy 0.836).
        if not parsed:
            return None, None, "all_unparseable"
        floored = [s for s in parsed if s["precision"] >= prec_floor]
        if not floored:
            return None, None, "no_sample_above_prec_floor"
        worst = min(floored, key=lambda s: s["recall"])
        by_q: dict[str, list[str]] = {}
        for qa in rec["gold_qas"]:
            by_q.setdefault(qa["question"], []).append(qa["answer"])
        chosen_text = render_completion(
            [{"question": q, "answers": a} for q, a in by_q.items()]
        )
        rejected_text = worst["text"]
        # Gold scores 1.0 by construction, so the margin is 1 - worst.
        margin = 1.0 - worst[score_key]
        if not chosen_text or not rejected_text:
            return None, None, "empty_side"
        if chosen_text.strip() == rejected_text.strip():
            return None, None, "identical"
        if margin < min_margin:
            return None, None, "below_margin"
        return chosen_text, rejected_text, {
            "margin": margin,
            "worst_n_qa": worst["n_qa"],
            "best_n_qa": worst["n_qa"],
            "n_gold": rec["n_gold"],
            "worst_recall": worst["recall"],
            "worst_precision": worst["precision"],
            # Positive is gold (P=R=1.0), so deltas are measured against 1.0.
            "d_recall": 1.0 - worst["recall"],
            "d_precision": 1.0 - worst["precision"],
        }

    if mode == "hybrid_hard":
        if not parsed:
            return None, None, "all_unparseable"
        pool = parsed
    else:
        pool = samples

    best = max(pool, key=lambda s: s[score_key])
    worst = min(pool, key=lambda s: s[score_key])

    if mode == "onpolicy":
        chosen_text, rejected_text = best["text"], worst["text"]
        margin = best[score_key] - worst[score_key]
    else:  # hybrid / hybrid_hard
        # load_split() yields ONE ROW PER ANSWER, so a question with multiple gold
        # answers appears several times. Group answers back under their question
        # (order-preserving) or render_completion would emit the same question
        # repeatedly instead of joining its answers with " <A> " — which is the
        # format the model was SFT-trained on.
        by_q: dict[str, list[str]] = {}
        for qa in rec["gold_qas"]:
            by_q.setdefault(qa["question"], []).append(qa["answer"])
        chosen_text = render_completion(
            [{"question": q, "answers": a} for q, a in by_q.items()]
        )
        rejected_text = worst["text"]
        # Gold scores 1.0 by construction, so the margin is 1 - worst.
        margin = 1.0 - worst[score_key]

    if not chosen_text or not rejected_text:
        return None, None, "empty_side"
    if chosen_text.strip() == rejected_text.strip():
        return None, None, "identical"
    if margin < min_margin:
        return None, None, "below_margin"

    return chosen_text, rejected_text, {
        "margin": margin,
        "worst_n_qa": worst["n_qa"],
        "best_n_qa": best["n_qa"],
        "n_gold": rec["n_gold"],
        "worst_recall": worst["recall"],
        "worst_precision": worst["precision"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True,
                    help="JSONL dumps from mine_onpolicy_samples.py")
    ap.add_argument("--mode", default="hybrid",
                    choices=["onpolicy", "hybrid", "hybrid_hard", "onpolicy_recall",
                             "hybrid_recall"])
    ap.add_argument("--prec-floor", type=float, default=0.6,
                    help="onpolicy_recall only: minimum precision for the negative, "
                         "so it is recall-deficient rather than globally degenerate.")
    ap.add_argument("--score", default="fbeta2", choices=["fbeta2", "f1"],
                    help="Which score ranks samples. fbeta2 = recall-weighted.")
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    recs = []
    for p in args.samples:
        with open(p, encoding="utf-8") as fh:
            recs.extend(json.loads(line) for line in fh if line.strip())
    print(f"loaded {len(recs)} groups from {len(args.samples)} shard(s)")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    train_rows, val_rows = [], []
    drops = Counter()
    margins, dn = [], []
    d_rec, d_prec = [], []

    for rec in recs:
        chosen, rejected, diag = build_pair(rec, args.mode, args.score,
                                            args.min_margin, args.prec_floor)
        if chosen is None:
            drops[diag] += 1
            continue
        row = {
            "prompt": make_prompt(tok, rec["sentence"], rec["predicate"]),
            "chosen": chosen,
            "rejected": rejected,
            "group_id": rec["group_id"],
            "predicate": rec["predicate"],
            "margin": diag["margin"],
        }
        (val_rows if val_bucket(rec["group_id"], args.val_frac) else train_rows).append(row)
        margins.append(diag["margin"])
        dn.append(diag["worst_n_qa"] - diag["n_gold"])
        if "d_recall" in diag:
            d_rec.append(diag["d_recall"])
            d_prec.append(diag["d_precision"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("dpo_train.jsonl", train_rows), ("dpo_val.jsonl", val_rows)]:
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    under = sum(1 for d in dn if d < 0)
    over = sum(1 for d in dn if d > 0)
    print(f"mode={args.mode} score={args.score} min_margin={args.min_margin}")
    print(f"  train={len(train_rows)}  val={len(val_rows)}  dropped={sum(drops.values())} {dict(drops)}")
    if margins:
        print(f"  margin mean={st.mean(margins):.3f} median={st.median(margins):.3f}")
        print(f"  rejected vs gold arg-count: under={under} ({under/len(dn):.1%}) "
              f"over={over} ({over/len(dn):.1%}) mean_delta={st.mean(dn):+.2f}")
    if d_rec:
        # The signal check: dRecall must dominate dPrecision, or the pairs are
        # directionless in the same way the inherited 50/50 add/truncate set was.
        print(f"  chosen-vs-rejected: dRecall={st.mean(d_rec):+.3f} "
              f"dPrecision={st.mean(d_prec):+.3f} "
              f"(recall-dominance {st.mean(d_rec)/max(st.mean(d_prec), 1e-9):.1f}x)")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
