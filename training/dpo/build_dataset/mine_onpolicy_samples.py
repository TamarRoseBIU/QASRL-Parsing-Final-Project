"""
On-policy sample mining for DPO preference pairs.

WHY:
Synthetic, off-policy pairs are a weaker signal — `chosen` is verbatim dev gold
(which SFT already maximized the likelihood of, on these exact groups) and `rejected`
is a mechanical edit of that gold. The model's real failure mode, under-generating
adjunct arguments under greedy decoding, never appears in a pair. Part I Exp 1 showed
the SFT policy's OWN samples carry large gold-scoreable spread (within-group F1 std
~0.16, union-of-8 recall 0.979). This script harvests that spread.

It only SAMPLES AND SCORES. Pair construction is a separate CPU step
(`build_onpolicy_pairs.py`) so that pairing rules — margin thresholds, beta, best-vs-worst
vs best-vs-median — can be swept without paying for GPU sampling again.

Scores every sample with qasrl_reward_full at BOTH beta=1.0 and beta=2.0 (plus raw
P/R/tp/fp/fn), so the recall-weighting decision is made downstream from one sampling run.

Sharding: --shard_idx / --n_shards partitions dev deterministically so the ~2,400 groups
fit under the 4h SLURM wall across two concurrent jobs.

Reads ONLY the dev split. test.json is never opened.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qasrl_inference_utils import (
    SYSTEM_PROMPT,      # noqa: F401  (imported for provenance/consistency)
    build_prompt,
    load_split,
    parse_completion,
)
from qasrl_reward import qasrl_reward_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SFT (CE) LoRA adapter dir")
    ap.add_argument("--base_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--split", default="dev", choices=["dev"],
                    help="dev only; the test split is never sampled.")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.split == "dev", "only dev may be sampled"
    torch.manual_seed(args.seed + 1000 * args.shard_idx)

    groups = load_split(args.split)
    # Deterministic, model-independent sharding by position in the stable load order.
    shard = [g for i, g in enumerate(groups) if i % args.n_shards == args.shard_idx]
    print(f"[shard {args.shard_idx}/{args.n_shards}] {len(shard)} of {len(groups)} dev groups",
          flush=True)

    print(f"Loading tokenizer/model from {args.ckpt} …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.ckpt, is_trainable=False)
    model.eval()
    eos = tok.eos_token

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_written = 0

    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(shard):
            sentence, predicate, gold_qas = row["sentence"], row["predicate"], row["gold_qas"]
            prompt = build_prompt(tok, sentence, predicate)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            plen = enc["input_ids"].shape[1]

            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                num_return_sequences=args.k,
                pad_token_id=tok.pad_token_id,
            )
            completions = tok.batch_decode(out[:, plen:], skip_special_tokens=False)

            samples = []
            for c in completions:
                text = c[: -len(eos)].strip() if eos and c.strip().endswith(eos) else c.strip()
                qas = parse_completion(c, eos)
                # One call: evaluate_predicate returns f1 AND fbeta from the same
                # precision/recall, so beta=2.0 gives both without re-running the
                # (expensive) span matching.
                r = qasrl_reward_full(qas, gold_qas, sentence, beta=2.0)
                samples.append({
                    "text": text,
                    "n_qa": len(qas),
                    "parse_ok": bool(qas),
                    "f1": r["f1"], "precision": r["precision"], "recall": r["recall"],
                    "fbeta2": r["fbeta"],
                    "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                })

            fh.write(json.dumps({
                # load_split() yields no sentence_id, so derive a stable,
                # model-independent group key for the val holdout downstream.
                "group_id": hashlib.md5(
                    f"{sentence}::{predicate}".encode("utf-8")).hexdigest(),
                "sentence": sentence,
                "predicate": predicate,
                "gold_qas": gold_qas,
                "n_gold": len(gold_qas),
                "samples": samples,
            }, ensure_ascii=False) + "\n")
            n_written += 1

            if (i + 1) % 50 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                print(f"  {i+1}/{len(shard)} groups | {el/60:.1f} min | "
                      f"eta {(len(shard)-i-1)/rate/60:.1f} min", flush=True)

    print(f"[shard {args.shard_idx}] wrote {n_written} groups -> {out_path} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
