"""
Checkpoint selection on the held-out DEV slice.

General-purpose PEFT-checkpoint selector — it has no DPO-specific internals; it
lives here because DPO checkpoint selection on the dev holdout is what it is used
for. Any list of adapter directories can be passed to --ckpts.

Greedy-decodes (do_sample=False, matching the real eval) every group in the
validation holdout and scores it with the SAME matching rule as the headline
metric (qasrl_reward, IoU>=0.3, one-to-one).

Reports MICRO F1 (global tp/fp/fn) as the selection statistic, because the
headline test metric is micro-averaged — you select on the statistic you report.
Macro is printed alongside for reference only.

The holdout group ids come from dpo_val.jsonl; the sentences and gold QAs are
re-derived from the dev split by recomputing the same md5 group key, so gold is
never duplicated into the pairs file and cannot drift out of sync.

Usage:
  python eval_on_val.py --ckpts DIR [DIR ...] --val dpo_val.jsonl --out sel.json

test.json is never read.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qasrl_inference_utils import build_prompt, load_split, parse_completion
from qasrl_reward import qasrl_reward_full


def gid(sentence: str, predicate: str) -> str:
    return hashlib.md5(f"{sentence}::{predicate}".encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wanted = set()
    with open(args.val, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                wanted.add(json.loads(line)["group_id"])
    groups = [g for g in load_split("dev") if gid(g["sentence"], g["predicate"]) in wanted]
    print(f"validation holdout: {len(groups)} groups (from {len(wanted)} ids)", flush=True)
    if not groups:
        raise SystemExit("no validation groups resolved — group_id scheme mismatch?")

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation="sdpa",
    )

    results = []
    for ckpt in args.ckpts:
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = PeftModel.from_pretrained(base, ckpt, is_trainable=False)
        model.eval()
        eos = tok.eos_token

        TP = FP = FN = 0
        macro = []
        with torch.no_grad():
            for row in groups:
                enc = tok(build_prompt(tok, row["sentence"], row["predicate"]),
                          return_tensors="pt").to(model.device)
                plen = enc["input_ids"].shape[1]
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=tok.pad_token_id)
                qas = parse_completion(tok.decode(out[0, plen:], skip_special_tokens=False), eos)
                r = qasrl_reward_full(qas, row["gold_qas"], row["sentence"])
                TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
                macro.append(r["f1"])

        p = TP / (TP + FP) if TP + FP else 0.0
        rc = TP / (TP + FN) if TP + FN else 0.0
        micro = 2 * p * rc / (p + rc) if p + rc else 0.0
        res = {"ckpt": ckpt, "val_micro_f1": micro, "val_precision": p, "val_recall": rc,
               "val_macro_f1": sum(macro) / len(macro), "tp": TP, "fp": FP, "fn": FN,
               "n_groups": len(groups)}
        results.append(res)
        print(f"  {Path(ckpt).name}: micro_F1={micro:.4f} P={p:.4f} R={rc:.4f} "
              f"(macro={res['val_macro_f1']:.4f})", flush=True)

        model.unload()
        del model

    results.sort(key=lambda r: -r["val_micro_f1"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSELECTED: {results[0]['ckpt']}  val_micro_F1={results[0]['val_micro_f1']:.4f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
