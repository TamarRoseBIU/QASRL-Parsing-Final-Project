"""
qasrl_inference_utils.py
------------------------
Shared inference/parsing helpers used by the DPO on-policy sample miner
(mine_onpolicy_samples.py) and the DPO checkpoint selector (eval_on_val.py):
`build_prompt`, `load_split`, `parse_completion`.

Run directly, its `main()` is a cheap, inference-only reward-signal diagnostic on
the FROZEN CE checkpoint (the GRPO init).

Question it answers, on the FROZEN CE checkpoint (the GRPO init):
  saturation: is there any within-group reward variance to learn from?
  recall ceiling: can the model even PRODUCE the missed arguments in
      *some* rollout (oracle union-recall) — i.e. is there headroom GRPO
      could reinforce, or is the policy fundamentally unable to cover them?

Reports, per split, for a random sample of prompts:
  - greedy   F1/P/R                         (matches the eval decoding)
  - sampled  mean F1  (temp=1.0, k rollouts)
  - mean WITHIN-GROUP std of F1            <-- GRPO advantage signal strength
  - % of groups fully collapsed (std==0)
  - best-of-k F1                            (upper bound of "pick the best rollout")
  - union-of-k  P/R/F1                      (recall ceiling of sampling)
  - parse-failure rate (empty parse)
  - mean #pred QAs (greedy) vs mean #gold QAs   (under-generation check)

Splits compared:
  dev         = current GRPO train set (CE was SFT'd on it — expect saturation)
  train_fresh = random sample of the unseen 92k train split (policy never saw it)

Usage (diagnostic mode):
  python qasrl_inference_utils.py --ckpt <CE_DIR> --n 150 --k 8
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import statistics as stats
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from qasrl_reward import qasrl_reward_full

# Same system prompt the CE + GRPO stages use (must stay in sync).
SYSTEM_PROMPT = (
    "You are a linguistic annotation assistant. Given a sentence and a target predicate, "
    "generate all QASRL question-answer pairs that describe the predicate's arguments.\n\n"
    "Rules:\n"
    "- Questions must follow QASRL style using WH-words (who, what, when, where, why, how, "
    "to whom, etc.) and include the target predicate. A question should represent a single "
    "semantic role of words or phrases in the sentence (e.g. Who did something?, When did "
    "someone do something?, etc.)\n"
    "- Answers must be exact verbatim spans from the sentence. The answer is an argument of "
    "the predicate (the someone/something who executed the action, or the time it was executed, etc.)\n"
    "- If multiple answer spans are valid for the same question, join them with the delimiter \" <A> \".\n"
    "- Generate a single Q&A pair for each semantic role expressed in the sentence. "
    "Separate distinct QA pairs with the delimiter \" <QA> \".\n"
    "- Cover all semantic roles explicitly expressed in the sentence, including agent, patient, "
    "recipient, instrument, location, time, manner, purpose, etc.\n"
    "- Do not invent semantic roles that are not grounded in the sentence.\n"
    "- Do not provide explanations, commentary, or formatting beyond the Q&A pairs.\n\n"
    "Examples:\n\n"
    "Prompt: \"Given the sentence: 'On Friday, Clark posted to Facebook to explain his decision.'\n"
    "Generate all QA pairs for the predicate 'posted'.\"\n"
    "Response: \"who posted something? Clark <QA> what did someone post? his decision <QA> "
    "where did someone post something? Facebook <QA> when did someone post something? On Friday\"\n\n"
    "Prompt: \"Given the sentence: 'As of 1:00 pm local time (0500 UTC), military investigators "
    "had not released the names of the deceased.'\n"
    "Generate all QA pairs for the predicate 'released'.\"\n"
    "Response: \"what hadn't someone released? the names of the deceased <QA> when hadn't someone "
    "released something? As of 1:00 pm local time (0500 UTC) <QA> who hadn't released something? "
    "military investigators\""
)

_URL = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/"
CACHE_DIR = Path(__file__).resolve().parent


def load_split(split_name: str) -> list[dict]:
    """Return list of {sentence, predicate, gold_qas:[{question,answer}]} groups."""
    cache = CACHE_DIR / f"{split_name}.json.cache"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        data = requests.get(_URL + f"{split_name}.json").json()
        try:
            cache.write_text(json.dumps(data))
        except Exception:
            pass

    hold: dict = {}
    for ex in data:
        s_id = ex["sentence_id"]
        pred = ex["predicate"]
        dt = ex["detokenized"]
        sentence = dt["sentence"]
        hold.setdefault(s_id, {"sentence": sentence})
        hold[s_id].setdefault(pred, [])
        for ans_text in dt["answers"]["text"]:
            hold[s_id][pred].append({"question": dt["question"], "answer": ans_text})

    groups = []
    for s_id, v in hold.items():
        sentence = v["sentence"]
        for pred, gold_qas in v.items():
            if pred == "sentence":
                continue
            groups.append({"sentence": sentence, "predicate": pred, "gold_qas": gold_qas})
    return groups


def build_prompt(tok, sentence, predicate) -> str:
    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Given the sentence: '{sentence}'\n"
                f"Generate all QA pairs for the predicate '{predicate}'."},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def parse_completion(text: str, eos: str) -> list[dict]:
    text = text.strip()
    if eos and text.endswith(eos):
        text = text[: -len(eos)].strip()
    qas = []
    for chunk in text.split("<QA>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        q_pos = chunk.find("?")
        if q_pos == -1:
            continue
        question = chunk[: q_pos + 1].strip()
        answer_part = chunk[q_pos + 1:].strip()
        for ans in answer_part.split("<A>"):
            ans = ans.strip()
            if ans:
                qas.append({"question": question, "answer": ans})
    return qas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
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

    import random
    for split in ["dev", "train"]:
        groups = load_split(split)
        rng = random.Random(args.seed)
        rng.shuffle(groups)
        sample = groups[: min(args.n, len(groups))]

        greedy_f1, greedy_p, greedy_r = [], [], []
        greedy_npred, gold_n = [], []
        group_means, group_stds, best_of_k, union_r, union_f1 = [], [], [], [], []
        n_collapsed = 0
        parse_fail = 0
        total_rollouts = 0

        with torch.no_grad():
            for row in sample:
                sentence, gold_qas = row["sentence"], row["gold_qas"]
                prompt = build_prompt(tok, sentence, row["predicate"])
                enc = tok(prompt, return_tensors="pt").to(model.device)
                plen = enc["input_ids"].shape[1]

                # ── greedy (matches eval decoding) ──
                g_out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, num_beams=1,
                    pad_token_id=tok.pad_token_id,
                )
                g_txt = tok.decode(g_out[0, plen:], skip_special_tokens=False)
                g_qas = parse_completion(g_txt, eos)
                gr = qasrl_reward_full(g_qas, gold_qas, sentence)
                greedy_f1.append(gr["f1"]); greedy_p.append(gr["precision"]); greedy_r.append(gr["recall"])
                greedy_npred.append(len(g_qas)); gold_n.append(len(gold_qas))

                # ── k sampled rollouts ──
                s_out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=0.95,
                    num_return_sequences=args.k,
                    pad_token_id=tok.pad_token_id,
                )
                completions = tok.batch_decode(s_out[:, plen:], skip_special_tokens=False)
                f1s, all_pred = [], []
                for c in completions:
                    pq = parse_completion(c, eos)
                    total_rollouts += 1
                    if not pq:
                        parse_fail += 1
                    all_pred.extend(pq)
                    f1s.append(qasrl_reward_full(pq, gold_qas, sentence)["f1"])

                group_means.append(stats.mean(f1s))
                g_std = stats.pstdev(f1s)
                group_stds.append(g_std)
                if g_std == 0.0:
                    n_collapsed += 1
                best_of_k.append(max(f1s))
                u = qasrl_reward_full(all_pred, gold_qas, sentence)  # union of all rollouts
                union_r.append(u["recall"]); union_f1.append(u["f1"])

        def m(x):
            return stats.mean(x) if x else 0.0

        print("\n" + "=" * 72)
        print(f"SPLIT={split}  n_prompts={len(sample)}  k={args.k}")
        print("-" * 72)
        print(f"  GREEDY  F1={m(greedy_f1):.4f}  P={m(greedy_p):.4f}  R={m(greedy_r):.4f}")
        print(f"  greedy #pred QAs={m(greedy_npred):.2f}   gold #QAs={m(gold_n):.2f}  "
              f"(<-- under-generation if pred<gold)")
        print(f"  SAMPLED mean F1               = {m(group_means):.4f}")
        print(f"  mean WITHIN-GROUP std of F1   = {m(group_stds):.4f}   <-- GRPO signal strength")
        print(f"  %% groups collapsed (std==0)   = {100*n_collapsed/len(sample):.1f}%")
        print(f"  BEST-of-{args.k} F1              = {m(best_of_k):.4f}   "
              f"(headroom vs greedy = {m(best_of_k)-m(greedy_f1):+.4f})")
        print(f"  UNION-of-{args.k} recall         = {m(union_r):.4f}   "
              f"(recall ceiling; greedy R={m(greedy_r):.4f})")
        print(f"  UNION-of-{args.k} F1             = {m(union_f1):.4f}")
        print(f"  parse-failure rate            = {100*parse_fail/max(total_rollouts,1):.1f}%")
        print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
