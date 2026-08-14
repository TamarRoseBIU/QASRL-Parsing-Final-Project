"""
qasrl_reward.py
---------------
Fast, self-contained F1 reward function for GRPO training of a QASRL model.

Spans are computed the same way as run_qwen3_base_inference.py:
    normalize_text(sentence + answer)  →  tokenize  →  find_answer_range

Input format
------------
Both `pred_qas` and `gold_qas` are plain dicts:

    {
        "question": str,   # e.g. "what does someone study?"
        "answer":   str,   # raw answer text, e.g. "how rocks and minerals form"
    }

The sentence is passed separately so spans can be resolved the same way
inference resolves them.
"""

from __future__ import annotations
import re
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Span   = Tuple[int, int]   # (token_start, token_end)  — end is exclusive
QAPair = Dict              # {"question": str, "answer": str}

MATCH_IOU_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Text normalisation & tokenisation
# (matches the answer-splitting rule used by the inference scripts)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return text.split()


def find_answer_range(
    sentence_tokens: List[str],
    answer_tokens: List[str],
) -> Optional[Span]:
    """
    Find the first occurrence of answer_tokens inside sentence_tokens.
    Returns (start, end) where end is exclusive, or None if not found.
    """
    if not answer_tokens:
        return None
    n = len(answer_tokens)
    for i in range(len(sentence_tokens) - n + 1):
        if sentence_tokens[i : i + n] == answer_tokens:
            return (i, i + n)
    return None


def answer_to_span(sentence_tokens: List[str], answer: str) -> Optional[Span]:
    """Normalise an answer string and find its token span in the sentence."""
    answer_tokens = tokenize(normalize_text(answer))
    return find_answer_range(sentence_tokens, answer_tokens)


# ---------------------------------------------------------------------------
# Span IoU
# ---------------------------------------------------------------------------

def _iou(a: Span, b: Span) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union


# ---------------------------------------------------------------------------
# Lightweight one-to-one bipartite matching (no networkx needed)
# ---------------------------------------------------------------------------

def _max_weight_matching(
    edges: List[Tuple[int, int, float]],
) -> Dict[int, int]:
    """Greedy max-weight one-to-one matching. Fast enough for <=~10 nodes."""
    used_sys: set = set()
    used_grt: set = set()
    mapping: Dict[int, int] = {}
    for s, g, _ in sorted(edges, key=lambda e: e[2], reverse=True):
        if s not in used_sys and g not in used_grt:
            mapping[s] = g
            used_sys.add(s)
            used_grt.add(g)
    return mapping


# ---------------------------------------------------------------------------
# Default question similarity: token-level F1 (fast, no model needed)
# ---------------------------------------------------------------------------

def _token_f1(q1: str, q2: str) -> float:
    t1, t2 = set(q1.lower().split()), set(q2.lower().split())
    if not t1 or not t2:
        return 0.0
    common = t1 & t2
    p = len(common) / len(t2)
    r = len(common) / len(t1)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _questions_match(
    q1: str,
    q2: str,
    question_match_fn,
    question_match_threshold: float,
) -> bool:
    if question_match_fn is not None:
        return bool(question_match_fn(q1, q2))
    return _token_f1(q1, q2) >= question_match_threshold


# ---------------------------------------------------------------------------
# Core per-predicate evaluation
# ---------------------------------------------------------------------------

def evaluate_predicate(
    pred_qas: List[QAPair],
    gold_qas: List[QAPair],
    sentence: str,
    *,
    iou_threshold: float = MATCH_IOU_THRESHOLD,
    question_match_fn=None,
    question_match_threshold: float = 0.5,
    return_all: bool = False,
    beta: float = 1.0,
):
    """
    Compute labeled-argument F1 for one predicate's QA pairs.

    Parameters
    ----------
    pred_qas : list of {"question", "answer"}
        Model predictions for this predicate.
    gold_qas : list of {"question", "answer"}
        Ground-truth annotations for this predicate.
    sentence : str
        The raw sentence used to resolve answer token spans, exactly as in
        run_qwen3_base_inference.py: normalize -> tokenize -> find_answer_range.
    iou_threshold : float
        Minimum token-span IoU for a candidate match (default 0.3).
    question_match_fn : callable(str, str) -> bool, optional
        Your paraphrase scorer. If None, falls back to token-F1 >= threshold.
    question_match_threshold : float
        Token-F1 cutoff when question_match_fn is None (default 0.5).
    return_all : bool
        True  -> dict with precision, recall, f1, tp, fp, fn.
        False -> just the f1 float (default, suitable for GRPO reward).
    """
    def _result(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        # F_beta (beta>1 weights recall more). beta=1.0 -> identical to f1.
        b2 = beta * beta
        fbeta = ((1 + b2) * precision * recall / (b2 * precision + recall)
                 if (b2 * precision + recall) > 0 else 0.0)
        if return_all:
            return {"precision": precision, "recall": recall, "f1": f1,
                    "fbeta": fbeta, "tp": tp, "fp": fp, "fn": fn}
        # When beta != 1 the scalar reward is F_beta; else plain F1 (unchanged).
        return fbeta if beta != 1.0 else f1

    # ── Edge cases ────────────────────────────────────────────────────────────
    if not gold_qas:
        return _result(0, len(pred_qas), 0)
    if not pred_qas:
        return _result(0, 0, len(gold_qas))

    # ── Resolve token spans from the sentence ─────────────────────────────────
    sentence_tokens = tokenize(normalize_text(sentence))

    pred_spans = [answer_to_span(sentence_tokens, qa["answer"]) for qa in pred_qas]
    gold_spans = [answer_to_span(sentence_tokens, qa["answer"]) for qa in gold_qas]

    # ── Candidate pairs with IoU >= threshold ─────────────────────────────────
    candidate_edges = []
    for pi, ps in enumerate(pred_spans):
        if ps is None:          # answer not found in sentence -> will be FP
            continue
        for gi, gs in enumerate(gold_spans):
            if gs is None:
                continue
            score = _iou(ps, gs)
            if score >= iou_threshold:
                candidate_edges.append((pi, gi, score))

    # ── Strict one-to-one matching (only optimal winners count as matched) ─────
    span_mapping = _max_weight_matching(candidate_edges)
    # span_mapping: pred_idx -> gold_idx

    # ── TP / FP / FN counts ───────────────────────────────────────────────────
    n_tp = len(span_mapping)
    n_fp = len(pred_qas) - n_tp    # unmatched preds (includes spans not found in sentence)
    n_fn = len(gold_qas) - n_tp    # unmatched gold

    # commented out for unlabeled evaluation
    # # ── Question label check on matched spans ─────────────────────────────────
    # for pred_idx, gold_idx in span_mapping.items():
    #     pred_q = pred_qas[pred_idx]["question"]
    #     gold_q = gold_qas[gold_idx]["question"]
    #     if not _questions_match(pred_q, gold_q, question_match_fn, question_match_threshold):
    #         # Span matched but wrong question -> flip TP to FP + FN
    #         n_tp -= 1
    #         n_fp += 1
    #         n_fn += 1

    return _result(n_tp, n_fp, n_fn)


# ---------------------------------------------------------------------------
# GRPO reward wrapper
# ---------------------------------------------------------------------------

def qasrl_reward(
    pred_qas: List[QAPair],
    gold_qas: List[QAPair],
    sentence: str,
    question_match_fn=None,
) -> float:
    """
    Drop-in GRPO reward. Returns labeled-argument F1 in [0, 1].

    Usage:
        from paraphrases import get_paraphrase_score
        reward = qasrl_reward(pred, gold, sentence,
                              question_match_fn=get_paraphrase_score)
    """
    return evaluate_predicate(pred_qas, gold_qas, sentence,
                               question_match_fn=question_match_fn)


def qasrl_reward_full(
    pred_qas: List[QAPair],
    gold_qas: List[QAPair],
    sentence: str,
    question_match_fn=None,
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    Same matching logic as qasrl_reward(), but returns the full breakdown:
        {"precision": float, "recall": float, "f1": float,
         "tp": int, "fp": int, "fn": int}

    Use this when you need precision/recall (e.g. for logging/plots)
    alongside the scalar reward, without computing the match twice.
    """
    return evaluate_predicate(pred_qas, gold_qas, sentence,
                               question_match_fn=question_match_fn,
                               return_all=True, beta=beta)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sentence = "Geologists study how rocks and minerals form."

    gold = [
        {"question": "what does someone study?",
         "answer":   "how rocks and minerals form"},
    ]

    cases = {
        "perfect":      [{"question": "what does someone study?",
                          "answer":   "how rocks and minerals form"}],
        "wrong_q":      [{"question": "who studies something?",
                          "answer":   "how rocks and minerals form"}],
        "partial_span": [{"question": "what does someone study?",
                          "answer":   "rocks and minerals form"}],
        "not_in_sent":  [{"question": "what does someone study?",
                          "answer":   "quantum physics"}],
        "empty_pred":   [],
    }

    for label, pred in cases.items():
        result = evaluate_predicate(pred, gold, sentence, return_all=True)
        print(f"{label:14s}  F1={result['f1']:.3f}  "
              f"P={result['precision']:.3f}  R={result['recall']:.3f}  "
              f"tp={result['tp']} fp={result['fp']} fn={result['fn']}")

    # Sanity check: qasrl_reward_full must agree with evaluate_predicate(return_all=True)
    full = qasrl_reward_full(cases["perfect"], gold, sentence)
    assert full == evaluate_predicate(cases["perfect"], gold, sentence, return_all=True)
    print("\nqasrl_reward_full() OK:", full)