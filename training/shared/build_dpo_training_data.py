import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG -- keep in sync with Stage_CE_Instruct_DEV.py / Stage_GRPO_Instruct_DEV.py
# ----------------------------------------------------------------------------

# These four constants are copied VERBATIM from Stage_GRPO_Instruct_DEV.py's
# own CONFIG section (BASE_DIR / MODEL / CE_RUN_NAME / PREV_STAGE), so that
# DEFAULT_TOKENIZER_DIR is always the SAME checkpoint GRPO itself loads its
# tokenizer from (CE_MODEL_DIR there) -- computed the same way, not just a
# string that happens to match today. If those change in the GRPO script,
# update them here too (keep in sync, per the CONFIG header note above).
BASE_DIR    = os.environ.get(
    "QASRL_BASE_DIR", str(Path(__file__).resolve().parents[2] / "runs"))
MODEL       = "Qwen3-30B-A3B-Instruct-2507"
CE_RUN_NAME = f"{MODEL}_train_dev_val_test"
PREV_STAGE  = "Stage_CE"

# The SFT (CE) adapter directory, used here only for its tokenizer. It is NOT
# distributable (multi-GB LoRA adapter on lab model storage) -- set QASRL_SFT_ADAPTER
# to your own SFT adapter dir, or leave it to fall back to the BASE_DIR layout.
# QASRL_SFT_ADAPTER is the single place every stage resolves the adapter from.
DEFAULT_TOKENIZER_DIR = os.environ.get(
    "QASRL_SFT_ADAPTER",
    f"{BASE_DIR}/models_save_baseline/{PREV_STAGE}/{CE_RUN_NAME}")

# Where build_dpo_data.py / build_dpo_data_test.py / llm_fix_grammar.py actually
# write their output (see those scripts' OUTPUT_PATH / --output defaults).
DATASET_DIR = os.environ.get(
    "QASRL_DATASET_DIR", str(Path(__file__).resolve().parents[2] / "data" / "processed"))
DEFAULT_TRAIN_INPUT = f"{DATASET_DIR}/dpo_pairs.grammar_fixed.manual_fixed.json"       # dev, grammar-fixed
DEFAULT_EVAL_INPUT = f"{DATASET_DIR}/dpo_pairs_test.grammar_fixed.json"  # test, grammar-fixed

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
    "Prompt: \"As of 1:00 pm local time (0500 UTC), military investigators "
    "had not released the names of the deceased.'\n"
    "Generate all QA pairs for the predicate 'released'.\"\n"
    "Response: \"what hadn't someone released? the names of the deceased <QA> when hadn't someone "
    "released something? As of 1:00 pm local time (0500 UTC) <QA> who hadn't released something? "
    "military investigators\""
)

QA_DELIM = " <QA> "
ANS_DELIM = " <A> "

LOWERCASE_QUESTIONS_FOR_TRAINING = True


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def decapitalize_question(question: str) -> str:
    for i, ch in enumerate(question):
        if ch.isalpha():
            return question[:i] + ch.lower() + question[i + 1:]
    return question


def render_completion(qa_pairs: list[dict]) -> str:
    chunks = []
    for qa in qa_pairs:
        question = qa["question"]
        if LOWERCASE_QUESTIONS_FOR_TRAINING:
            question = decapitalize_question(question)
        answers = [a for a in qa.get("answers", []) if a and a.strip()]
        if not answers:
            continue
        chunks.append(f"{question} {ANS_DELIM.join(answers)}")
    return QA_DELIM.join(chunks)


def make_prompt(tok, sentence: str, predicate: str) -> str:
    return tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Given the sentence: '{sentence}'\n"
                f"Generate all QA pairs for the predicate '{predicate}'."},
        ],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def process_records(records: list[dict], tokenizer, skip_needs_review: bool) -> list[dict]:
    """Processes raw records into tokenized prompt/chosen/rejected rows."""
    n_skipped_review = 0
    n_skipped_empty = 0
    rows = []
    
    for rec in records:
        if skip_needs_review and rec.get("needs_review"):
            n_skipped_review += 1
            continue

        chosen = render_completion(rec["accepted"])
        rejected = render_completion(rec["rejected"])

        if not chosen or not rejected:
            n_skipped_empty += 1
            continue

        if chosen == rejected:
            n_skipped_empty += 1
            continue

        prompt = make_prompt(tokenizer, rec["sentence_text"], rec["predicate"])

        rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "sentence_id": rec["sentence_id"],
            "predicate": rec["predicate"],
            "group_id": f"{rec['sentence_id']}::{rec['predicate']}",
            "process": rec["process"],
            "needs_review": bool(rec.get("needs_review", False)),
        })
        
    if n_skipped_review or n_skipped_empty:
        log.info(f"Filtered out {n_skipped_review} flagged review items and {n_skipped_empty} empty/identical pairs.")
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input", type=str, default=DEFAULT_TRAIN_INPUT,
        help="Path to the training data pairs. Defaults to the dev-set pairs "
             f"(grammar-fixed) produced by build_dpo_data.py + llm_fix_grammar.py: "
             f"{DEFAULT_TRAIN_INPUT}"
    )
    ap.add_argument(
        "--eval-input", type=str, default=DEFAULT_EVAL_INPUT,
        help="Path to evaluation data pairs. Defaults to the test-set pairs "
             f"(grammar-fixed) produced by build_dpo_data_test.py + the test-set "
             f"grammar-fix job: {DEFAULT_EVAL_INPUT}. 100%% of --input goes to "
             "train and 100%% of this file goes to eval. Pass --eval-input '' "
             "(empty string) to fall back to carving --eval-fraction off of "
             "--input instead."
    )
    ap.add_argument(
        "--eval-fraction", type=float, default=0.1,
        help="Fraction of --input to carve off for eval ONLY if --eval-input is not provided."
    )
    ap.add_argument("--tokenizer-dir", type=str, default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument("--output-dir", type=str, default="./datasets")
    ap.add_argument(
        "--skip-needs-review", action="store_true",
        help="Drop records still flagged needs_review=True."
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Primary input path {args.input} does not exist. If this is the "
            f"default dev-set path, run build_dpo_data.py and then "
            f"run_grammar_fix.sh first."
        )

    log.info(f"Loading tokenizer from {args.tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    # 1. Process Training Set
    log.info(f"Reading training pairs from {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    train_rows = process_records(train_records, tokenizer, args.skip_needs_review)

    # 2. Process or Split Evaluation Set
    if args.eval_input:
        eval_path = Path(args.eval_input)
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Evaluation input path {args.eval_input} does not exist. "
                f"If this is the default test-set path, run build_dpo_data_test.py "
                f"and then the test-set grammar-fix job (run_grammar_fix_test.sh) "
                f"first -- or pass --eval-input '' to fall back to splitting --input "
                f"instead."
            )
        
        log.info(f"Reading evaluation pairs from {eval_path} ...")
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_records = json.load(f)
        eval_rows = process_records(eval_records, tokenizer, args.skip_needs_review)
        log.info(f"Explicit evaluation file provided. Train rows: {len(train_rows)}, Eval rows: {len(eval_rows)}")
    else:
        # Fall back to original behavior (splitting the single input file)
        log.info(f"No --eval-input provided. Splitting training data using fraction {args.eval_fraction}...")
        train_rows.sort(key=lambda r: r["group_id"])
        n_eval = max(1, int(len(train_rows) * args.eval_fraction)) if train_rows else 0
        eval_rows = train_rows[-n_eval:] if n_eval else []
        train_rows = train_rows[:-n_eval] if n_eval else train_rows
        log.info(f"Split results -> Train rows: {len(train_rows)}, Eval rows: {len(eval_rows)}")

    # 3. Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "dpo_train.jsonl"
    eval_path = out_dir / "dpo_eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Successfully wrote training data to -> {train_path}")
    log.info(f"Successfully wrote evaluation data to -> {eval_path}")


if __name__ == "__main__":
    main()