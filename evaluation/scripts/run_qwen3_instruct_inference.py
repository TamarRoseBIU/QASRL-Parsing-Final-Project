#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen3-30B-A3B-Instruct variant.
Loads LoRA adapters saved by Stage_CE_Instruct_TRAIN.py via AutoPeftModelForCausalLM,
or falls back to a plain AutoModelForCausalLM if no adapter_config.json is found.

Usage:
    python run_qwen3_baseline_inference_Instruct.py \
        --model_path /path/to/models_save_baseline/Stage_CE/<run_name> \
        --input <in.csv> --output <out.csv>

model_path should be the MODEL_SAVE_DIR produced by the training script,
or a HuggingFace model ID (e.g. "Qwen/Qwen3-30B-A3B-Instruct").
"""

import os
import re
import csv
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================================
# Load Model
# ============================================================================
def load_model(model_path: str):
    """
    Load the Qwen3-Instruct model and tokenizer.

    If model_path contains an adapter_config.json (i.e. it is a LoRA checkpoint
    saved by the training script), the model is loaded with AutoPeftModelForCausalLM
    and the adapter weights are merged into the base model before inference.
    Otherwise a plain AutoModelForCausalLM is used (e.g. for the raw HF model).
    """
    logging.info(f"Loading Instruct model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    is_peft = Path(model_path).joinpath("adapter_config.json").exists()

    if is_peft:
        logging.info("adapter_config.json detected — loading as PEFT/LoRA model and merging weights.")
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = model.merge_and_unload()  # fuse LoRA deltas into base weights
    else:
        logging.info("No adapter_config.json found — loading as a standard model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    model.eval()

    # Force-disable thinking mode at the generation config level.
    # apply_chat_template(enable_thinking=False) only controls prompt formatting;
    # the saved generation_config.json may still have thinking enabled, causing
    # the model to run a hidden reasoning pass that produces paraphrased answers
    # instead of verbatim spans. Patching here covers both the merged and plain paths.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.update({"thinking": False})
        logging.info("Patched generation_config: thinking=False")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Model loaded. Primary device: {device}")

    return model, tokenizer, device


# ============================================================================
# Input Formatting (Match Training Format)
# ============================================================================
def format_input(sentence: str, predicate: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": (
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
            ),
        },
        {
            "role": "user",
            "content": f"Given the sentence: '{sentence}'\nGenerate all QA pairs for the predicate '{predicate}'.",
        },
    ]
    # Instruct variant: enable_thinking=False suppresses the <think>...</think>
    # block. This must match how the model was fine-tuned (training script also
    # passes enable_thinking=False). If thinking mode was used during training,
    # set this to True here as well.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ============================================================================
# Text Normalization and Range Calculation
# ============================================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return text.split()


def find_answer_range(
    sentence_tokens: List[str], answer_tokens: List[str]
) -> Optional[Tuple[int, int]]:
    if not answer_tokens:
        return None

    answer_len = len(answer_tokens)
    for i in range(len(sentence_tokens) - answer_len + 1):
        if sentence_tokens[i:i + answer_len] == answer_tokens:
            return (i, i + answer_len)

    return None


def process_multi_answer(
    answer_text: str, sentence_tokens: List[str]
) -> Tuple[Optional[str], List[dict]]:
    """
    Map each answer span to a token range in the (normalised) sentence.

    Answers arrive joined by DELIMITER. Returns (range_string, failed_list).
    range_string is None only when ALL answers fail — partial failures are
    logged but valid spans are still kept, so a single un-locatable span
    doesn't discard the whole QA pair.
    """
    DELIMITER = "~!~"
    answers = answer_text.split(DELIMITER)
    answer_ranges = []
    failed_answers = []

    for answer in answers:
        norm_answer = normalize_text(answer)
        answer_tokens = tokenize(norm_answer)
        answer_range = find_answer_range(sentence_tokens, answer_tokens)

        if answer_range is None:
            failed_answers.append({
                'original': answer,
                'normalized': norm_answer,
                'tokens': answer_tokens
            })
            answer_ranges.append(None)
        else:
            answer_ranges.append(f"{answer_range[0]}:{answer_range[1]}")

    # Replace every failed span with the invalid placeholder so the number of
    # range entries always matches the number of answers (e.g. "0:1~!~999:1998").
    # Return None only when every single span failed.
    INVALID = "999:1998"
    final_ranges = [r if r is not None else INVALID for r in answer_ranges]

    if all(r == INVALID for r in final_ranges):
        return None, failed_answers

    return DELIMITER.join(final_ranges), failed_answers


# ============================================================================
# QA Parsing
# ============================================================================
def parse_qa_output(generated_text: str) -> List[Dict[str, any]]:
    """
    Parse the generated QA pairs from model output.
    Format: "question? answer <QA> question? answer <A> alt_answer"

    For the Instruct variant, we also strip any residual thinking-block text
    (e.g. "<think>...</think>") that may appear if thinking mode leaks through,
    before parsing the QA pairs.
    """
    # Strip any residual <think>...</think> block just in case
    generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

    qa_pairs = []
    segments = generated_text.split("<QA>")

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        match = re.match(r'^(.+?\?)\s*(.+?)$', segment, re.DOTALL)

        if match:
            question = match.group(1).strip()
            answer_text = match.group(2).strip()
        elif '?' in segment:
            parts = segment.split('?', 1)
            question = parts[0].strip() + "?"
            answer_text = parts[1].strip()
        else:
            continue

        # NOTE: do NOT rstrip('.') here — answers are verbatim spans and may
        # legitimately end with punctuation (e.g. "Inc.", "U.S.", "etc.").
        # Stripping it causes find_answer_range to fail on valid answers.
        answers = [ans.strip() for ans in answer_text.split("<A>")]
        answers = [ans for ans in answers if ans]

        if answers:
            qa_pairs.append({'question': question, 'answers': answers})

    return qa_pairs


# ============================================================================
# Inference
# ============================================================================
def generate_qas_for_predicate(
    model,
    tokenizer,
    device: str,
    sentence: str,
    predicate: str,
    max_new_tokens: int = 512
) -> str:
    """
    Generate QA pairs for a given sentence and predicate.

    FIX (Instruct): Qwen3-Instruct chat template uses <|im_end|> as the
    assistant turn stop token, which is DIFFERENT from tokenizer.eos_token_id
    (<|endoftext|>). Both must be passed to eos_token_id, otherwise the model
    never stops and loops until max_new_tokens is exhausted.

    For the Instruct variant we also pass temperature=None and top_p=None to
    ensure fully deterministic greedy output (the Instruct model's default
    generation config may otherwise activate sampling).

    NOTE: repetition_penalty is intentionally omitted. On a span-extraction task
    the model must copy tokens verbatim from the input sentence; any penalty on
    repeated tokens actively suppresses exact-span answers and causes hallucination.

    NOTE: After loading, we patch generation_config to force-disable thinking mode
    at the generation level, not just at the prompt-template level. This prevents
    the Instruct model from running a hidden reasoning pass that produces paraphrased
    rather than verbatim answers.
    """
    input_text = format_input(sentence, predicate, tokenizer)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        add_special_tokens=False
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    # Resolve both stop token IDs
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    stop_ids = list({eos_id, im_end_id})  # deduplicate in case they're equal

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=None,         # must be None when do_sample=False
            top_p=None,               # must be None when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    generated_ids = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ============================================================================
# CSV Processing
# ============================================================================
def read_input_csv(input_path: str) -> List[Dict[str, str]]:
    logging.info(f"Reading input from {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        required_cols = {'sentence_id', 'predicate_idx', 'predicate', 'sentence'}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"Input CSV missing required columns. "
                f"Expected: {required_cols}, Found: {reader.fieldnames}"
            )

        for row in reader:
            rows.append(row)

    logging.info(f"Read {len(rows)} rows from input CSV")
    return rows


def process_input_row(
    model,
    tokenizer,
    device: str,
    row: Dict[str, str]
) -> List[Dict[str, str]]:
    sentence_id = row['sentence_id']
    predicate_idx = row['predicate_idx']
    predicate = row['predicate']
    sentence = row['sentence']

    norm_sentence = normalize_text(sentence)
    sentence_tokens = tokenize(norm_sentence)

    logging.info(f"Processing: {sentence_id} | predicate='{predicate}' (idx={predicate_idx})")

    results = []

    generated_text = generate_qas_for_predicate(
        model, tokenizer, device, sentence, predicate
    )
    logging.info(f"  Generated: {generated_text}")

    qa_pairs = parse_qa_output(generated_text)
    logging.info(f"  Extracted {len(qa_pairs)} QA pairs")

    for qa in qa_pairs:
        qasrl_id = f"{sentence_id}"

        combined_answer = "~!~".join(qa['answers'])
        answer_ranges_str, failed = process_multi_answer(combined_answer, sentence_tokens)

        if answer_ranges_str is None:
            logging.warning(f"  Could not find range(s) for answer(s) in '{sentence_id}':")
            for f_ans in failed:
                logging.warning(
                    f"    answer='{f_ans['original']}' -> normalized='{f_ans['normalized']}' "
                    f"tokens={f_ans['tokens']}"
                )
            answer_ranges_str = "N/A"

        results.append({
            'qasrl_id': qasrl_id,
            'verb_idx': predicate_idx,
            'verb': predicate,
            'question': qa['question'],
            'answer_range': answer_ranges_str,
            'answer': combined_answer
        })

    return results


def write_output_csv(results: List[Dict[str, str]], output_path: str):
    logging.info(f"Writing {len(results)} records to {output_path}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['qasrl_id', 'verb_idx', 'verb', 'question', 'answer_range', 'answer']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"✓ Results saved to {output_path}")


# ============================================================================
# Main
# ============================================================================
def run_inference(model_path: str, input_path: str, output_path: str):
    model, tokenizer, device = load_model(model_path)

    input_rows = read_input_csv(input_path)
    all_results = []
    failed_rows = []
    INVALID_PLACEHOLDER = "999:1998"

    for i, row in enumerate(input_rows, 1):
        logging.info(f"\n{'='*80}\nProcessing row {i}/{len(input_rows)}\n{'='*80}")
        results = process_input_row(model, tokenizer, device, row)

        for res in results:
            if res['answer_range'] == "N/A":
                res['answer_range'] = INVALID_PLACEHOLDER
                failed_rows.append(res)

            all_results.append(res)

    write_output_csv(all_results, output_path)

    if failed_rows:
        print(f"\n{'!'*20} FAILED TO FIND RANGES {'!'*20}")
        for fail in failed_rows:
            print(f"ID: {fail['qasrl_id']} | Verb: {fail['verb']} | Question: {fail['question']} | Answer: {fail['answer']}")

    logging.info(f"Inference complete! Saved {len(all_results)} records. Excluded {len(failed_rows)} N/A records.")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-Instruct QASRL inference")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model dir (MODEL_SAVE_DIR from training) or a HF model ID')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV')

    args = parser.parse_args()

    try:
        run_inference(args.model_path, args.input, args.output)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()