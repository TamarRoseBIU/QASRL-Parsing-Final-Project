"""
Stage 1: Cross-Entropy (CE) SFT for QA-SRL (Stage_CE_Instruct_DEV.py)
---------------------------------------------------------------------
Fine-tunes the base Qwen model using Supervised Fine-Tuning (CE loss) on QA-SRL data.
Serves as the warm-start baseline adapter for downstream RL stages (GRPO / DPO).

Key Mechanics:
- Uses the chat template (enable_thinking=False) with system + user + assistant turns.
- Masked Labels: Sets -100 on prompt tokens so loss is computed solely on completions.
- Dynamic Splitting: Carves out a grouped sentence-level dev_val split when training on dev 
  to prevent data leakage during validation.
"""

import os
import subprocess
import pandas as pd
from io import StringIO
from pathlib import Path

# AUTO-SELECT THE GPU WITH THE MOST FREE MEMORY AT THE VERY TOP
def get_best_gpu():
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        df = pd.read_csv(StringIO(output), names=["index", "free_mem"], sep=",")
        best = df.sort_values(by="free_mem", ascending=False).iloc[0]["index"]
        return str(int(best))
    except Exception:
        return "0"  # fallback

_best_gpu = get_best_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = _best_gpu
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(f"--- Auto-selected GPU {_best_gpu} based on free memory ---")

# ── MASTER CONFIGURATION BLOCK ──────────────────────────────────────────────
# Pick ONE of the four combinations below and set MODEL_VARIANT + TRAIN_ON.
# VALIDATE_ON and SUBSET_SIZE are derived automatically — do not touch them.
#
#   Combo 1 → MODEL_VARIANT = "Base",      TRAIN_ON = "train"
#   Combo 2 → MODEL_VARIANT = "Instruct",  TRAIN_ON = "train"
#   Combo 3 → MODEL_VARIANT = "Base",      TRAIN_ON = "dev"
#   Combo 4 → MODEL_VARIANT = "Instruct",  TRAIN_ON = "dev"
# ────────────────────────────────────────────────────────────────────────────
MODEL_VARIANT = "Instruct"   # "Base" or "Instruct"
# Which split to fine-tune on. "dev" (default) is the headline SFT baseline;
# "train" reproduces the full-train-split CE row in the baseline table.
# Override with QASRL_SFT_TRAIN_ON=train.
TRAIN_ON      = os.environ.get("QASRL_SFT_TRAIN_ON", "dev")   # "train" or "dev"
assert TRAIN_ON in ("train", "dev"), f"QASRL_SFT_TRAIN_ON must be train|dev, got {TRAIN_ON!r}"
# ────────────────────────────────────────────────────────────────────────────

# Auto-derived — no manual edits needed below this line
SUBSET_SIZE  = 1000   # validation samples taken from train when TRAIN_ON="dev"

# When TRAIN_ON="dev" we carve a held-out validation split OUT OF DEV itself so we
# never evaluate on the same examples we train on, and TEST is left completely untouched.
DEV_VAL_FRACTION = 0.10   # fraction of DEV held out for validation (90% train / 10% val)
SPLIT_SEED       = 42     # fixed seed → reproducible, leak-free DEV train/val split
# ── MASTER CONFIGURATION BLOCK ──
# VALIDATE_ON is a dict when training on "train"; when training on "dev" we validate
# ONLY on the held-out DEV validation split ("dev_val") — never on TEST.
VALIDATE_ON = {"dev": "dev", "test": "test"} if TRAIN_ON == "train" else "dev_val"
import pathlib
import transformers
import glob
import torch
import logging
import sys
import json
import random
import requests
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    set_seed
)

# =====================
# 1. Setup & Logging
# =====================
SCRIPT_DIR = Path(__file__).resolve().parent
# Storage root for this stage's outputs (checkpoints / saved adapters / logs).
# Override with QASRL_BASE_DIR; defaults to <repo>/runs so a fresh clone runs
# without editing this file.
BASE_DIR = pathlib.Path(os.environ.get("QASRL_BASE_DIR", SCRIPT_DIR.parents[1] / "runs"))

MODEL = f"Qwen3-30B-A3B-{MODEL_VARIANT}-2507"
MODEL_ID = f"Qwen/{MODEL}"
CURRENT_STAGE = "Stage_CE"

# Automatically changes directory names based on your top configuration choices
VALIDATE_ON_STR = "dev+test" if TRAIN_ON == "train" else "dev_val"
RUN_NAME = f"{MODEL}_train_{TRAIN_ON}_val_{VALIDATE_ON_STR}"

CHECKPOINT_DIR = BASE_DIR / "trainer_runs_baseline" / CURRENT_STAGE / RUN_NAME
MODEL_SAVE_DIR = BASE_DIR / "models_save_baseline" / CURRENT_STAGE / RUN_NAME
LOG_DIR        = BASE_DIR / "logs_baseline"         / CURRENT_STAGE / RUN_NAME

CHECKPOINT_DIR, MODEL_SAVE_DIR, LOG_DIR = map(str, [CHECKPOINT_DIR, MODEL_SAVE_DIR, LOG_DIR])

for d in [CHECKPOINT_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - [STAGE 1 - CE - {RUN_NAME}] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "CE_training.log")),
        logging.StreamHandler()
    ]
)

# Also route HuggingFace/transformers logs into the same log file
_hf_file_handler = logging.FileHandler(os.path.join(LOG_DIR, "training.log"))
_hf_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
transformers.utils.logging.get_logger().addHandler(_hf_file_handler)
transformers.utils.logging.set_verbosity_info()
set_seed(42)

# =====================
# 2. Dataset Loading & Dynamic Splitting
# =====================
def load_and_prepare_dataset():
    logging.info("Loading Raw QASrl datasets...")
    _URL = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/"

    # Base files mapping.
    # When TRAIN_ON="dev" we deliberately load ONLY dev.json, When TRAIN_ON="train" we keep
    # the original behaviour (train for training, dev+test for validation).
    if TRAIN_ON == "train":
        raw_urls = {"train": _URL + "train.json", "dev": _URL + "dev.json", "test": _URL + "test.json"}
    else:
        raw_urls = {"dev": _URL + "dev.json"}
    ds_dict = {}

    # Process each file down to a flat list of samples.
    # We now also carry sentence_id so the DEV split can be grouped by sentence.
    for split_name, url in raw_urls.items():
        response = requests.get(url)
        data = response.json()
        hold = {}
        for example in data:
            s_id, pred = example["sentence_id"], example["predicate"]
            dt = example["detokenized"]
            if s_id not in hold: hold[s_id] = {"sentence": dt["sentence"]}
            if pred not in hold[s_id]: hold[s_id][pred] = []
            ans = " <A> ".join(dt["answers"]["text"])
            hold[s_id][pred].append(f"{dt['question']} {ans}")

        samples = [{"sentence": v["sentence"], "predicate": p, "qa": " <QA> ".join(qas),
                    "sentence_id": s_id}
                   for s_id, v in hold.items() for p, qas in v.items() if p != "sentence"]
        ds_dict[split_name] = Dataset.from_list(samples)

    # ── Held-out DEV validation split (only when training on DEV) ──────────────
    # We carve the validation split OUT OF DEV itself so we never evaluate on the
    # exact examples we train on. The split is GROUPED BY sentence_id: every example
    # of a given sentence goes entirely to train OR entirely to val, so a sentence can
    # never appear in both halves (prevents leakage across predicates of the same
    # sentence). The fixed SPLIT_SEED makes the partition identical on every rerun.
    if TRAIN_ON == "dev":
        dev = ds_dict["dev"]
        unique_sids = sorted(set(dev["sentence_id"]))   # sort → deterministic before seeded shuffle
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(unique_sids)
        n_val = max(1, int(round(len(unique_sids) * DEV_VAL_FRACTION)))
        val_sids = set(unique_sids[:n_val])
        ds_dict["dev_train"] = dev.filter(lambda ex: ex["sentence_id"] not in val_sids)
        ds_dict["dev_val"]   = dev.filter(lambda ex: ex["sentence_id"] in val_sids)
        logging.info(
            f"DEV grouped split (seed={SPLIT_SEED}, val_frac={DEV_VAL_FRACTION}): "
            f"{len(unique_sids)} sentences → {len(unique_sids) - n_val} train / {n_val} val | "
            f"{len(ds_dict['dev_train'])} train / {len(ds_dict['dev_val'])} val examples"
        )
        return DatasetDict(ds_dict)

    # Generate the optional "train_subset" from the structured training set
    # Seed ensures the random slice is identical across script executions
    full_train = ds_dict["train"]
    # shuffled_train = full_train.shuffle(seed=42)

    ds_dict["train_subset"] = full_train.select(range(min(SUBSET_SIZE, len(full_train))))
    # Shrink the core training split slightly so evaluation data isn't leaked into your training pool
    ds_dict["train_remainder"] = full_train.select(range(min(SUBSET_SIZE, len(full_train)), len(full_train)))

    return DatasetDict(ds_dict)

dataset = load_and_prepare_dataset()

logging.info(f"Configured Training Split: '{TRAIN_ON}'")
logging.info(f"Configured Validation Split: '{VALIDATE_ON}'")

# =====================
# 3. Model & Tokenizer
# =====================
logging.info(f"Loading {MODEL_ID} onto GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": int(_best_gpu)},
    trust_remote_code=True,
    attn_implementation="sdpa"
)

model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
))
model.enable_input_require_grads()
# =====================
# 4. Tokenization & Mapping
# =====================
MIN_ANSWER_TOKENS = 64   # tokens reserved for the answer when computing sentence budget
MAX_LENGTH        = 2048  # increased to accommodate system prompt (~430 tokens) + sentence + answer

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

def tokenize_fn(ex):
    ins, lbs, masks = [], [], []
    for s, p, q in zip(ex["sentence"], ex["predicate"], ex["qa"]):

        def make_full_text(sentence):
            """Format the full conversation (system + user + assistant) via chat template."""
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"Given the sentence: '{sentence}'\n"
                        f"Generate all QA pairs for the predicate '{p}'."},
                    {"role": "assistant", "content": q},
                ],
                tokenize=False, add_generation_prompt=False, enable_thinking=False
            )

        def make_prompt(sentence):
            """Format only the prompt portion (system + user) to compute label mask boundary."""
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                        f"Given the sentence: '{sentence}'\n"
                        f"Generate all QA pairs for the predicate '{p}'."},
                ],
                tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

        # Compute how many tokens the sentence can use without overflowing MAX_LENGTH
        answer_ids        = tokenizer(q, add_special_tokens=False)["input_ids"]
        overhead_ids      = tokenizer(make_prompt(""), add_special_tokens=False)["input_ids"]
        sentence_budget   = MAX_LENGTH - len(overhead_ids) - min(len(answer_ids), MIN_ANSWER_TOKENS)
        sentence_ids      = tokenizer(s, add_special_tokens=False)["input_ids"][:sentence_budget]
        s_truncated       = tokenizer.decode(sentence_ids)

        # Tokenize full text and prompt-only to determine where labels should start
        full_text  = make_full_text(s_truncated)
        prompt     = make_prompt(s_truncated)
        full_ids   = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LENGTH]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        labels = [-100] * prompt_len + full_ids[prompt_len:]

        ins.append(full_ids)
        lbs.append(labels[:MAX_LENGTH])
        masks.append([1] * len(full_ids))

    return {"input_ids": ins, "labels": lbs, "attention_mask": masks}

# Derive column names from the first available split (safe across all splits)
_source_columns = next(iter(dataset.values())).column_names
tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=_source_columns)

# =====================
# 5. Execution Setup
# =====================
metric_for_best_model = "eval_dev_loss" if TRAIN_ON == "train" else "eval_loss"

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    logging_dir=LOG_DIR,
    logging_strategy="steps",
    logging_steps=100,             
    eval_strategy="steps",
    eval_steps=200,               
    save_strategy="steps",
    save_steps=200,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=False,           
    load_best_model_at_end=True,  
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_total_limit=5,           
    report_to="tensorboard"
)

# When training on DEV, train on the held-out DEV *train* split and evaluate on the
# held-out DEV *val* split — never on the training examples, and never on TEST.
train_split_key = "dev_train" if TRAIN_ON == "dev" else TRAIN_ON

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds[train_split_key],
    eval_dataset=(
        {"dev": tokenized_ds["dev"], "test": tokenized_ds["test"]}
        if TRAIN_ON == "train"
        else tokenized_ds["dev_val"]
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

# Train
ckpt = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
trainer.train(resume_from_checkpoint=sorted(ckpt, key=os.path.getmtime)[-1] if ckpt else None)

# Save
trainer.model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
logging.info(f"Training complete. Run '{RUN_NAME}' saved to {MODEL_SAVE_DIR}")