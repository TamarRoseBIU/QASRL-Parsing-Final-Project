"""
Stage 2: GRPO Fine-Tuning for QA-SRL (Stage_GRPO_Instruct_DEV.py)
-----------------------------------------------------------------------
Fine-tunes the QA-SRL model using Group Relative Policy Optimization (TRL GRPOTrainer) 
to directly optimize answer-span F1/F-beta reward. Warm-starts from the Stage 1 CE 
adapter trained on the "dev" split.

Key Mechanics:
- Prompt/completion formats match Stage 1 CE (chat-templated, enable_thinking=False).
- Prompts correspond to (sentence, predicate) pairs; completions contain all QA pairs.
- Uses the frozen Stage 1 CE adapter as the implicit KL reference model.
- Evaluates on both "dev" (fit tracking) and "test" (held-out generalization) splits.
"""

import os
import time
from pathlib import Path
import pathlib
import glob
import json
import logging
import requests
import torch
import transformers
import signal
import sys
import itertools
from datasets import Dataset, DatasetDict
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import GRPOConfig, GRPOTrainer
from qasrl_reward import qasrl_reward_full

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===========================================================================
# 1. Paths & Logging
#    Mirrors the CE script's directory layout; only CURRENT_STAGE changes.
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = pathlib.Path(os.environ.get("QASRL_BASE_DIR", SCRIPT_DIR.parents[1] / "runs"))

MODEL      = "Qwen3-30B-A3B-Instruct-2507"
MODEL_ID   = f"Qwen/{MODEL}"
CE_RUN_NAME   = f"{MODEL}_train_dev_val_test"
PREV_STAGE    = "Stage_CE"

# ────── Env-configurable so variants don't need code edits) ───────
#   REWARD_BETA : F_beta for the reward. 1.0 = plain F1 (baseline). >1 weights
#                 recall more — motivated by the error analysis (model is
#                 recall-limited, systematically missing adjunct roles).
#   NUM_GEN     : rollouts per prompt (group size for the GRPO advantage).
#   KL_BETA     : KL anchor strength to the frozen CE reference.
#   GRPO_LR, GRPO_EPOCHS, GRPO_TEMP : optimisation / sampling.
#   EXP_TAG     : appended to RUN_NAME so each variant lands in its own dir.
REWARD_BETA  = float(os.environ.get("REWARD_BETA", "2.0"))
NUM_GEN      = int(os.environ.get("NUM_GEN", "4"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))
KL_BETA      = float(os.environ.get("KL_BETA", "0.04"))
GRPO_LR      = float(os.environ.get("GRPO_LR", "1e-5"))
GRPO_EPOCHS  = int(os.environ.get("GRPO_EPOCHS", "2"))
# TRL requires num_generations to evenly divide the global batch
# (per_device_train_batch_size(=1) * num_processes(=1) * gradient_accumulation_steps).
# Fail fast here rather than deep inside GRPOConfig on a scarce GPU.
assert (1 * GRAD_ACCUM) % NUM_GEN == 0, (
    f"NUM_GEN={NUM_GEN} must divide per_device(1)*GRAD_ACCUM({GRAD_ACCUM}); "
    f"set GRAD_ACCUM to a multiple of NUM_GEN.")
GRPO_TEMP    = float(os.environ.get("GRPO_TEMP", "1.0"))
EVAL_SUBSET  = int(os.environ.get("EVAL_SUBSET", "256"))   # prompts per split; 0 = full
EVAL_STEPS   = int(os.environ.get("EVAL_STEPS", "100"))
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", "100"))
EXP_TAG      = os.environ.get("EXP_TAG", f"grpo_fbeta{REWARD_BETA:g}_ng{NUM_GEN}")
CURRENT_STAGE = "Stage_GRPO_Instruct_DEV"

# Where we load the CE adapter from 
# Stage_CE_Instruct_DEV.py's CHECKPOINT_DIR/MODEL_SAVE_DIR/LOG_DIR convention)
# Resolve the SFT adapter from QASRL_SFT_ADAPTER when set; otherwise fall back to
# the BASE_DIR layout. Set the env var to whatever directory training/sft/ wrote --
# its RUN_NAME depends on the split/validation mode it ran in.
CE_MODEL_DIR   = pathlib.Path(os.environ.get(
    "QASRL_SFT_ADAPTER",
    str(BASE_DIR / "models_save_baseline" / PREV_STAGE / CE_RUN_NAME)))

# Where this stage writes its outputs. RUN_NAME carries the variant + the
# CE split it was warm-started from, so a future Base-model or train-split
# GRPO run will land in its own directory and never collide with this one.
RUN_NAME = f"{MODEL}_grpo_from_{CE_RUN_NAME}_{EXP_TAG}"

CHECKPOINT_DIR = BASE_DIR / "trainer_runs_baseline" / CURRENT_STAGE / RUN_NAME
MODEL_SAVE_DIR = BASE_DIR / "models_save_baseline"  / CURRENT_STAGE / RUN_NAME
LOG_DIR        = BASE_DIR / "logs_baseline"         / CURRENT_STAGE / RUN_NAME

for d in [CHECKPOINT_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(str(d), exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - [STAGE 2 - GRPO - {RUN_NAME}] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "grpo_training.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
batch_stats_buffer = []

# Also hook into HuggingFace's internal logger
_hf_handler = logging.FileHandler(str(LOG_DIR / "training.log"))
_hf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
transformers.utils.logging.get_logger().addHandler(_hf_handler)
transformers.utils.logging.set_verbosity_info()

set_seed(42)

log.info(
    "[EXP2 CONFIG] "
    f"EXP_TAG={EXP_TAG} REWARD_BETA={REWARD_BETA} NUM_GEN={NUM_GEN} "
    f"GRAD_ACCUM={GRAD_ACCUM} KL_BETA={KL_BETA} GRPO_LR={GRPO_LR} "
    f"GRPO_EPOCHS={GRPO_EPOCHS} GRPO_TEMP={GRPO_TEMP} "
    f"EVAL_SUBSET={EVAL_SUBSET} EVAL_STEPS={EVAL_STEPS} SAVE_STEPS={SAVE_STEPS} "
    f"RUN_NAME={RUN_NAME}"
)


# ===========================================================================
# 2. Dataset Loading
#    Groups the flat per-QA rows by (sentence_id, predicate), exactly as the
#    CE script does, but stores gold QAs as a JSON-serialised list of dicts
#    so the reward function can consume them directly.
#
#    PROMPT FORMAT: built with the tokenizer's chat template, using the
#    EXACT same system prompt the Instruct model saw during CE training
#    (Stage_CE_Instruct_DEV.py). This keeps GRPO prompts in-distribution
#    with what the warm-started policy was actually trained on.
# ===========================================================================

# Identical to SYSTEM_PROMPT in Stage_CE_Instruct_DEV.py — keep these in
# sync if the CE prompt ever changes, or GRPO will optimise against an
# out-of-distribution instruction.
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

def load_grpo_dataset(tok) -> DatasetDict:
    """
    Returns a DatasetDict with columns:
        prompt    – chat-templated prompt (system + user turn), built with
                    `tok.apply_chat_template(..., add_generation_prompt=True,
                    enable_thinking=False)` — IDENTICAL formatting to the
                    Instruct CE training prompts. (for GRPOTrainer)
        sentence  – raw sentence string                         (for reward_fn)
        predicate – predicate string                            (for logging)
        gold_qas  – JSON str: List[{"question": str, "answer": str}]

    Splits: "train" (train.json), "dev" (dev.json), "test" (test.json).
    "dev" and "test" are both passed to GRPOTrainer as separate eval sets
    so metrics/dashboards/plots can be tracked independently per split
    (e.g. eval_dev_loss vs eval_test_loss, "eval_dev/f1" vs "eval_test/f1").
    """
    log.info("Loading QA-SRL dataset …")
    _URL   = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/"
    splits = {
        "train": _URL + "train.json",
        "dev":   _URL + "dev.json",
        "test":  _URL + "test.json",
    }

    def make_prompt(sentence: str, predicate: str) -> str:
        """Mirrors Stage_CE_Instruct_DEV.py's make_prompt(): system + user
        turn only, with add_generation_prompt=True so the template appends
        the assistant-turn opening tokens the model must continue from."""
        return tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Given the sentence: '{sentence}'\n"
                    f"Generate all QA pairs for the predicate '{predicate}'."},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    ds_dict = {}
    for split_name, url in splits.items():
        response = requests.get(url)
        data     = response.json()

        # ── Group all QA rows by (sentence_id, predicate) ────────────────
        hold: dict = {}
        for example in data:
            s_id     = example["sentence_id"]
            pred     = example["predicate"]
            dt       = example["detokenized"]
            sentence = dt["sentence"]

            if s_id not in hold:
                hold[s_id] = {"sentence": sentence}
            if pred not in hold[s_id]:
                hold[s_id][pred] = []

            # One QAPair per (question, answer_text) combination.
            # Multi-answer QAs (joined with <A> in CE output) each become
            # a separate dict here so qasrl_reward receives a flat list.
            for ans_text in dt["answers"]["text"]:
                hold[s_id][pred].append({
                    "question": dt["question"],
                    "answer":   ans_text,
                })

        # ── Build one sample per (sentence, predicate) ───────────────────
        samples = []
        for s_id, v in hold.items():
            sentence = v["sentence"]
            for pred, gold_qas in v.items():
                if pred == "sentence":   # skip the metadata key
                    continue
                samples.append({
                    # GRPOTrainer requires exactly the column name "prompt"
                    "prompt":    make_prompt(sentence, pred),
                    "sentence":  sentence,
                    "predicate": pred,
                    # Serialise as JSON str – HuggingFace Datasets handles
                    # List[dict] poorly, and strings survive all collation.
                    "gold_qas":  json.dumps(gold_qas),
                    # A unique key for grouping rollouts of the SAME
                    # (sentence, predicate) prompt in the dashboard.
                    "group_id":  f"{s_id}::{pred}",
                    # Explicit split tag, passed through kwargs to reward_fn.
                    # This is more reliable than checking model.training,
                    # which doesn't distinguish "dev" eval from "test" eval.
                    "split":     split_name,
                })

        ds_dict[split_name] = Dataset.from_list(samples)
        log.info(f"  {split_name}: {len(ds_dict[split_name]):,} (sentence, predicate) groups")

    return DatasetDict(ds_dict)


# NOTE: dataset is loaded AFTER the tokenizer (below), since prompt
# construction now depends on tokenizer.apply_chat_template().

# ===========================================================================
# 3. Model & Tokenizer
#    Load the base model, then hot-swap in the CE LoRA adapter
#    as a *trainable* PeftModel.  TRL's GRPOTrainer automatically uses the
#    frozen adapter weights as the KL reference model.
# ===========================================================================

log.info(f"Loading tokenizer from CE checkpoint: {CE_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(
    str(CE_MODEL_DIR),
    trust_remote_code=True,
)
tokenizer.pad_token    = tokenizer.eos_token
# Left-padding is required for correct batch generation with decoder-only
# models – the generated tokens must always be at the right end.
tokenizer.padding_side = "left"

# Now that the tokenizer (with its chat template) is available, build the
# GRPO prompts identically to how Stage_CE_Instruct_DEV.py built them.
dataset = load_grpo_dataset(tokenizer)

log.info(f"Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},           # single GPU, same as CE stage
    trust_remote_code=True,
    attn_implementation="sdpa",
)

log.info(f"Attaching CE LoRA adapter (trainable): {CE_MODEL_DIR}")
model = PeftModel.from_pretrained(
    base_model,
    str(CE_MODEL_DIR),
    is_trainable=True,   # keep adapter weights in the computation graph
)
log.info("Model ready.")

# ===========================================================================
# 3b. Dataset Pre-processing
# ===================================================================================================================================================

def add_prompt_length(example):
    """Add token length of the prompt so we can audit/filter."""
    example["prompt_len"] = len(tokenizer(
        example["prompt"],
        add_special_tokens=False
    )["input_ids"])
    return example

# Audit lengths before any filtering
dataset = dataset.map(add_prompt_length)

for split in ["train", "dev", "test"]:
    lengths = dataset[split]["prompt_len"]
    print(f"\n{split} prompt length stats:")
    print(f"  Min    : {min(lengths)}")
    print(f"  Max    : {max(lengths)}")
    print(f"  Mean   : {sum(lengths)/len(lengths):.1f}")
    print(f"  > 512  : {sum(1 for l in lengths if l > 512)} samples")
    print(f"  > 768  : {sum(1 for l in lengths if l > 768)} samples")

# Filter — don't truncate. A broken sentence is useless for QA-SRL.
# IMPORTANT: unlike the Base-model script's raw prompt ("sentence: ...\n
# predicate: ...\nanswer: ", ~20-30 tokens), every prompt here carries the
# full chat-templated SYSTEM_PROMPT (~430 tokens, matching the CE script's
# own comment "system prompt (~430 tokens)"). 768 leaves headroom for the
# system prompt + a reasonably long sentence, while keeping
# system_prompt + sentence + completion comfortably under the CE script's
# MAX_LENGTH=2048 once max_completion_length (96, see GRPOConfig below) is
# added back in.
MAX_PROMPT_TOKENS = 768

dataset = dataset.filter(lambda ex: ex["prompt_len"] <= MAX_PROMPT_TOKENS)

for split in ["train", "dev", "test"]:
    print(f"{split} after filter: {len(dataset[split]):,} examples")

# Print the expected step count for this run, now that we know the
# actual size of dataset["dev"] post-filter 
_n_train = len(dataset["dev"])
_effective_batch = 1 * 4   # per_device_train_batch_size * gradient_accumulation_steps
_epochs = 2
_steps_per_epoch = _n_train // _effective_batch
_total_steps = _steps_per_epoch * _epochs
print(
    f"\n[GRPO step estimate] train examples (dev split)={_n_train:,} | "
    f"effective_batch={_effective_batch} | epochs={_epochs} -> "
    f"steps/epoch={_steps_per_epoch:,} | total_steps={_total_steps:,}\n"
)

# ===========================================================================
# 4. Completion Parser
#    Mirrors the CE output format exactly so parse errors are minimised.
#
#    CE output format:
#        "Q1? ans1a <A> ans1b <QA> Q2? ans2"  (+ EOS token)
#
#    Returns List[{"question": str, "answer": str}] — same type as gold_qas.
# ===========================================================================

_EOS = tokenizer.eos_token

def parse_completion(text: str) -> list[dict]:
    # Strip EOS and whitespace
    text = text.strip()
    if _EOS and text.endswith(_EOS):
        text = text[: -len(_EOS)].strip()

    qas: list[dict] = []
    for chunk in text.split("<QA>"):
        chunk = chunk.strip()
        if not chunk:
            continue

        q_pos = chunk.find("?")
        if q_pos == -1:
            # Malformed chunk – no question mark found; skip
            continue

        question    = chunk[: q_pos + 1].strip()
        answer_part = chunk[q_pos + 1 :].strip()

        # Each answer alternative becomes its own QAPair (matches gold format)
        for ans in answer_part.split("<A>"):
            ans = ans.strip()
            if ans:
                qas.append({"question": question, "answer": ans})

    return qas


# ===========================================================================
# 5. Reward Function
#    TRL calls reward_fn(completions, **kwargs) where:
#      - completions : List[str]  length = per_device_train_batch_size × num_generations
#      - **kwargs    : dataset columns, each repeated num_generations times
#    Returns List[float] of the same length.
#
#    Each call also buffers full precision/recall/f1 + log_label + group_id
#    into batch_stats_buffer, which GRPODashboardCallback drains on every
#    on_log.
#
#    IMPORTANT — why log_label is NOT just the "split" column:
#    GRPO trains on dataset["dev"] AND evaluates on dataset["dev"] (to track
#    trend lines only — no gradient updates come from the eval call, since
#    Trainer.evaluate() never calls .backward()/.step()). That means the
#    SAME underlying rows ("split" == "dev") are used in two different
#    contexts: as live training rollouts, and as a clean eval snapshot. If
#    we logged purely by the "split" column, both would land under the
#    single TensorBoard tag "dev" and get mixed together — exactly the
#    ambiguity we need to avoid for clean trend graphs.
#
#    log_label disambiguates by ALSO checking model.training:
#      - dev  data + model.training=True  -> "train"      (training rollout)
#      - dev  data + model.training=False -> "eval_dev"    (eval snapshot)
#      - test data + model.training=False -> "eval_test"   (eval snapshot)
#    model.training is reliable for THIS distinction (train-step vs
#    Trainer.evaluate() call) because Trainer always calls model.eval()
#    before generating for evaluate() and model.train() afterward.
# ===========================================================================

def reward_fn(completions: list[str], **kwargs) -> list[float]:
    sentences     = kwargs["sentence"]
    gold_qas_strs = kwargs["gold_qas"]
    predicates    = kwargs["predicate"]      # for the dashboard
    group_ids     = kwargs["group_id"]       # for within-prompt best/worst
    splits        = kwargs["split"]          # raw data source: "dev" / "test"

    is_training_step = model.training  # snapshot once per call, not per-row

    rewards: list[float] = []
    for completion, sentence, gold_qas_str, pred, group_id, split in zip(
        completions, sentences, gold_qas_strs, predicates, group_ids, splits
    ):
        gold_qas = json.loads(gold_qas_str)
        pred_qas = parse_completion(completion)
        result   = qasrl_reward_full(pred_qas, gold_qas, sentence, beta=REWARD_BETA)
        f1        = result["f1"]
        fbeta     = result["fbeta"]     # == f1 when REWARD_BETA == 1.0
        precision = result["precision"]
        recall    = result["recall"]

        # Resolve the TensorBoard/dashboard label (see docstring above).
        if is_training_step:
            log_label = "train"        # dev data, used as the training set
        else:
            log_label = f"eval_{split}"  # "eval_dev" or "eval_test" snapshot

        # Store metadata for the custom dashboard AND for the P/R/F1 curves.
        # Buffered for every context (train / eval_dev / eval_test).
        batch_stats_buffer.append({
            "sentence":   sentence,
            "predicate":  pred,
            "completion": completion,
            "gold":       gold_qas_str,
            "group_id":   group_id,
            "split":      log_label,
            "f1":         float(f1),
            "precision":  float(precision),
            "recall":     float(recall),
        })
        # The scalar REWARD is F_beta (== F1 when REWARD_BETA==1.0). The
        # dashboard still tracks true F1/P/R above so we monitor the real
        # eval-aligned metric while optimising the recall-weighted objective.
        rewards.append(float(fbeta))

    return rewards
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

# Dedicated SummaryWriter so P/R/F1 curves land in TensorBoard regardless
# of HF's own report_to="tensorboard" callback ordering/timing. Writes to
# the same LOG_DIR, so `tensorboard --logdir <LOG_DIR>` picks up both.
_tb_writer = SummaryWriter(log_dir=str(LOG_DIR))


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


class GRPODashboardCallback(TrainerCallback):
    """
    Drains batch_stats_buffer on every on_log call and reports, SEPARATELY
    PER SPLIT (train / dev / test):

      1. Scalar curves -> TensorBoard, so you can plot F1/precision/recall
         over training steps later:
            <split>/f1, <split>/precision, <split>/recall

      2. WITHIN-PROMPT best vs worst: among the several rollouts
         (num_generations) generated for the SAME (sentence, predicate),
         which one scored highest vs lowest? Surfaced for the prompt with
         the largest within-group spread, since that's the most
         informative case to inspect.

      3. PER-GROUP AVERAGE best vs worst: average F1 across all rollouts
         of each (sentence, predicate), then show which group scored
         best/worst ON AVERAGE. This is what tells you whether there's a
         consistent type of sentence/predicate the model struggles with,
         as opposed to just noisy single-sample variance.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        global batch_stats_buffer
        logs = logs or {}

        # ── 0. Always mirror HF/TRL's own loss-family scalars into our
        # writer, independent of whether report_to="tensorboard"'s own
        # TensorBoardCallback actually got attached (it silently no-ops if
        # `is_tensorboard_available()` is False in this env, even though
        # our manually-created _tb_writer works fine via torch's bundled
        # writer).
        for key in ("loss", "grad_norm", "learning_rate", "reward",
                    "reward_std", "kl", "completion_length"):
            if key in logs:
                _tb_writer.add_scalar(f"train/{key}", logs[key], state.global_step)
        for raw_key in ("dev", "test"):
            for suffix in ("loss", "reward", "reward_std", "kl", "completion_length"):
                k = f"eval_{raw_key}_{suffix}"
                if k in logs:
                    _tb_writer.add_scalar(f"eval_{raw_key}/{suffix}", logs[k], state.global_step)

        if not batch_stats_buffer:
            return

        is_eval  = any(k.startswith("eval_") for k in logs)
        is_train = "reward" in logs or "loss" in logs

        if not is_eval and not is_train:
            batch_stats_buffer = []
            return

        # Buffered entries may span multiple splits in one flush (e.g. TRL
        # evaluates "dev" then "test" before the next on_log). Always
        # break down by split so dev/test numbers never get blended.
        buffer_snapshot = list(batch_stats_buffer)
        present_splits  = sorted(set(s["split"] for s in buffer_snapshot))

        for split in present_splits:
            split_samples = [s for s in buffer_snapshot if s["split"] == split]
            n = len(split_samples)

            # ── 1. TensorBoard scalar curves (f1 / precision / recall) ───
            mean_f1   = _mean(s["f1"]        for s in split_samples)
            mean_prec = _mean(s["precision"] for s in split_samples)
            mean_rec  = _mean(s["recall"]    for s in split_samples)

            tag_prefix = f"{split}"  # "train" / "eval_dev" / "eval_test"
            _tb_writer.add_scalar(f"{tag_prefix}/f1",        mean_f1,   state.global_step)
            _tb_writer.add_scalar(f"{tag_prefix}/precision", mean_prec, state.global_step)
            _tb_writer.add_scalar(f"{tag_prefix}/recall",    mean_rec, state.global_step)

            # ── 2. WITHIN-PROMPT best vs worst (same sentence×predicate) ──
            # Group this split's samples by group_id; requires pre-sorting
            # since itertools.groupby only groups CONSECUTIVE matching keys.
            sorted_for_groupby = sorted(split_samples, key=lambda x: x["group_id"])
            groups = {
                gid: list(grp)
                for gid, grp in itertools.groupby(sorted_for_groupby, key=lambda x: x["group_id"])
            }

            # Pick the group with the largest within-group F1 spread —
            # the most informative case for "different scores on the same
            # sentence" (only meaningful when num_generations > 1).
            widest_group_id, widest_group = None, None
            widest_spread = -1.0
            for gid, members in groups.items():
                if len(members) < 2:
                    continue
                scores = [m["f1"] for m in members]
                spread = max(scores) - min(scores)
                if spread > widest_spread:
                    widest_spread, widest_group_id, widest_group = spread, gid, members

            # ── 3. PER-GROUP AVERAGE best vs worst (across ALL groups) ────
            # Average F1 per (sentence, predicate) group, then find which
            # group is best/worst ON AVERAGE -> surfaces sentence "types"
            # the model consistently struggles with, not just one noisy
            # rollout.
            group_avgs = [
                {
                    "group_id": gid,
                    "sentence": members[0]["sentence"],
                    "predicate": members[0]["predicate"],
                    "avg_f1": _mean(m["f1"] for m in members),
                    "n": len(members),
                    "example_completion": members[0]["completion"],
                }
                for gid, members in groups.items()
            ]
            best_group  = max(group_avgs, key=lambda g: g["avg_f1"]) if group_avgs else None
            worst_group = min(group_avgs, key=lambda g: g["avg_f1"]) if group_avgs else None

            # ── Build console / log-file report for this split ───────────
            header_extra = ""
            if is_eval:
                # TRL's own metric keys for a dict eval_dataset={"dev":...,
                # "test":...} are "eval_dev_reward" / "eval_test_reward"
                # (it prepends "eval_" to the dict key once). Our display
                # label is "eval_dev"/"eval_test", so strip our own prefix
                # back off to get TRL's raw key before doing the lookup.
                raw_key = split.removeprefix("eval_")  # "dev" / "test"
                eval_reward = (
                    logs.get(f"eval_{raw_key}_reward_fn/mean")
                    or logs.get(f"eval_{raw_key}_reward")
                    or logs.get("eval_reward_fn/mean")
                    or logs.get("eval_reward/mean")
                    or logs.get("eval_reward")
                    or 0.0
                )
                header_extra = f"  (logged eval_reward: {eval_reward:.4f})"
            else:
                kl_div = logs.get("kl", 0)
                loss   = logs.get("loss", 0)
                header_extra = f"  KL: {kl_div:.4f}  Loss: {loss:.6f}"

            within_block = "  (no group with >1 rollout in this flush — increase num_generations to see spread)"
            if widest_group is not None:
                wg_sorted = sorted(widest_group, key=lambda x: x["f1"])
                wg_worst, wg_best = wg_sorted[0], wg_sorted[-1]
                within_block = f"""  Group with largest within-prompt spread (spread={widest_spread:.3f}, n={len(widest_group)} rollouts):
    Sentence  : {wg_best['sentence']}
    Predicate : {wg_best['predicate']}
    BEST  rollout F1={wg_best['f1']:.3f}  -> {wg_best['completion']}.
    WORST rollout F1={wg_worst['f1']:.3f}  -> {wg_worst['completion']}."""

            group_block = "  (no groups in this flush)"
            if best_group is not None:
                group_block = f"""  BEST  avg group  (avg_f1={best_group['avg_f1']:.3f}, n={best_group['n']}):
    Sentence  : {best_group['sentence']}
    Predicate : {best_group['predicate']}
  WORST avg group  (avg_f1={worst_group['avg_f1']:.3f}, n={worst_group['n']}):
    Sentence  : {worst_group['sentence']}
    Predicate : {worst_group['predicate']}
    Example completion: {worst_group['example_completion']}."""

            output = f"""
========================================================================
  [{split.upper()}]  STEP {state.global_step: >7} |  epoch {state.epoch:.2f}   [{n} samples]
------------------------------------------------------------------------
  Mean F1={mean_f1:.4f}  P={mean_prec:.4f}  R={mean_rec:.4f}{header_extra}
------------------------------------------------------------------------
  WITHIN-PROMPT best vs worst (same sentence x predicate, different rollouts):
{within_block}
------------------------------------------------------------------------
  PER-GROUP AVERAGE best vs worst (which sentence/predicate type struggles):
{group_block}
========================================================================
"""
            print(output)
            log.info(output)

        # Clear buffer for the next logging interval
        batch_stats_buffer = []

# ── Add the RequeueCallback class ────────────────────────────────────
# after the GRPODashboardCallback class (before Section 6).
#
# How it works:
#   - SLURM sends SIGUSR1 ~90s before the hard time limit.
#   - The signal handler sets a flag on this callback instance.
#   - on_step_end() checks the flag every step and, when set, tells the Trainer
#     to save a full checkpoint (model + optimizer + scheduler + random states)
#     and stop training cleanly.
#   - On the next SLURM run, the existing resumption logic (Section 7) picks up
#     that checkpoint automatically.
 
class RequeueCallback(transformers.TrainerCallback):
    """Catches SIGUSR1 from SLURM and saves a checkpoint before the job dies."""
 
    def __init__(self):
        self._requeue_requested = False
 
    def register_signal(self):
        """Call this after the trainer is created so `self` is in scope."""
        def _handler(signum, frame):
            log.warning(
                "SIGUSR1 received — SLURM time limit approaching. "
                "Saving checkpoint and stopping cleanly …"
            )
            self._requeue_requested = True
 
        signal.signal(signal.SIGUSR1, _handler)
        log.info("SIGUSR1 handler registered (graceful requeue checkpointing active).")
 
    def on_step_end(self, args, state, control, **kwargs):
        if self._requeue_requested:
            log.warning(
                f"Requeue flag set at step {state.global_step}. "
                "Triggering checkpoint save and clean exit."
            )
            control.should_save             = True   # save full trainer checkpoint
            control.should_training_stop    = True   # exit train() loop cleanly
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log.info(f"Eval metrics: {metrics}")
        """Checks for the signal even during the validation loop."""
        if self._requeue_requested:
            log.warning("Signal detected during validation! Stopping early.")
            control.should_training_stop = True    
 
# ===========================================================================
# 6. GRPO Config & Trainer
# ===========================================================================

grpo_config = GRPOConfig(
    # ── Output paths ──────────────────────────────────────────────────
    output_dir  = str(CHECKPOINT_DIR),
    logging_dir = str(LOG_DIR),

    # ── Generation ────────────────────────────────────────────────────
    num_generations       = NUM_GEN,
    max_completion_length = 96,
    temperature            = GRPO_TEMP,
    top_p                  = 0.95,

    # ── GRPO-specific ─────────────────────────────────────────────────
    beta    = KL_BETA,
    epsilon = 0.2,

    # ── Optimisation ──────────────────────────────────────────────────
    learning_rate               = GRPO_LR,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = GRPO_EPOCHS,
    bf16                        = True,
    gradient_checkpointing      = True,
    optim                       = "paged_adamw_8bit",
    # weight_decay / lr_scheduler_type: safe, standard additions, kept as
    # synced from the reference.
    weight_decay                = 0.001,
    warmup_steps                = 30,
    lr_scheduler_type           = "linear",

    # ── Logging & Saving ──────────────────────────────────────────────
    logging_strategy       = "steps",
    logging_steps          = 10,
    save_strategy          = "steps",
    save_steps             = SAVE_STEPS,
    eval_strategy          = "steps",
    eval_steps             = EVAL_STEPS,
    save_total_limit       = 8,
    load_best_model_at_end = False,
    report_to              = "tensorboard",
)


requeue_cb = RequeueCallback()
requeue_cb.register_signal()

# ── Cheap eval sets ──
if EVAL_SUBSET and EVAL_SUBSET > 0:
    _dev_eval  = dataset["dev"].shuffle(seed=42).select(range(min(EVAL_SUBSET, len(dataset["dev"]))))
    _test_eval = dataset["test"].shuffle(seed=42).select(range(min(EVAL_SUBSET, len(dataset["test"]))))
else:
    _dev_eval, _test_eval = dataset["dev"], dataset["test"]
_eval_sets = {"dev": _dev_eval, "test": _test_eval}
log.info(f"[EXP2 EVAL] subset={EVAL_SUBSET} -> dev_eval={len(_dev_eval)} test_eval={len(_test_eval)} "
         f"eval_steps={EVAL_STEPS} save_steps={SAVE_STEPS}")

trainer = GRPOTrainer(
    model             = model,
    args              = grpo_config,
    # FIXED: train on "dev" (2,406 sentence x predicate groups), matching
    # the CE stage this run warm-starts from (Stage_CE_Instruct_DEV.py,
    # MODEL_VARIANT="Instruct", TRAIN_ON="dev" -> RUN_NAME ends in
    # "_train_dev_val_test"). dataset["train"] (92,805 groups, full
    # train.json) is NOT used here.
    train_dataset     = dataset["dev"],
    # eval on BOTH "dev" and "test":
    #   - "test"  : the real held-out metric — no gradient updates ever
    #               come from it, so it's a clean indicator of generalization.
    #   - "dev"   : also the TRAINING data here. Including it in eval is
    #               safe for TREND-TRACKING ONLY (Trainer.evaluate() never
    #               calls .backward()/.step(), so no extra learning happens
    #               from evaluating on it) — but its numbers should NOT be
    #               read as held-out generalization, since the model is
    #               actively fit to this exact data. reward_fn's log_label
    #               logic (see Section 5) tags these "eval_dev" so they
    #               stay visually distinct from "train" rollout stats and
    #               from "eval_test" in TensorBoard/dashboard output.
    # Both are evaluated in the SAME trainer.evaluate() call at a given
    # eval_steps checkpoint, so "eval_dev" and "eval_test" numbers at any
    # given step are always a clean, same-checkpoint snapshot pair.
    eval_dataset      = _eval_sets,
    reward_funcs      = reward_fn,
    processing_class  = tokenizer,
    callbacks         = [GRPODashboardCallback(), requeue_cb],  # <-- added requeue_cb
)



# ===========================================================================
# 7. Train (with checkpoint resumption)
# ===========================================================================

ckpt_glob = glob.glob(str(CHECKPOINT_DIR / "checkpoint-*"))
resume    = sorted(ckpt_glob, key=os.path.getmtime)[-1] if ckpt_glob else None
if resume:
    log.info(f"Resuming from checkpoint: {resume}")

_train_start = time.time()
trainer.train(resume_from_checkpoint=resume)
_train_elapsed = time.time() - _train_start

_hours, _rem   = divmod(int(_train_elapsed), 3600)
_mins,  _secs  = divmod(_rem, 60)
_time_str = f"{_hours}h {_mins:02d}m {_secs:02d}s"
log.info(f"Training complete. Total time: {_time_str}  ({_train_elapsed:.1f} s)")


# ===========================================================================
# 8. Save
# ===========================================================================

trainer.model.save_pretrained(str(MODEL_SAVE_DIR))
tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
log.info(f"GRPO stage complete. Adapter saved to {MODEL_SAVE_DIR}")