"""
Stage 2: DPO Fine-Tuning for QA-SRL (Stage_DPO_Instruct_DEV.py)
----------------------------------------------------------------------
Optimizes the QA-SRL model via Direct Preference Optimization (TRL DPOTrainer).
Warm-starts from the Stage 1 CE adapter and trains on contrastive pairs 
(chosen vs. rejected) built by build_dpo_training_data.py.

Key Mechanics:
- Prompt/completion formats match Stage 1 (CE) and Stage 2 (GRPO).
- Uses the frozen CE adapter as the implicit reference model.
- Evaluates pairwise preference loss on dpo_val.jsonl. Final checkpoint
  selection is handled post-hoc on the dev set via eval_on_val.py.

Changes from the earlier DPO trainer this file supersedes:
- DPO_ARM selects the preference-pair construction arm; it is appended to
  RUN_NAME so arms never overwrite each other's checkpoints.
- The pair directory and seed are env-driven (DPO_DATA_DIR, DPO_SEED) instead
  of hard-coded, so an arm/seed sweep needs no source edits.
- Preference-loss evaluation reads dpo_val.jsonl (a held-out slice) rather than
  dpo_eval.jsonl, which overlapped the training pairs.
- The pair loader guards against a stray process/index column in the pair files,
  which previously shifted the chosen/rejected fields.
"""

import os
import time
from pathlib import Path
import pathlib
# ── Pin to one GPU at the very top, before any CUDA init ──────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import json
import logging
import signal
import requests
import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import DPOConfig, DPOTrainer
from qasrl_reward import qasrl_reward_full

# ===========================================================================
# 1. Paths & Logging
#    Mirrors Stage_GRPO_Instruct_DEV.py's directory layout; only
#    CURRENT_STAGE / the trainer type changes.
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = pathlib.Path(os.environ.get("QASRL_BASE_DIR", SCRIPT_DIR.parents[1] / "runs"))
MODEL      = "Qwen3-30B-A3B-Instruct-2507"
MODEL_ID   = f"Qwen/{MODEL}"

# Names the SFT run this stage warm-starts from. Used for THIS stage's output
# directory name only -- the adapter itself is located via QASRL_SFT_ADAPTER
# below, so the two need not agree.
CE_RUN_NAME   = f"{MODEL}_train_dev_val_test"

PREV_STAGE    = "Stage_CE"
CURRENT_STAGE = "Stage_DPO_Instruct_DEV"

# Where we load the warm-start adapter from. Defaults to the CE checkpoint,
# matching Stage_GRPO_Instruct_DEV.py (both stages branch independently off
# CE). To instead run DPO *after* GRPO, point this at GRPO's MODEL_SAVE_DIR:
#   PREV_STAGE  = "Stage_GRPO_Instruct_DEV"
#   PREV_MODEL_DIR = BASE_DIR / "models_save_baseline" / PREV_STAGE / \
#                    f"{MODEL}_grpo_from_{CE_RUN_NAME}"
# Resolve the warm-start adapter from QASRL_SFT_ADAPTER when set; otherwise fall
# back to the BASE_DIR layout. Set it to whatever directory training/sft/ wrote.
PREV_MODEL_DIR = pathlib.Path(os.environ.get(
    "QASRL_SFT_ADAPTER",
    str(BASE_DIR / "models_save_baseline" / PREV_STAGE / CE_RUN_NAME)))

# Where this stage writes its outputs.
# DPO_ARM tags the run so concurrent arms/seeds never share a checkpoint dir
# (without it a second run would resume from the first one's checkpoints).
_ARM = os.environ.get("DPO_ARM", "")
RUN_NAME = f"{MODEL}_dpo_from_{CE_RUN_NAME}" + (f"_{_ARM}" if _ARM else "")

CHECKPOINT_DIR = BASE_DIR / "trainer_runs_baseline" / CURRENT_STAGE / RUN_NAME
MODEL_SAVE_DIR = BASE_DIR / "models_save_baseline"  / CURRENT_STAGE / RUN_NAME
LOG_DIR        = BASE_DIR / "logs_baseline"         / CURRENT_STAGE / RUN_NAME
DATA_DIR   = pathlib.Path(os.environ.get(
    "DPO_DATA_DIR",
    str(SCRIPT_DIR / "existing_dataset" / "pairs_onpolicy_recall")))
TRAIN_FILE = DATA_DIR / "dpo_train.jsonl"
# The eval split is the held-out DEV slice produced by build_onpolicy_pairs.py.
EVAL_FILE  = DATA_DIR / "dpo_val.jsonl"

for d in [CHECKPOINT_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(str(d), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - [STAGE 3 - DPO - {RUN_NAME}] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "dpo_training.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Also hook into HuggingFace's internal logger
_hf_handler = logging.FileHandler(str(LOG_DIR / "training.log"))
_hf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
transformers.utils.logging.get_logger().addHandler(_hf_handler)
transformers.utils.logging.set_verbosity_info()

# v2: seed is env-driven — the noise-floor measurement is two runs of the SAME
# config differing ONLY here.
SEED = int(os.environ.get("DPO_SEED", "42"))
set_seed(SEED)


# ===========================================================================
# 2. Dataset Loading
#    prompt/chosen/rejected are already fully built (chat-templated prompt,
#    CE-format completions) by build_dpo_training_data.py -- this stage
#    just loads the JSONL as-is. See that script's module docstring for the
#    exact rendering rules.
# ===========================================================================

log.info(f"Loading DPO train data from {TRAIN_FILE}")
log.info(f"Loading DPO eval  data from {EVAL_FILE}")
if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"{TRAIN_FILE} not found. Run build_dpo_data.py, then (optionally) "
        f"llm_fix_grammar.py, then build_dpo_training_data.py first."
    )

data_files = {"train": str(TRAIN_FILE)}
if EVAL_FILE.exists():
    data_files["eval"] = str(EVAL_FILE)
else:
    log.warning(f"{EVAL_FILE} not found -- proceeding with no eval set.")

raw_datasets = load_dataset("json", data_files=data_files)
log.info(f"  train: {len(raw_datasets['train']):,} preference pairs")
if "eval" in raw_datasets:
    log.info(f"  eval : {len(raw_datasets['eval']):,} preference pairs")

for split in raw_datasets:
    # `process` (add/truncate) exists only for the SYNTHETIC pair sets. On-policy
    # pair sets have no such column -- their negatives are real model samples, not
    # scripted perturbations -- so this breakdown is simply not applicable there.
    # Diagnostic print only; must never gate training.
    if "process" in raw_datasets[split].column_names:
        n_add = sum(1 for r in raw_datasets[split]["process"] if r == "add")
        n_trunc = len(raw_datasets[split]) - n_add
        log.info(f"  [{split}] process breakdown: add={n_add:,}  truncate={n_trunc:,}")
    else:
        log.info(f"  [{split}] no `process` column (on-policy pairs) -- breakdown skipped")


# ===========================================================================
# 2b. Raw test.json loading + parsing, for F1EvalCallback (Section 5b below)
#     -- SEPARATE from raw_datasets above. That dataset holds pre-rendered
#     chosen/rejected strings for pairwise loss; this needs raw gold QA
#     lists to score actual GENERATIONS against, so it's built straight
#     from test.json the same way Stage_GRPO_Instruct_DEV.py's
#     load_grpo_dataset() builds its "test" split.
# ===========================================================================

# Identical to SYSTEM_PROMPT in Stage_CE_Instruct_DEV.py / build_dpo_training_data.py
# / Stage_GRPO_Instruct_DEV.py -- keep these in sync, or this eval's prompts
# drift out-of-distribution from what the model was actually trained on.
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

def parse_completion_for_f1(text: str) -> list[dict]:
    """
    Parses a generated completion string ("Q1? ans1a <A> ans1b <QA> Q2? ans2")
    back into List[{"question", "answer"}] -- same CE output format and same
    splitting logic as Stage_GRPO_Instruct_DEV.py's parse_completion(). Not
    imported from there (that script isn't a module other scripts import
    from) -- duplicated deliberately, keep the two in sync if the completion
    format ever changes.

    Assumes the completion was decoded with skip_special_tokens=True (see
    F1EvalCallback._run_eval below), so there's no EOS token to strip here
    -- unlike GRPO's version, which parses raw un-skipped decodes.
    """
    text = text.strip()
    qas: list[dict] = []
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


# ===========================================================================
# 3. Model & Tokenizer
#    Load the base model, then hot-swap in the CE LoRA adapter as a
#    *trainable* PeftModel -- identical pattern to Stage_GRPO_Instruct_DEV.py.
#    DPOTrainer automatically uses the frozen adapter weights (via
#    disable_adapter) as the reference model for a PeftModel policy, so no
#    separate ref_model is constructed here.
# ===========================================================================

log.info(f"Loading tokenizer from: {PREV_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(
    str(PREV_MODEL_DIR),
    trust_remote_code=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

log.info(f"Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="bfloat16",
    device_map={"": 0},           # single GPU, same as CE / GRPO stages
    trust_remote_code=True,
    attn_implementation="sdpa",
)

log.info(f"Attaching warm-start LoRA adapter (trainable): {PREV_MODEL_DIR}")
model = PeftModel.from_pretrained(
    base_model,
    str(PREV_MODEL_DIR),
    is_trainable=True,
)
model.config.use_cache = False  # required with gradient checkpointing
log.info("Model ready.")


# ===========================================================================
# 3b. Prompt-length audit (mirrors Stage_GRPO_Instruct_DEV.py Section 3b)
# ===========================================================================

def _prompt_len(example):
    example["prompt_len"] = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
    return example

raw_datasets = raw_datasets.map(_prompt_len)

for split in raw_datasets:
    lengths = raw_datasets[split]["prompt_len"]
    print(f"\n{split} prompt length stats:")
    print(f"  Min    : {min(lengths)}")
    print(f"  Max    : {max(lengths)}")
    print(f"  Mean   : {sum(lengths)/len(lengths):.1f}")
    print(f"  > 512  : {sum(1 for l in lengths if l > 512)} samples")
    print(f"  > 768  : {sum(1 for l in lengths if l > 768)} samples")

# Cap raised 768 -> 928 to match max_prompt_length in DPOConfig below
# (928 + 96-token completion = 1024 total, per the requested max_seq_length).
# Same rationale as Stage_GRPO_Instruct_DEV.py otherwise: don't truncate a
# broken sentence, drop it instead.
MAX_PROMPT_TOKENS = 928
# Named here (not just inline in DPOConfig) so F1EvalCallback's max_new_tokens
# below can reference the SAME value rather than a second hardcoded "96" that
# could silently drift out of sync with DPOConfig's max_completion_length.
MAX_COMPLETION_TOKENS = 96
raw_datasets = raw_datasets.filter(lambda ex: ex["prompt_len"] <= MAX_PROMPT_TOKENS)
for split in raw_datasets:
    print(f"{split} after filter: {len(raw_datasets[split]):,} examples")

# Print the real expected step count now that we know the post-filter train
# size -- mirrors Stage_GRPO_Instruct_DEV.py's "[GRPO step estimate]" block.

_n_train = len(raw_datasets["train"])
_effective_batch = 2 * 4   # per_device_train_batch_size * gradient_accumulation_steps
_epochs = 1
_steps_per_epoch = _n_train // _effective_batch
_total_steps = _steps_per_epoch * _epochs
print(
    f"\n[DPO step estimate] train examples={_n_train:,} | "
    f"effective_batch={_effective_batch} | epochs={_epochs} -> "
    f"steps/epoch={_steps_per_epoch:,} | total_steps={_total_steps:,}\n"
)


# ===========================================================================
# 4. Requeue callback -- identical to Stage_GRPO_Instruct_DEV.py's, so a
#    SLURM SIGUSR1 requeue mid-DPO-run behaves the same way.
# ===========================================================================

class RequeueCallback(transformers.TrainerCallback):
    """Catches SIGUSR1 from SLURM and saves a checkpoint before the job dies."""

    def __init__(self):
        self._requeue_requested = False

    def register_signal(self):
        def _handler(signum, frame):
            log.warning(
                "SIGUSR1 received -- SLURM time limit approaching. "
                "Saving checkpoint and stopping cleanly ..."
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
            control.should_save          = True
            control.should_training_stop = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log.info(f"Eval metrics: {metrics}")
        if self._requeue_requested:
            log.warning("Signal detected during validation! Stopping early.")
            control.should_training_stop = True


# ===========================================================================
# 4b. F1 eval callback -- generates completions for the FULL test.json set
#     every F1_EVAL_STEPS steps and scores them with qasrl_reward_full(),
#     independent of (and much less frequent than) DPOTrainer's own
#     eval_steps=200 pairwise-loss eval. See the "TWO SEPARATE EVAL LOOPS"
#     module docstring section above for why these are kept apart.
# ===========================================================================

# Dedicated writer at LOG_DIR (not the f1_eval/ subdir) so
# loss/eval_loss/reward scalars land alongside where TensorBoard's own
# report_to="tensorboard" callback WOULD have written them, had it
# actually attached. See Stage_GRPO_Instruct_DEV.py for the same fix and
# rationale: is_tensorboard_available() can silently return False in the
# actual training job even when it checks out fine interactively, causing
# report_to="tensorboard" to no-op with no error/warning either way --
# we don't rely on it at all anymore, we just mirror the scalars ourselves.
_tb_writer = SummaryWriter(log_dir=str(LOG_DIR))


class F1EvalCallback(transformers.TrainerCallback):
    """
    Every `eval_every` optimizer steps: generates a completion for every
    (sentence, predicate) in `test_samples`, parses it back into QA pairs,
    and scores it against gold with qasrl_reward_full() -- the SAME metric
    Stage_GRPO_Instruct_DEV.py reports as `eval_test/f1`, so the numbers
    are directly comparable across the two training stages.

    Deliberately does NOT hook into DPOTrainer's compute_metrics /
    Trainer.evaluate() path -- that path only ever does the pairwise
    chosen/rejected forward passes DPO's loss needs, there's no supported
    way to make it also run generation. This is a fully separate pass:
    it flips the model to eval mode, generates, scores, flips back, and
    tracks its own best-F1 bookkeeping (see save_dir below), since
    `load_best_model_at_end` has no visibility into a metric computed
    outside DPOTrainer's own eval loop.

    Writes scalars to its own TensorBoard log dir (`f1_eval/` under
    LOG_DIR) so the curve shows up alongside DPOTrainer's own
    report_to="tensorboard" logs, under a distinctly-named run.
    """

    def __init__(
        self,
        tokenizer,
        test_samples: list[dict],
        eval_every: int,
        max_new_tokens: int,
        max_prompt_tokens: int,
        save_dir: Path,
        log_dir: Path,
        gen_batch_size: int = 8,  # tune to GPU memory -- this is a SEPARATE
                                   # generation pass, not gated by DPOConfig's
                                   # per_device_*_batch_size.
    ):
        self.tokenizer = tokenizer
        self.test_samples = test_samples
        self.eval_every = eval_every
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.save_dir = Path(save_dir)
        self.gen_batch_size = gen_batch_size
        self.best_f1 = -1.0
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / "f1_eval"))

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Mirror DPOTrainer's own loss-family scalars into _tb_writer,
        # independent of whether report_to="tensorboard"'s built-in
        # TensorBoardCallback actually attached in this job (it can
        # silently no-op with no warning -- see Stage_GRPO_Instruct_DEV.py's
        # identical fix). eval_dataset here is a single dataset (not a
        # dict like GRPO's dev/test split), so HF uses the plain "eval_"
        # prefix -- no eval_dev_/eval_test_ splitting needed here.
        logs = logs or {}
        for key in ("loss", "grad_norm", "learning_rate",
                    "rewards/chosen", "rewards/rejected",
                    "rewards/accuracies", "rewards/margins",
                    "logps/chosen", "logps/rejected"):
            if key in logs:
                _tb_writer.add_scalar(f"train/{key}", logs[key], state.global_step)
        for key in ("eval_loss", "eval_rewards/chosen", "eval_rewards/rejected",
                    "eval_rewards/accuracies", "eval_rewards/margins",
                    "eval_logps/chosen", "eval_logps/rejected"):
            if key in logs:
                tag = key.replace("eval_", "eval/", 1)
                _tb_writer.add_scalar(tag, logs[key], state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0:
            return control
        if state.global_step % self.eval_every != 0:
            return control
        self._run_eval(model, state)
        return control

    def _run_eval(self, model, state):
        log.info(
            f"[F1EvalCallback] step {state.global_step}: running full test.json "
            f"generation-F1 eval ({len(self.test_samples):,} groups) ..."
        )

        was_training = model.training
        prev_use_cache = getattr(model.config, "use_cache", None)
        prev_padding_side = self.tokenizer.padding_side

        model.eval()
        model.config.use_cache = True  # generate() wants this on; training
                                        # runs with it off for grad checkpointing
        self.tokenizer.padding_side = "left"  # required for correct batch
                                               # generation with a decoder-only
                                               # model -- restored below

        f1s, precisions, recalls = [], [], []
        t0 = time.time()

        with torch.no_grad():
            for i in range(0, len(self.test_samples), self.gen_batch_size):
                batch = self.test_samples[i: i + self.gen_batch_size]
                prompts = [b["prompt"] for b in batch]
                enc = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_prompt_tokens,
                ).to(model.device)

                gen = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # greedy -- this is a reproducible
                                      # checkpoint-comparison metric, not a
                                      # diversity-seeking rollout like GRPO's
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                gen_only = gen[:, enc["input_ids"].shape[1]:]
                completions = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

                for completion, sample in zip(completions, batch):
                    pred_qas = parse_completion_for_f1(completion)
                    result = qasrl_reward_full(pred_qas, sample["gold_qas"], sample["sentence"])
                    f1s.append(result["f1"])
                    precisions.append(result["precision"])
                    recalls.append(result["recall"])

        elapsed = time.time() - t0
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        mean_p = sum(precisions) / len(precisions) if precisions else 0.0
        mean_r = sum(recalls) / len(recalls) if recalls else 0.0

        log.info(
            f"[F1EvalCallback] step {state.global_step}: eval_test/f1={mean_f1:.4f} "
            f"precision={mean_p:.4f} recall={mean_r:.4f} (n={len(f1s)}, {elapsed:.1f}s)"
        )
        self.writer.add_scalar("eval_test/f1", mean_f1, state.global_step)
        self.writer.add_scalar("eval_test/precision", mean_p, state.global_step)
        self.writer.add_scalar("eval_test/recall", mean_r, state.global_step)
        self.writer.flush()

        # Restore training-mode state exactly as it was before this ran.
        self.tokenizer.padding_side = prev_padding_side
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        if was_training:
            model.train()

        if mean_f1 > self.best_f1:
            log.info(
                f"[F1EvalCallback] New best eval_test/f1: {mean_f1:.4f} "
                f"(previous best: {self.best_f1:.4f}) -- saving adapter to {self.save_dir}"
            )
            self.best_f1 = mean_f1
            self.save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(self.save_dir))
            self.tokenizer.save_pretrained(str(self.save_dir))


# ===========================================================================
# 5. DPO Config & Trainer
# ===========================================================================

dpo_config = DPOConfig(
    # ── Output paths ──────────────────────────────────────────────────
    output_dir  = str(CHECKPOINT_DIR),
    logging_dir = str(LOG_DIR),

    # ── DPO-specific ─────────────────────────────────────────────────
    # beta: KL-anchor strength against the frozen CE reference. SET TO 0.2
    # (not GRPO's beta=0.04) per the requested recipe -- this is a real
    # behavior change, not just matching a number: 0.2 anchors DPO much
    # more tightly to the CE checkpoint than the GRPO stage was allowed to
    # drift. 
    beta = float(os.environ.get("DPO_BETA", "0.2")),
    # NOTE: newer trl removed DPOConfig's separate max_prompt_length /
    # max_completion_length arguments.
    max_length = MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS,
    learning_rate               = float(os.environ.get("DPO_LR", "5e-6")),
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    gradient_accumulation_steps = 4,
    num_train_epochs            = float(os.environ.get("DPO_EPOCHS", "2")),
    bf16                        = True,
    gradient_checkpointing      = True,
    optim                       = "paged_adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "linear",
    warmup_ratio                = 0.1,

    # ── Logging & Saving ─────────────────────────────────────────────
    logging_strategy       = "steps",
    logging_steps           = 10,
    save_strategy           = "steps",
    save_steps               = 50,
    eval_strategy            = "steps" if "eval" in raw_datasets else "no",
    eval_steps               = 50,
    save_total_limit        = 6,
    load_best_model_at_end  = False,
    report_to               = "tensorboard",

    # DPOConfig-specific: our "prompt" column is already a fully rendered,
    # chat-templated string (built by build_dpo_training_data.py using the
    # same tokenizer.apply_chat_template() call the CE/GRPO stages use) --
    # NOT a list of {"role", "content"} messages. Telling DPOTrainer the
    # dataset is already in "standard" (plain-text) format, not
    # "conversational", stops it from trying to re-apply a chat template
    # on top of an already-templated string.
    dataset_num_proc = 4,
)

requeue_cb = RequeueCallback()
requeue_cb.register_signal()

# F1_EVAL_STEPS is deliberately independent from (and much larger than)
# DPOConfig's eval_steps=200 above -- this pass does real generation over
# the full test set, so it costs meaningfully more per call. Tune based on
# the per-call wall-clock F1EvalCallback logs the first time it runs; 1000-
# 2000 is a reasonable starting point for a ~1,000-step total run (see the
# "[DPO step estimate]" print above) -- i.e. a handful of F1 checkpoints
# across the whole run, not one every eval_steps.
# ===========================================================================
# F1EvalCallback is DELIBERATELY NOT INSTALLED.
#
# Checkpoint selection happens OUT OF BAND and AFTER training, by greedy
# generation on the held-out DEV slice (dpo_val.jsonl) -- see eval_on_val.py.
#
# Not installing it is also a large speedup: in-training full-split generation
# every 100 steps dominates wall-clock over a ~300-step run, starving training
# of compute. The class is left defined above (unused) rather than deleted.
# ===========================================================================

trainer = DPOTrainer(
    model            = model,
    args             = dpo_config,
    train_dataset    = raw_datasets["train"],
    eval_dataset     = raw_datasets["eval"] if "eval" in raw_datasets else None,
    processing_class = tokenizer,
    callbacks        = [requeue_cb],
)


# ===========================================================================
# 6. Train (with checkpoint resumption) -- identical pattern to
#    Stage_GRPO_Instruct_DEV.py Section 7.
# ===========================================================================

ckpt_glob = glob.glob(str(CHECKPOINT_DIR / "checkpoint-*"))
resume    = sorted(ckpt_glob, key=os.path.getmtime)[-1] if ckpt_glob else None
if resume:
    log.info(f"Resuming from checkpoint: {resume}")

_train_start = time.time()
trainer.train(resume_from_checkpoint=resume)
_train_elapsed = time.time() - _train_start

_hours, _rem  = divmod(int(_train_elapsed), 3600)
_mins, _secs  = divmod(_rem, 60)
log.info(f"Training complete. Total time: {_hours}h {_mins:02d}m {_secs:02d}s ({_train_elapsed:.1f} s)")

# ===========================================================================
# 7. Save
# ===========================================================================

trainer.model.save_pretrained(str(MODEL_SAVE_DIR))
tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
log.info(f"DPO stage complete. Final adapter saved to {MODEL_SAVE_DIR}")
# Checkpoints are selected post-hoc on the DEV holdout via eval_on_val.py.
log.info(
    f"Select a checkpoint from {CHECKPOINT_DIR} using eval_on_val.py on the dev holdout."
)

final_metrics = trainer.evaluate()
log.info(f"Final eval: {final_metrics}")