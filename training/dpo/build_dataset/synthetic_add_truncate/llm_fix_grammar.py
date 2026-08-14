# ==========================================================================
# Second pass over build_dpo_data.py's output: re-generates the questions of
# Add-injected (is_foreign) pairs so the borrowed question is grammatical for its new
# predicate. Runs a LOCAL vLLM model on a GPU node -- no external API, no credentials.
# Reads dpo_pairs.json, writes dpo_pairs.grammar_fixed.json.
#
# Part of the SUPERSEDED add/truncate DPO arm -- kept for the record only.
# The reported DPO result (79.59) came from the on-policy pairs in ../ .
# ==========================================================================
"""
llm_fix_grammar.py

Second-pass grammar correction for Add-step questions flagged by
build_dpo_data.py (qa.needs_review == True and qa.is_foreign == True).

============================================================================
WHY THIS IS A SEPARATE SCRIPT
============================================================================
build_dpo_data.py needs to run on a CPU-only login/build node (it's just
JSON wrangling + string ops). This script needs a GPU + vLLM, which is a
much heavier dependency footprint and should be a separate cluster job step.
Keeping them separate means:
  - build_dpo_data.py stays fast and trivially runnable anywhere.
  - This script can be submitted as its own SLURM job (e.g. `sbatch` with a
    GPU partition) without dragging GPU requirements into the main pipeline.
  - If the LLM step fails, breaks, or you want to swap models, you re-run
    only this script, not the whole data-construction pipeline.

============================================================================
WHAT THIS SCRIPT DOES
============================================================================
1. Loads dpo_pairs.json (the output of build_dpo_data.py).
2. Collects EVERY rejected QA pair where qa["is_foreign"] is True -- i.e.
   every Add-step question, not just the subset build_dpo_data.py already
   flagged with needs_review=True (its deterministic verb-slot swap flags
   the cases it suspects are ungrammatical -- see `adapt_question_to_predicate`
   there). This script runs on all of them anyway, as a second, independent
   pass; `needs_review` is carried along per-pair purely as provenance for
   why build_dpo_data.py did or didn't suspect an issue, not as a filter.
   The prompt tells the model to return already-correct questions unchanged,
   so a pair that didn't need fixing just comes back as-is (see rule 5 in
   PROMPT_TEMPLATE) and costs a cheap batched call, not a wrong edit.
3. Batches them all into one vLLM `.generate()` call (vLLM's whole point is
   batched throughput -- looping one prompt at a time wastes that).
4. Parses each model response as strict JSON (see PROMPT_TEMPLATE below)
   and validates the result with cheap sanity checks (non-empty, ends in
   '?', not wildly different in length from the original, no injected
   content). Three outcomes:
     - PARSE_FAILED: no usable candidate at all. Keeps the ORIGINAL
       deterministic question, needs_review forced to True.
     - VALIDATION_FAILED: the LLM DID produce a candidate, it just tripped
       one of the mechanical guards. Since those guards are deliberately
       strict rather than a semantic judge, the candidate is often still
       an improvement over the pre-LLM text -- so by default
       (APPLY_LLM_VERSION_ON_VALIDATION_FAILURE=True) it's applied anyway,
       with needs_review FORCED to True regardless of its prior value so
       it always surfaces in review_server.py for a human to confirm or
       revert. Flip that constant to False to instead keep the original
       deterministic question here, matching the old conservative
       behavior.
     - OK: passed every check. Applied, needs_review cleared.
   This script never silently drops or blanks a question either way.
5. Writes a NEW file (does not overwrite dpo_pairs.json) with every
   applied pair's `question` updated in place, `needs_review` updated per
   the rules above, and a new `llm_grammar_fix` field recording what
   happened -- including an `applied_despite_validation_failure` flag when
   relevant -- so you can audit LLM edits separately from the
   deterministic ones, or revert them.
6. ALSO writes, for every full (non-sample) run:
     - a changes log ("<output>.changes.txt" by default): plain-text
       original -> fixed pairs, split into successful fixes vs. left
       unchanged, for quick skimming.
     - a timestamped unified diff ("changes_<timestamp>.diff", written into
       --diff-dir, default the current directory): a REAL diff (same format
       as `diff -u`) of the source questions vs. this run's result. Every
       run gets its own timestamped file -- nothing is overwritten -- so
       you can compare two different runs (different prompt versions,
       different models, different temperature, etc) by running
       `diff changes_<run_a>.diff changes_<run_b>.diff` directly.

============================================================================
SAMPLE MODE
============================================================================
Run with --sample N first (e.g. --sample 30) to process only a random
subset and print every (original, fixed) pair to stdout for you to
eyeball quality before committing to the full ~350-example run. Nothing is
written to disk in sample mode unless you also pass --write-sample-output.

============================================================================
USAGE
============================================================================
These are the arguments this script accepts. On THIS cluster, don't run
this .py file directly -- submit it via `sbatch run_grammar_fix.sbatch`,
which invokes one of the commands below from inside an actual GPU job (see
the GPU REQUIREMENT section further down for why).

    # Quality check on 30 random flagged examples, print only, no file written:
    python3 llm_fix_grammar.py --input dpo_pairs.json --sample 30

    # Full run over all flagged examples, writes dpo_pairs.grammar_fixed.json
    # plus a changes log and a timestamped diff file:
    python3 llm_fix_grammar.py --input dpo_pairs.json --output dpo_pairs.grammar_fixed.json --diff-dir ./diffs

    # Compare two runs directly (e.g. after changing PROMPT_TEMPLATE or MODEL_NAME):
    diff ./diffs/changes_20260625_140000.diff ./diffs/changes_20260625_153000.diff

============================================================================
MODEL CHOICE
============================================================================
Defaults to Qwen2.5-7B-Instruct -- a 7B instruct model is comfortably
sufficient for a narrow, mechanical edit task like this (fix tense/agreement,
don't change meaning) and runs fast in vLLM. Change MODEL_NAME below to swap
to a different local checkpoint you already have (e.g. a Qwen3 variant) --
nothing else in the script needs to change for a same-family model swap.

============================================================================
GPU REQUIREMENT -- READ THIS BEFORE RUNNING
============================================================================
This script REQUIRES a visible GPU. vLLM does not have a usable CPU
inference path for a model this size -- it will error out (not silently
fall back to CPU) if no CUDA device is visible. Nothing about how you
invoke `python3 llm_fix_grammar.py` makes this happen automatically: it
depends entirely on the environment the process runs in.

On this cluster, GPUs are only available via `sbatch` (no `srun` for
interactive GPU allocations) -- submit the accompanying `run_grammar_fix.sbatch`
job script instead of running this file directly:

    sbatch run_grammar_fix.sbatch

That script requests the GPU partition under the configured SLURM account,
mirroring the existing Qwen3 training job's structure. Edit the script to
pass `--sample 30` instead of the full run the first time -- see the
comment inside it for the exact line to change. Running this .py file
directly on a login node (with no GPU allocated) will fail.

To make that failure clear and immediate rather than a confusing crash deep
inside vLLM's model loading, `check_gpu_available()` below checks for a
visible CUDA device via `torch.cuda.is_available()` BEFORE attempting to
load the model, and exits with a clear error message if none is found.

The GPU_DEVICE_INDEX / TENSOR_PARALLEL_SIZE / GPU_MEMORY_UTILIZATION
constants below are passed explicitly to vLLM's `LLM(...)` constructor --
previously this script let vLLM pick its own defaults for all of these,
which is exactly the kind of implicit-and-invisible behavior worth avoiding
on a shared cluster where you don't want a job grabbing more of a GPU (or
the wrong GPU) than intended.

============================================================================
NOT YET RUN AGAINST A REAL GPU -- PLEASE SMOKE-TEST
============================================================================
This script was written without GPU/vLLM access in the environment it was
authored in. The prompt construction, JSON-parsing, validation, and
merge-back logic are all unit-testable without a GPU and HAVE been tested
that way (see the accompanying test notes). The actual vLLM API calls
(`LLM(...)`, `.generate(...)`, `SamplingParams(...)`) are written against
vLLM's documented interface but have NOT been executed end-to-end here --
please run the --sample mode first and let me know if vLLM's API surface
has changed or behaves differently in your installed version.
"""

import argparse
import difflib
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # change to a local checkpoint path if you have one downloaded
MAX_NEW_TOKENS = 256  # bumped up slightly from 200 as headroom -- with llm.chat() the
# model should now go straight to the JSON, but this leaves a small margin
# rather than risking truncation mid-object again if it occasionally adds
# a short lead-in.
TEMPERATURE = 0.0  # deterministic -- this is a correction task, not a creative one
RANDOM_SEED = 42

# GPU configuration -- passed explicitly to vLLM rather than relying on its
# defaults. See the "GPU REQUIREMENT" section in the module docstring above.
GPU_MEMORY_UTILIZATION = 0.85  # fraction of GPU memory vLLM is allowed to claim
TENSOR_PARALLEL_SIZE = 1       # set >1 only if splitting one model across multiple GPUs

# Cheap sanity bounds on the LLM's output, applied AFTER JSON parsing.
# These catch "the model went off the rails" cases without needing a second
# model call to judge the judge.
MAX_WORD_INCREASE = 4    # fixed question can't have more than 4 extra words vs the original
MAX_WORD_DECREASE = 2    # or fewer than (original - 2) words -- catches truncated outputs
# (Switched from a character-length ratio to a word-count difference: char
# ratio is a poor gate on short questions -- "who stop?" -> "who stopped
# something?" is a completely legitimate fix (it even restores the missing
# object placeholder) but is 2.4x the original's CHARACTER length purely
# because the original is so short. Word-count difference doesn't have that
# problem: it's +1 word either way, regardless of how short the base
# question is. The word-set check below is the primary defense against real
# content injection (extra names/topics); this is just a cheap shape check
# to catch "the model went off the rails" and produced multiple sentences,
# a rambling answer, etc.)

# Words the LLM is allowed to ADD that weren't in the original question --
# these are exactly the "missing auxiliary verb" fixes this pipeline exists
# for (e.g. adding "did" or "does"), not content injection. Anything else
# new gets rejected by validate_fixed_question's word-set check.
AUXILIARY_WHITELIST = {
    "did", "does", "do", "is", "are", "was", "were",
    "has", "have", "had", "will", "would", "can", "could",
}

# Some upstream-generated questions are missing their object placeholder
# entirely (e.g. "who stop?" instead of "who stop something?" -- a template
# bug, not something this pass is meant to fix). When that happens, the
# grammatically correct fix often restores the missing placeholder (e.g.
# "who stopped something?"), which is a legitimate repair -- exactly as
# legitimate as adding a missing auxiliary -- not content injection. So
# "someone"/"something" are allowed to be ADDED here, same as the auxiliary
# whitelist above. (Rule 3 in the prompt still forbids the model from
# REPLACING an existing placeholder with something more specific -- that's
# a separate, still-enforced check below.)
PLACEHOLDER_WHITELIST = {"someone", "something", "somewhere"}

# Small closed class of connector prepositions used across different verb
# templates ("consist OF something", "listen TO something", "look AT
# something", "remind someone OF something", ...). Like the placeholders
# above, these are template scaffolding, not semantic content -- a given
# verb may or may not grammatically take one, so the model adding/dropping
# one to match the target verb's actual argument structure (e.g. "project"
# doesn't take "of": "what project of something?" -> "what does something
# project?") is a legitimate repair, not content injection or content loss.
TEMPLATE_PREPOSITION_WHITELIST = {
    "of", "to", "in", "at", "from", "with", "about", "on", "for", "as",
}

# Common irregular verbs whose past tense / participle forms don't share a
# prefix with the base verb (say -> said, drive -> drove, go -> went, etc).
# The prefix-based stem check below handles regular inflection fine, but it
# mistakes these for content injection since e.g. "said" doesn't start with
# "say". Mapping base verb -> its irregular forms lets the content-injection
# check explicitly allow them through.
IRREGULAR_VERB_FORMS = {
    "be": {"am", "is", "are", "was", "were", "been", "being"},
    "go": {"went", "gone", "going", "goes"},
    "say": {"said", "saying", "says"},
    "do": {"did", "done", "doing", "does"},
    "have": {"had", "having", "has"},
    "make": {"made", "making", "makes"},
    "take": {"took", "taken", "taking", "takes"},
    "see": {"saw", "seen", "seeing", "sees"},
    "come": {"came", "coming", "comes"},
    "give": {"gave", "given", "giving", "gives"},
    "find": {"found", "finding", "finds"},
    "think": {"thought", "thinking", "thinks"},
    "tell": {"told", "telling", "tells"},
    "become": {"became", "becoming", "becomes"},
    "leave": {"left", "leaving", "leaves"},
    "feel": {"felt", "feeling", "feels"},
    "bring": {"brought", "bringing", "brings"},
    "begin": {"began", "begun", "beginning", "begins"},
    "keep": {"kept", "keeping", "keeps"},
    "hold": {"held", "holding", "holds"},
    "write": {"wrote", "written", "writing", "writes"},
    "stand": {"stood", "standing", "stands"},
    "hear": {"heard", "hearing", "hears"},
    "let": {"let", "letting", "lets"},
    "mean": {"meant", "meaning", "means"},
    "set": {"set", "setting", "sets"},
    "meet": {"met", "meeting", "meets"},
    "run": {"ran", "running", "runs"},
    "pay": {"paid", "paying", "pays"},
    "sit": {"sat", "sitting", "sits"},
    "speak": {"spoke", "spoken", "speaking", "speaks"},
    "lie": {"lay", "lain", "lying", "lies"},
    "lead": {"led", "leading", "leads"},
    "read": {"read", "reading", "reads"},
    "grow": {"grew", "grown", "growing", "grows"},
    "lose": {"lost", "losing", "loses"},
    "fall": {"fell", "fallen", "falling", "falls"},
    "send": {"sent", "sending", "sends"},
    "build": {"built", "building", "builds"},
    "understand": {"understood", "understanding", "understands"},
    "draw": {"drew", "drawn", "drawing", "draws"},
    "break": {"broke", "broken", "breaking", "breaks"},
    "spend": {"spent", "spending", "spends"},
    "cut": {"cut", "cutting", "cuts"},
    "rise": {"rose", "risen", "rising", "rises"},
    "drive": {"drove", "driven", "driving", "drives"},
    "buy": {"bought", "buying", "buys"},
    "wear": {"wore", "worn", "wearing", "wears"},
    "choose": {"chose", "chosen", "choosing", "chooses"},
    "eat": {"ate", "eaten", "eating", "eats"},
    "win": {"won", "winning", "wins"},
    "teach": {"taught", "teaching", "teaches"},
    "sell": {"sold", "selling", "sells"},
    "catch": {"caught", "catching", "catches"},
    "fly": {"flew", "flown", "flying", "flies"},
    "fight": {"fought", "fighting", "fights"},
    "throw": {"threw", "thrown", "throwing", "throws"},
    "shoot": {"shot", "shooting", "shoots"},
    "arise": {"arose", "arisen", "arising", "arises"},
    "swear": {"swore", "sworn", "swearing", "swears"},
    "seek": {"sought", "seeking", "seeks"},
    "sleep": {"slept", "sleeping", "sleeps"},
    "dig": {"dug", "digging", "digs"},
    "deal": {"dealt", "dealing", "deals"},
    "get": {"got", "gotten", "getting", "gets"},
    "know": {"knew", "known", "knowing", "knows"},
}

PROMPT_TEMPLATE = """You fix the grammar of auto-generated template questions. Nothing else.

These questions are built from a template like "who VERBs something?" by swapping in a different verb. The swap sometimes breaks subject-verb agreement or tense (e.g. "who post something?" should be "who posts something?").

THE TENSE OF THE FIXED QUESTION MUST MATCH THE TENSE THE VERB HAS IN THE ORIGINAL SENTENCE BELOW.
Do not default to present tense. Read how "{verb_form}" is actually used in the sentence -- past, present, etc. -- and produce a question in that SAME tense. There are two question shapes, and they take tense differently:
  - SUBJECT questions, where "who"/"what" IS the subject (e.g. "who post something?" = who is the one posting): tense is carried by the verb itself, NOT by inserting an auxiliary. Present: "who posts something?" (add -s). Past: "who posted something?" (inflect to past). Never turn this into "who did someone post something?" -- that wrongly duplicates the subject ("who" and "someone" can't both be the subject).
  - OBJECT questions, where "someone"/"something" already occupies the subject slot before the verb (e.g. "what did someone arrest?" / "what involve something?" once corrected to subject-first order): tense is carried by an inserted auxiliary + the base verb. Present: "what does something involve?". Past: "what did something involve?".
  - If unsure from the sentence, prefer the tense already implied by the ORIGINAL question over silently defaulting to present.

Original sentence (use this to determine the correct tense): "{sentence}"

STRICT RULES:
1. Only fix grammar: verb tense, subject-verb agreement, missing auxiliary verbs. That's it.
2. NEVER add any information that isn't already in the question. No names, no topics, no extra detail, no answering the question.
3. NEVER remove the placeholder words "someone" or "something" -- keep them exactly as they are. Do not replace them with anything more specific.
4. NEVER change capitalization unless you are also fixing a real grammar issue. Capitalization alone is not a fix.
5. If the question is already grammatical AND already matches the sentence's tense, return it completely unchanged.
6. Keep the same question word (who/what/when/where/why/how) and the same sentence structure. Only the verb form/auxiliary may change.
7. The question you are given always starts with a capitalized WH-word. Your output must too -- never lowercase it.

Target verb (the question must grammatically use this verb): "{verb_form}"

EXAMPLES of correct fixes (note: the WH-word is always capitalized, both in the input you'll receive and the output you must produce):

  -- SUBJECT questions ("who"/"what" is the one doing the action) --
  sentence: "John posts updates every morning."
  input:  "Who post something?"          (verb: post, present tense in sentence)
  output: "Who posts something?"          (present: add -s only, "who" stays the subject)

  sentence: "John posted the update to the group yesterday."
  input:  "Who post something?"          (verb: post, PAST tense in sentence)
  output: "Who posted something?"         (past: inflect the verb itself -- do NOT insert "did someone")

  sentence: "The teacher explains the rule to the class."
  input:  "Who explain something?"        (verb: explain, present tense in sentence)
  output: "Who explains something?"       (present: add -s only)

  -- OBJECT questions ("someone"/"something" already sits in the subject slot) --
  sentence: "The officer arrested the suspect."
  input:  "What did someone arrest?"      (verb: arrest, past tense in sentence)
  output: "What did someone arrest?"      (already correct: "did" + base verb, past -- returned unchanged)

  sentence: "The new policy involves several departments."
  input:  "What involve something?"
  output: "What does something involve?"  (present: "does" + base verb, "something" as subject)

  sentence: "The new policy involved several departments last year."
  input:  "What involve something?"
  output: "What did something involve?"   (past: "did" + base verb, matches sentence's past tense)

EXAMPLES of WRONG fixes (do not do this):
  sentence: "Someone said the plan was ready."
  input:  "What say something?"
  WRONG:  "What did John Walker say?"     (added a name that wasn't there -- NEVER do this)
  WRONG:  "What says something?"          (defaulted to present -- sentence is PAST tense)
  WRONG:  "what did someone say?"          (lowercase WH-word -- always keep it capitalized)
  RIGHT:  "What did someone say?"          (past tense via "did", kept "someone", capitalized)

  sentence: "John posted the update to the group yesterday."
  input:  "Who post something?"
  WRONG:  "Who did someone post something?"  (duplicates the subject -- "who" is ALREADY the subject, don't also insert "someone")
  RIGHT:  "Who posted something?"            (past tense carried by the verb alone)

  sentence: "The new policy involves several departments."
  input:  "What involve something?"
  WRONG:  "What companies involve something with AI?"   (added topic/subject info -- NEVER do this)
  RIGHT:  "What does something involve?"   (fixed agreement only, present tense matches sentence, kept "something")

Now fix this one. Original question: "{question}"

Respond with ONLY a JSON object, no other text, in exactly this format:
{{"fixed_question": "...", "changed": true_or_false}}
"""


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

@dataclass
class FlaggedQuestion:
    record_idx: int          # index into the top-level records list
    qa_idx: int               # index into that record's "rejected" list
    original_question: str
    sentence_text: str
    verb_form: str
    prior_attempt: Optional[dict] = None  # previous llm_grammar_fix dict, if any (retry mode)


def load_flagged_questions(records: list[dict], mode: str = "fresh") -> list[FlaggedQuestion]:
    """
    Scans every record's rejected list for foreign (Add-step) pairs.

    Every `is_foreign` pair is sent through the LLM fix -- not just the
    subset build_dpo_data.py already flagged with `needs_review=True`
    (those are the ones its deterministic verb-slot swap suspects are
    ungrammatical; see `adapt_question_to_predicate` there). We still run on
    the rest too so a second, independent pass double-checks the ones the
    deterministic swap thought were fine -- `needs_review` is kept around on
    each pair purely as provenance (why build_dpo_data.py did or didn't
    suspect an issue), it's no longer what gates whether this script looks
    at a pair.

    mode="fresh": process every foreign item, regardless of whether it
        already carries an `llm_grammar_fix` field from an earlier run
        (that field, if present, is ignored).
    mode="retry": only process foreign items that already carry an
        `llm_grammar_fix` field with a non-OK status (i.e. a previous run
        attempted and failed on them). Items that have never been attempted
        are skipped -- run in "fresh" mode first to attempt those. The
        previous attempt's detail is carried along on `prior_attempt` so
        `build_prompt` can show the model what went wrong last time.
    """
    flagged = []
    skipped_never_attempted = 0
    for r_idx, record in enumerate(records):
        sentence_text = record.get("sentence_text", "")
        for qa_idx, qa in enumerate(record.get("rejected", [])):
            if not qa.get("is_foreign"):
                continue
            prior = qa.get("llm_grammar_fix")
            if mode == "retry":
                if prior is None or prior.get("status") == "OK":
                    skipped_never_attempted += 1
                    continue
            flagged.append(FlaggedQuestion(
                record_idx=r_idx,
                qa_idx=qa_idx,
                original_question=qa["question"],
                sentence_text=sentence_text,
                verb_form=qa.get("verb_form", ""),
                prior_attempt=prior if mode == "retry" else None,
            ))
    if mode == "retry" and skipped_never_attempted:
        log.info(f"Retry mode: skipping {skipped_never_attempted} flagged item(s) that were never "
                 f"attempted before (run in fresh mode first to attempt those).")
    return flagged


# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------

def build_prompt(fq: FlaggedQuestion) -> str:
    prompt = PROMPT_TEMPLATE.format(
        verb_form=fq.verb_form,
        sentence=fq.sentence_text.replace('"', "'"),
        question=fq.original_question.replace('"', "'"),
    )
    if fq.prior_attempt is not None:
        prev_fixed = fq.prior_attempt.get("applied_question") or fq.prior_attempt.get("raw_model_output", "")
        prev_detail = fq.prior_attempt.get("detail", "")
        prompt += (
            "\n\nNOTE: a previous attempt at this SAME question was rejected. "
            f"Previous attempt: \"{prev_fixed}\". "
            f"Why it was rejected: {prev_detail}. "
            "Produce a different, corrected fix that avoids that specific problem."
        )
    return prompt


# ----------------------------------------------------------------------------
# Response parsing + validation
# ----------------------------------------------------------------------------

def capitalize_question(question: str) -> str:
    """
    Capitalizes the first alphabetic character of a question string, leaving
    everything else untouched. Mirrors build_dpo_data.py's function of the
    same name (duplicated rather than imported, to keep this script's only
    hard dependency on the GPU/vLLM stack -- see the module docstring).

    Applied to every accepted LLM fix before it's written back: the few-shot
    examples in PROMPT_TEMPLATE are lowercase (matching dev.json's raw
    question casing), so the model reliably tends to answer in lowercase too
    even though the ORIGINAL question it was given was capitalized. Nothing
    else in this pipeline re-capitalizes that output, so without this call a
    successful grammar fix would silently decapitalize the question's WH-word
    on its way back into the dataset.
    """
    for i, ch in enumerate(question):
        if ch.isalpha():
            return question[:i] + ch.upper() + question[i + 1:]
    return question


def parse_llm_response(raw_text: str) -> Optional[dict]:
    text = raw_text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find the first '{' and decode only the first complete JSON object from
    # there, ignoring anything that follows (reasoning text before, a
    # duplicated second object after, trailing prose, etc). This is more
    # robust than a greedy `\{.*\}` regex, which spans multiple JSON objects
    # if the model emits more than one (e.g. reasoning + JSON, or JSON
    # repeated twice) and produces invalid concatenated JSON.
    start = text.find("{")
    if start == -1:
        return None

    decoder = json.JSONDecoder()
    try:
        parsed, _end_index = decoder.raw_decode(text, idx=start)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or "fixed_question" not in parsed:
        return None
    return parsed


def validate_fixed_question(original: str, fixed: str, verb_form: str = "") -> tuple[bool, str]:
    """
    Sanity checks on the LLM's proposed fix. Returns (is_valid, reason).

    Two kinds of checks:
      1. Cheap shape checks (empty, missing '?', wildly different length) --
         these catch "the model went off the rails" formatting failures.
      2. A content-injection check: every word in the ORIGINAL question
         (other than the verb itself, which is allowed to change form for
         tense/agreement, and template scaffolding -- placeholders
         someone/something/somewhere and connector prepositions like
         of/to/at, which may be added, dropped, or swapped to match the
         target verb's actual argument structure) must still appear
         somewhere in the FIXED question. This is the check that catches
         cases like "what say something?" -> "What did John Walker say?" --
         "John" and "Walker" are new words that weren't in the original,
         aren't the verb, and aren't scaffolding, so this fails it. It's
         deliberately a strict, mechanical check rather than a semantic one:
         a real grammar fix should only ever touch the verb area or
         scaffolding, never introduce new nouns/names/topics. A few
         legitimate function words (did/does/is/are/was/were/has/have) are
         allowed to be ADDED since that's exactly the missing-auxiliary fix
         this whole pipeline exists for -- but nothing else new is
         tolerated.

    This is NOT a grammatical-correctness judge (that's the LLM's job, and a
    human reviewer is the final backstop for anything subtler than this
    catches) -- it's a mechanical guard against the model wandering outside
    its narrow task.
    """
    if not fixed or not isinstance(fixed, str):
        return False, "Empty or non-string output."
    fixed = fixed.strip()
    if not fixed:
        return False, "Output was empty after stripping whitespace."
    if not fixed.endswith("?"):
        return False, "Fixed question does not end in '?'."

    def word_count(text: str) -> int:
        return len(re.findall(r"[a-zA-Z']+", text))

    orig_wc = word_count(original)
    fixed_wc = word_count(fixed)
    word_diff = fixed_wc - orig_wc
    if word_diff > MAX_WORD_INCREASE:
        return False, (
            f"Fixed question has {word_diff} more word(s) than the original "
            f"({orig_wc} -> {fixed_wc}, suspiciously long)."
        )
    if word_diff < -MAX_WORD_DECREASE:
        return False, (
            f"Fixed question has {-word_diff} fewer word(s) than the original "
            f"({orig_wc} -> {fixed_wc}, suspiciously short/truncated)."
        )

    # Content-injection check (word-set comparison).
    def normalize_words(text: str) -> set[str]:
        return set(re.findall(r"[a-z']+", text.lower()))

    original_words = normalize_words(original)
    fixed_words = normalize_words(fixed)
    verb_word = (verb_form or "").lower().strip()

    # General "nothing gets silently dropped" check: every word in the
    # original (other than the verb, which is allowed to change form) must
    # still appear in the fixed version. This has to run BEFORE we credit
    # the placeholder whitelist above, otherwise a REPLACEMENT like
    # "somewhere" -> "something" would slip through: "something" is a
    # legitimate addition on its own, but only if it's genuinely additive,
    # not swapped in for a different word that quietly disappeared.
    # AUXILIARY_WHITELIST is exempted here too, not just on the addition
    # side below -- a correct fix sometimes needs to swap one auxiliary for
    # another (e.g. "is" -> "does" to match a base-form verb like "think"),
    # or drop one entirely (a SUBJECT question -- "who VERBs something?" --
    # never takes an inserted auxiliary at all per the prompt's own rules,
    # so "Who was go something?" -> "Who went something?" is a legitimate
    # repair of a spuriously-inserted "was", not a content drop). It's a
    # small closed class of function words, not open-ended content, so
    # exempting it here doesn't weaken the check's real purpose: catching
    # new nouns/names/topics being introduced.
    dropped_words = (
        original_words - fixed_words - {verb_word}
        - PLACEHOLDER_WHITELIST - TEMPLATE_PREPOSITION_WHITELIST
        - AUXILIARY_WHITELIST
    )
    if dropped_words:
        return False, (
            f"Fixed question dropped word(s) that were in the original and "
            f"aren't the verb: {sorted(dropped_words)} -- this usually means "
            f"a word got replaced/removed rather than just a grammar fix."
        )

    genuinely_new_words = (
        fixed_words - original_words - AUXILIARY_WHITELIST
        - PLACEHOLDER_WHITELIST - TEMPLATE_PREPOSITION_WHITELIST
    )
    # The verb is allowed to change FORM (post -> posts -> posted -> posting)
    # since that's the entire point of this fix. Rather than requiring an
    # exact match against the bare lemma, allow any new word that shares the
    # lemma's stem (a cheap prefix check -- not a real morphological
    # analyzer, but sufficient to cover regular English inflection: post/
    # posts/posted/posting, explain/explains/explained/explaining, etc).
    # Irregular verbs (say -> said, drive -> drove) don't share a prefix with
    # their base form, so they're allowed through via IRREGULAR_VERB_FORMS
    # instead.
    if verb_word:
        irregular_forms = IRREGULAR_VERB_FORMS.get(verb_word, set())
        genuinely_new_words = {
            w for w in genuinely_new_words
            if not (
                (w.startswith(verb_word[:max(3, len(verb_word) - 2)]) and len(verb_word) >= 3)
                or w in irregular_forms
            )
        }

    if genuinely_new_words:
        return False, (
            f"Fixed question introduces word(s) not in the original and not an "
            f"allowed auxiliary/verb: {sorted(genuinely_new_words)} -- this usually "
            f"means the model added information instead of just fixing grammar."
        )

    # Placeholder-dropping is caught by the general dropped_words check
    # above (which runs before the placeholder whitelist is applied), so
    # there's no separate check needed here.

    # Capitalization-only "fixes" aren't real fixes: the model needs to
    # actually address tense/agreement, not just capitalize the first
    # letter. Catch this here rather than relying on the prompt alone.
    if fixed.lower() == original.lower() and fixed != original:
        return False, (
            "Fixed question differs from the original ONLY in capitalization -- "
            "this isn't a grammar fix (no tense/agreement issue was actually "
            "addressed), so it's rejected rather than counted as a successful change."
        )

    return True, "OK"


# ----------------------------------------------------------------------------
# vLLM call (the only part requiring a GPU -- isolated here for clarity)
# ----------------------------------------------------------------------------

def check_gpu_available() -> None:
    """
    Fails fast and clearly if no CUDA device is visible, BEFORE attempting
    to load a multi-GB model into vLLM. Without this check, running this
    script on a CPU-only node (e.g. a SLURM login node, by accident) would
    fail anyway -- but with a confusing error buried inside vLLM/CUDA
    initialization rather than a clear, actionable message naming the real
    problem.

    Imports torch lazily (same reasoning as vllm being imported lazily in
    run_vllm_batch): keeps this script importable on a machine without
    torch/CUDA installed, for testing the non-GPU logic.
    """
    try:
        import torch
    except ImportError:
        log.error(
            "Could not import torch -- it should be installed as a vLLM "
            "dependency. If this is a fresh environment, install vllm first "
            "(`pip install vllm`), which will pull in a CUDA-enabled torch."
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        log.error(
            "No CUDA device is visible to this process (torch.cuda.is_available() "
            "is False). This script requires a GPU and will not run on CPU.\n\n"
            "You likely ran this .py file directly on a login node instead of "
            "through a GPU job. Submit it via:\n\n"
            "    sbatch run_grammar_fix.sbatch\n\n"
            "(this cluster only allocates GPUs through sbatch, not srun)."
        )
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if n_gpus > 0 else "unknown"
    log.info(f"GPU check passed: {n_gpus} CUDA device(s) visible (device 0: {gpu_name}).")


def run_vllm_batch(prompts: list[str]) -> list[str]:
    """
    Runs all prompts through vLLM in a single batched .chat() call and
    returns the raw text outputs, in the same order as the input prompts.

    IMPORTANT: this uses llm.chat(), NOT llm.generate(). Qwen2.5-7B-Instruct
    is an instruction-tuned model that expects its prompt wrapped in its
    chat template (<|im_start|>user ... <|im_end|> etc) so it knows a reply
    is expected. Passing raw strings to llm.generate() skips that template
    entirely -- the model then just does raw next-token continuation of the
    prompt text itself, rather than answering it. In practice that produced
    exactly the failure modes seen in early runs: the model would continue
    writing more of the PROMPT's own instructions instead of a response,
    fall into degenerate repetition loops with no proper stop token, or
    burn the whole token budget on unrequested reasoning before ever
    reaching the JSON (truncating it mid-object). llm.chat() applies the
    model's chat template automatically and gives it a clear "assistant
    turn to fill in," which is what actually makes it behave like an
    instruct model rather than a base model.

    NOTE: imports vllm lazily (inside the function) so that the rest of this
    script -- prompt construction, parsing, validation, merge logic -- can be
    imported and unit-tested on a machine without vllm/GPU installed (e.g.
    for CI or for testing on a login node before submitting the GPU job).
    """
    check_gpu_available()

    from vllm import LLM, SamplingParams

    log.info(f"Loading model '{MODEL_NAME}' via vLLM "
             f"(gpu_memory_utilization={GPU_MEMORY_UTILIZATION}, "
             f"tensor_parallel_size={TENSOR_PARALLEL_SIZE}) ...")
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Each conversation is a one-turn chat: a system message pinning down
    # the "JSON only" behavior, plus the actual task as the user turn. The
    # system message is redundant with the instructions already inside
    # PROMPT_TEMPLATE, but costs nothing and adds a second, model-level
    # nudge away from the "let me explain my reasoning first" behavior
    # instruct models default to on open-ended tasks.
    conversations = [
        [
            {
                "role": "system",
                "content": (
                    "You respond with ONLY a single JSON object and nothing "
                    "else -- no reasoning, no explanation, no markdown "
                    "fences, no repeated/duplicate JSON objects."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    log.info(f"Running batched chat generation over {len(prompts)} prompts ...")
    outputs = llm.chat(conversations, sampling_params)

    # vLLM's .chat() preserves input order; each output has a list of
    # candidate completions under .outputs, we want the first (only, given
    # n=1 default in SamplingParams) completion's .text.
    return [output.outputs[0].text for output in outputs]


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

# Whether VALIDATION_FAILED items still get the LLM's candidate question
# applied (with needs_review forced True for human confirmation) rather than
# falling back to the pre-LLM deterministic text. See merge_results_into_
# records's docstring for the full rationale. Single source of truth so
# merge_results_into_records and the logging/diff functions below never
# disagree about what was actually applied.
APPLY_LLM_VERSION_ON_VALIDATION_FAILURE = True


def result_was_applied(result: dict) -> bool:
    """True if this result's fixed_question ends up as the qa's question."""
    status = result["status"]
    if status == "OK":
        return True
    if status == "VALIDATION_FAILED" and APPLY_LLM_VERSION_ON_VALIDATION_FAILURE and result.get("fixed_question"):
        return True
    return False


def result_final_question(result: dict) -> str:
    """The question text that actually lands in the record after merging."""
    return result["fixed_question"] if result_was_applied(result) else result["fq"].original_question


def process_flagged_questions(
    flagged: list[FlaggedQuestion],
) -> list[dict]:
    """
    Runs the full LLM-fix pass over a list of flagged questions and returns
    a list of result dicts (one per flagged question), each containing the
    original, the model's raw output, the parsed/validated fix (or None),
    and a human-readable status string.
    """
    prompts = [build_prompt(fq) for fq in flagged]
    raw_outputs = run_vllm_batch(prompts)

    results = []
    for fq, raw_output in zip(flagged, raw_outputs):
        parsed = parse_llm_response(raw_output)

        if parsed is None:
            results.append({
                "fq": fq,
                "raw_output": raw_output,
                "fixed_question": None,
                "status": "PARSE_FAILED",
                "detail": "Could not parse a JSON object with a 'fixed_question' key out of the model's output.",
            })
            continue

        candidate = parsed.get("fixed_question", "")
        is_valid, reason = validate_fixed_question(fq.original_question, candidate, fq.verb_form)

        if not is_valid:
            # Even though this failed the strict sanity checks, keep a
            # capitalized, ready-to-apply version of the LLM's candidate
            # around -- merge_results_into_records may still choose to use
            # it (see APPLY_LLM_VERSION_ON_VALIDATION_FAILURE below), and
            # review_server.py's "suggested fix" panel wants it either way.
            candidate_str = candidate.strip() if isinstance(candidate, str) else ""
            results.append({
                "fq": fq,
                "raw_output": raw_output,
                "fixed_question": capitalize_question(candidate_str) if candidate_str else None,
                "status": "VALIDATION_FAILED",
                "detail": reason,
            })
            continue

        results.append({
            "fq": fq,
            "raw_output": raw_output,
            # Force-capitalized regardless of what case the model returned --
            # see capitalize_question's docstring for why this is needed.
            "fixed_question": capitalize_question(candidate.strip()),
            "status": "OK",
            "detail": "Model reported changed={}".format(parsed.get("changed", "unspecified")),
        })

    return results


def merge_results_into_records(records: list[dict], results: list[dict]) -> tuple[list[dict], dict]:
    """
    Applies fixes back into the records structure (deep-copied -- never
    mutates the caller's original `records` list). Returns the updated
    records plus a summary dict of counts by status.

    Policy on which question text survives, per status:

      - OK: the LLM's fix passed every sanity check. Apply it, clear
        needs_review.

      - VALIDATION_FAILED: the LLM DID produce a candidate, but it tripped
        one of validate_fixed_question's mechanical guards (content
        injection, dropped words, suspicious length, etc). Those checks
        are deliberately strict/mechanical, not a semantic judge -- so a
        chunk of what lands here is still a genuinely better question than
        the pre-LLM deterministic version, just one the cheap heuristics
        couldn't rubber-stamp. We apply it anyway (APPLY_LLM_VERSION_ON_
        VALIDATION_FAILURE below), but -- unlike an OK -- we FORCE
        needs_review=True regardless of what it was before, so every one
        of these still surfaces in review_server.py for a human to
        confirm or revert. This is a deliberate trade-off: previously,
        a pair that started needs_review=False and failed validation
        silently kept the un-vetted deterministic text AND stayed
        invisible to review_server.py. Now it's the (usually-better) LLM
        text, but never silently invisible -- you always get a chance to
        catch a genuinely bad one.

      - PARSE_FAILED: there is no usable candidate at all (the model's
        output couldn't be parsed as JSON), so there's nothing to apply.
        Keeps the original deterministic question, and -- same as above --
        forces needs_review=True so it isn't silently invisible either.
    """
    import copy
    updated = copy.deepcopy(records)

    counts = {"OK": 0, "PARSE_FAILED": 0, "VALIDATION_FAILED": 0}
    n_validation_failed_applied = 0

    for result in results:
        fq = result["fq"]
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

        qa = updated[fq.record_idx]["rejected"][fq.qa_idx]
        prior_attempts = (fq.prior_attempt or {}).get("attempt_count", 0)
        qa["llm_grammar_fix"] = {
            "status": status,
            "detail": result["detail"],
            "original_question": fq.original_question,
            "raw_model_output": result["raw_output"],
            "attempt_count": prior_attempts + 1,
        }

        if status == "OK":
            qa["question"] = result["fixed_question"]
            qa["needs_review"] = False
            qa["llm_grammar_fix"]["applied_question"] = result["fixed_question"]

        elif status == "VALIDATION_FAILED" and APPLY_LLM_VERSION_ON_VALIDATION_FAILURE and result["fixed_question"]:
            qa["question"] = result["fixed_question"]
            # Force True regardless of the incoming value -- this pair
            # bypassed the sanity checks, so it must not be silently
            # invisible to review_server.py no matter how it started.
            qa["needs_review"] = True
            qa["llm_grammar_fix"]["applied_question"] = result["fixed_question"]
            qa["llm_grammar_fix"]["applied_despite_validation_failure"] = True
            n_validation_failed_applied += 1

        else:
            # PARSE_FAILED, or VALIDATION_FAILED with no usable candidate
            # (empty/non-string fixed_question) -- nothing to apply.
            # question stays as the original deterministic text. Force
            # needs_review=True here too, for the same "never silently
            # invisible" reason as above.
            qa["needs_review"] = True

    if n_validation_failed_applied:
        log.info(
            f"Applied the LLM's candidate question on {n_validation_failed_applied} "
            f"VALIDATION_FAILED item(s) despite failing sanity checks (all forced to "
            f"needs_review=True for human confirmation in review_server.py)."
        )

    return updated, counts


# ----------------------------------------------------------------------------
# Change logging
# ----------------------------------------------------------------------------

def print_change_summary(results: list[dict], verbose: bool = True) -> int:
    """
    Prints a before/after summary for every result. When verbose=True
    (sample mode), prints full context per result (sentence, verb_form,
    status/detail, raw output on failure). When verbose=False (full run),
    prints a terser one-line-per-change "original -> fixed" log, which is
    what you actually want to skim in a job's .out log after a full run
    over hundreds of examples.

    Returns the count of successfully applied fixes.
    """
    n_ok = 0
    for result in results:
        fq = result["fq"]
        applied = result_was_applied(result)
        if result["status"] == "OK":
            n_ok += 1
            if verbose:
                print("-" * 70)
                print(f"sentence:   {fq.sentence_text}")
                print(f"verb_form:  {fq.verb_form}")
                print(f"original:   {fq.original_question}")
                print(f"status:     {result['status']}  ({result['detail']})")
                print(f"fixed:      {result['fixed_question']}")
            else:
                print(f"[OK] sentence: \"{fq.sentence_text}\" (verb: {fq.verb_form})")
                print(f"     \"{fq.original_question}\" -> \"{result['fixed_question']}\"")
        elif applied:
            # VALIDATION_FAILED but applied anyway (see APPLY_LLM_VERSION_ON_
            # VALIDATION_FAILURE) -- distinct from a true no-op: the question
            # DID change, it just still needs a human to confirm it.
            if verbose:
                print("-" * 70)
                print(f"sentence:   {fq.sentence_text}")
                print(f"verb_form:  {fq.verb_form}")
                print(f"original:   {fq.original_question}")
                print(f"status:     {result['status']}  ({result['detail']})")
                print(f"applied (despite failed validation, needs human confirm): {result['fixed_question']}")
            else:
                print(f"[{result['status']} -> APPLIED, needs review] sentence: \"{fq.sentence_text}\" (verb: {fq.verb_form})")
                print(f"     \"{fq.original_question}\" -> \"{result['fixed_question']}\" ({result['detail']})")
        else:
            if verbose:
                print("-" * 70)
                print(f"sentence:   {fq.sentence_text}")
                print(f"verb_form:  {fq.verb_form}")
                print(f"original:   {fq.original_question}")
                print(f"status:     {result['status']}  ({result['detail']})")
                print(f"raw output: {result['raw_output']!r}")
            else:
                print(f"[{result['status']}] sentence: \"{fq.sentence_text}\" (verb: {fq.verb_form})")
                print(f"     \"{fq.original_question}\" -> suggested: \"{result.get('fixed_question') or result['raw_output']}\" "
                      f"-> UNCHANGED ({result['detail']})")
    return n_ok


def write_changes_log(results: list[dict], path: str) -> None:
    """
    Writes a dedicated, human-skimmable changes log -- one entry per flagged
    question, in plain "original -> fixed" form (or "-> UNCHANGED (reason)"
    for anything that didn't get applied) -- separate from the full output
    JSON. This is the file to open when you just want to audit what the LLM
    pass actually did, without paging through the whole dpo_pairs file.
    """
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_applied_despite_failure = sum(
        1 for r in results if r["status"] != "OK" and result_was_applied(r)
    )
    n_unchanged = len(results) - n_ok - n_applied_despite_failure
    with open(path, "w", encoding="utf-8") as f:
        f.write("LLM grammar-fix changes log\n")
        f.write(f"Total flagged questions processed: {len(results)}\n")
        f.write(f"Successfully fixed (validated OK): {n_ok}\n")
        f.write(f"Applied despite failed validation (needs human confirm): {n_applied_despite_failure}\n")
        f.write(f"Left fully unchanged (no usable candidate): {n_unchanged}\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- SUCCESSFUL FIXES (validated OK) ---\n\n")
        for result in results:
            if result["status"] == "OK":
                fq = result["fq"]
                f.write(f"record {fq.record_idx}, rejected[{fq.qa_idx}]\n")
                f.write(f"  sentence: {fq.sentence_text}\n")
                f.write(f"  verb:     {fq.verb_form}\n")
                f.write(f"  original: {fq.original_question}\n")
                f.write(f"  fixed:    {result['fixed_question']}\n\n")

        f.write("\n--- APPLIED DESPITE FAILED VALIDATION (flagged needs_review=True for human confirm) ---\n\n")
        for result in results:
            if result["status"] != "OK" and result_was_applied(result):
                fq = result["fq"]
                f.write(f"record {fq.record_idx}, rejected[{fq.qa_idx}]\n")
                f.write(f"  sentence: {fq.sentence_text}\n")
                f.write(f"  verb:     {fq.verb_form}\n")
                f.write(f"  original: {fq.original_question}\n")
                f.write(f"  applied:  {result['fixed_question']}\n")
                f.write(f"  why validation failed: {result['detail']}\n\n")

        f.write("\n--- FULLY UNCHANGED (no usable candidate -- still flagged for manual review) ---\n\n")
        for result in results:
            if result["status"] != "OK" and not result_was_applied(result):
                fq = result["fq"]
                f.write(f"record {fq.record_idx}, rejected[{fq.qa_idx}]\n")
                f.write(f"  original:  {fq.original_question}\n")
                f.write(f"  sentence:  {fq.sentence_text}\n")
                f.write(f"  verb:      {fq.verb_form}\n")
                f.write(f"  suggested: {result.get('fixed_question') or result['raw_output']!r}\n")
                f.write(f"  status:    {result['status']} -- {result['detail']}\n\n")

    log.info(
        f"Wrote changes log to {path} ({n_ok} validated fixes, "
        f"{n_applied_despite_failure} applied despite failed validation, "
        f"{n_unchanged} left fully unchanged)."
    )


def write_diff_file(results: list[dict], source_path: str, diff_path: str) -> None:
    """
    Writes a real unified diff (via difflib, the same format `diff -u`
    produces) comparing the SOURCE question text against this run's
    resulting question text, for every flagged question.

    Each "line" being diffed is tagged with a stable
    "record_idx:qa_idx | question text" identifier rather than just the raw
    question text. This matters because plain `difflib.unified_diff` aligns
    by line content -- if you used bare question text as the lines, two
    unrelated questions that happen to be identical strings would get
    silently treated as the same line, and a question whose fix didn't
    change anything would produce no diff entry at all (which is correct),
    but you'd have no way to tell WHICH flagged question is WHICH if you
    later diff two different runs' .diff files against each other. Tagging
    each line with its (record_idx, qa_idx) keeps every flagged question
    individually identifiable and directly comparable across runs.

    Because this writes a NEW timestamped file every run (see `main`, which
    builds `diff_path` from the current time) rather than overwriting a
    fixed name, you end up with one .diff file per run -- diff two of them
    directly with `diff run_a.diff run_b.diff` to see exactly where two
    different prompt/model versions disagreed.
    """
    source_lines = []
    result_lines = []
    for result in results:
        fq = result["fq"]
        tag = f"[{fq.record_idx}:{fq.qa_idx}]"
        source_lines.append(f"{tag} {fq.original_question}\n")
        # result_final_question already returns the original question for
        # anything that wasn't actually applied (PARSE_FAILED, or
        # VALIDATION_FAILED with no usable candidate), so those correctly
        # produce a no-op diff line -- and VALIDATION_FAILED items that WERE
        # applied anyway correctly show up as a real change here too.
        result_lines.append(f"{tag} {result_final_question(result)}\n")

    diff_lines = list(difflib.unified_diff(
        source_lines, result_lines,
        fromfile=source_path, tofile=f"{diff_path} (this run)",
        lineterm="\n",
    ))

    n_changed = sum(1 for r in results if result_was_applied(r))
    with open(diff_path, "w", encoding="utf-8") as f:
        f.write(f"# LLM grammar-fix diff -- generated {datetime.now().isoformat()}\n")
        f.write(f"# Source: {source_path}\n")
        f.write(f"# {n_changed}/{len(results)} flagged questions changed in this run.\n")
        f.write("#\n")
        f.write("# Each line is tagged [record_idx:qa_idx] so you can diff this file\n")
        f.write("# directly against another run's .diff to compare prompt/model versions.\n")
        f.write("#\n")
        if diff_lines:
            f.writelines(diff_lines)
        else:
            f.write("# (no changes -- every flagged question was left unchanged this run)\n")

    log.info(f"Wrote diff file to {diff_path} ({n_changed}/{len(results)} lines changed).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dpo_pairs.json", help="Path to build_dpo_data.py's output JSON.")
    parser.add_argument("--output", default=None,
                         help="Path to write the updated JSON. Defaults to '<input>.grammar_fixed.json'. "
                              "Never overwrites --input.")
    parser.add_argument("--sample", type=int, default=None,
                         help="If set, only process a random sample of N flagged questions and print "
                              "before/after pairs to stdout (no file written unless --write-sample-output).")
    parser.add_argument("--write-sample-output", action="store_true",
                         help="When used with --sample, also write the (partial) output file.")
    parser.add_argument("--changes-log", default=None,
                         help="Path to write a plain-text 'original -> fixed' changes log "
                              "(full run only). Defaults to '<output>.changes.txt'.")
    parser.add_argument("--diff-dir", default=".",
                         help="Directory to write this run's timestamped diff file into "
                              "(full run only). Defaults to the current directory. Each run "
                              "writes a new 'changes_<timestamp>.diff' file here -- nothing "
                              "is ever overwritten, so you can diff two runs against each other.")
    parser.add_argument("--mode", choices=["fresh", "retry"], default="fresh",
                         help="'fresh' (default): process every flagged item, ignoring any "
                              "llm_grammar_fix info already in --input. Use this on the original "
                              "dpo_pairs.json, or any time you just want a clean full pass. "
                              "'retry': only re-attempt items that already have a non-OK "
                              "llm_grammar_fix result in --input (i.e. previously failed) -- "
                              "pass a prior run's --output back in as --input to retry just the "
                              "failures, e.g. after tweaking the prompt. Items already fixed "
                              "(status OK) or never attempted are left untouched. The previous "
                              "failure reason is shown to the model so it can try something "
                              "different.")
    args = parser.parse_args()

    if args.output is None:
        base = args.input
        if base.endswith(".grammar_fixed.json"):
            base = base[: -len(".grammar_fixed.json")]
        elif base.endswith(".json"):
            base = base[:-5]
        args.output = base + ".grammar_fixed.json"
        if args.output == args.input:
            # Retrying directly on a previous run's output: don't silently
            # collide with --input. Add a suffix instead of overwriting.
            args.output = base + ".grammar_fixed.retry.json"

    if args.output == args.input:
        raise SystemExit("Refusing to run: --output would overwrite --input. Pick a different output path.")

    log.info(f"Loading records from {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    log.info(f"Loaded {len(records)} records.")

    flagged = load_flagged_questions(records, mode=args.mode)
    log.info(f"Mode: {args.mode}. Found {len(flagged)} flagged question(s) to process.")

    if not flagged:
        log.info("Nothing to do -- no flagged questions found. Exiting without writing any file.")
        return

    rng = random.Random(RANDOM_SEED)
    if args.sample is not None:
        sample_size = min(args.sample, len(flagged))
        flagged_to_process = rng.sample(flagged, sample_size)
        log.info(f"--sample {args.sample} given: processing a random subset of {sample_size}.")
    else:
        flagged_to_process = flagged

    results = process_flagged_questions(flagged_to_process)

    if args.sample is not None and not args.write_sample_output:
        log.info("Sample mode: printing before/after pairs, NOT writing any file "
                  "(pass --write-sample-output to also write a partial file).")
        n_ok = print_change_summary(results, verbose=True)
        print("-" * 70)
        print(f"\n{n_ok}/{len(results)} fixes accepted. Review the before/after pairs above, "
              f"then re-run without --sample (or with a larger --sample) once you're happy "
              f"with the quality.")
        return

    updated_records, counts = merge_results_into_records(records, results)

    log.info(f"Result counts: {counts}")

    # Terse one-line-per-change log straight into this job's .out log, so the
    # changes are visible without opening any extra file.
    log.info("Changes made by the LLM grammar-fix pass:")
    print_change_summary(results, verbose=False)

    # Dedicated changes-log file, separate from the full output JSON, for
    # easy skimming/diffing later.
    if args.changes_log is None:
        if args.output.endswith(".json"):
            args.changes_log = args.output[:-5] + ".changes.txt"
        else:
            args.changes_log = args.output + ".changes.txt"
    write_changes_log(results, args.changes_log)

    # Timestamped diff file -- one per run, never overwritten, so you can
    # compare different runs (different prompt versions, different models,
    # etc) by diffing two of these .diff files directly against each other.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_path = f"{args.diff_dir.rstrip('/')}/changes_{timestamp}.diff"
    write_diff_file(results, args.input, diff_path)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote {args.output}.")
    log.info(
        f"To compare this run against a previous one: "
        f"diff {args.diff_dir.rstrip('/')}/changes_<earlier_timestamp>.diff {diff_path}"
    )



if __name__ == "__main__":
    main()