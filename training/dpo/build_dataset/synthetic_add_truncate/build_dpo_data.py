# ==========================================================================
# Builds the SUPERSEDED synthetic add/truncate preference pairs.
# Downloads the QA-SRL dev gold, then corrupts each group's gold completion into a
# `rejected` variant: `truncate` drops a QA pair, `add` injects a wrong-role QA pair
# borrowed from another predicate. Forced to an exact 50/50 split. Writes dpo_pairs.json.
#
# Part of the SUPERSEDED add/truncate DPO arm -- kept for the record only.
# The reported DPO result (79.59) came from the on-policy pairs in ../ .
# ==========================================================================
"""
build_dpo_data.py

Constructs (accepted, rejected) pairs for QASRL DPO training from the dev set.

============================================================================
SCHEMA -- confirmed against a real row from dev.json (not guessed)
============================================================================
Each raw row looks like:

    {
        "id": "Wiki1k:wikinews:1002218:0:0_KMQTT",
        "sentence_id": "Wiki1k:wikinews:1002218:0:0",
        "predicate": "posted",            # inflected surface form
        "verb_form": "post",              # bare lemma
        "is_verbal": true,
        "question_slots": ["what","did","someone","post","_","_","_","?"],
        "detokenized": {
            "sentence": "On Friday, Clark posted to Facebook ...",
            "question": "what did someone post?",
            "contextualized_question": "What did Clark post?",
            "answers": {
                "text": ["..."],
                "answer_start_token": [11],
                "answer_start_char": [68]
            },
            "predicate_idx_token": 3,
            "predicate_idx_char": 17,
            "predicate_idx_char_end": 23
        },
        "tokenized": { ... same shape, token-based sentence + char offsets
                        that DO NOT line up with detokenized offsets ... }
    }

`question_slots` is POSITIONAL, 8 slots: [WH, AUX, SUBJ, VERB, OBJ, PREP,
OBJ2, "?"], with "_" marking an empty/unused slot. Slot index 3 holds the
bare verb lemma (matches `verb_form`, not `predicate`).

We standardize on `detokenized` throughout (matches the original
tmp_inspect_dev.py script and is what a human reviewer wants to read).
`tokenized` offsets are NOT mixed in anywhere, since they disagree with
detokenized offsets (punctuation spacing differs between the two).

Answer end-offsets aren't stored explicitly -- we derive them as
`answer_start_char + len(text)`, which is exact since each answer is a
contiguous span copied verbatim out of the sentence string.

============================================================================
PIPELINE OVERVIEW
============================================================================
1. Download dev.json, group raw QA rows into (sentence_id, predicate) groups.
2. For each group, the "accepted" response is the full gold QA list.
3. Decide, for every group, ONE process -- "add" or "truncate" (never both):

   - Every group with EXACTLY 1 QA pair is forced to "add" (truncate is
     either impossible or destructive on a single pair).
   - Every other group (>1 pair) is split between "add" and "truncate" so
     that, ACROSS THE WHOLE DATASET (including the forced-add singles
     above), the OVERALL split lands at 50% add / 50% truncate. This split
     is computed exactly via shuffle-and-slice over the eligible (>1-pair)
     groups, not via independent per-group coin flips (see
     `assign_processes` for the exact formula and rationale).

   TRUNCATE process:
       - Pick ONE QA pair from the group at random.
       - If it has multiple answers: flip a coin between removing the
         whole pair vs. removing one of its answers.
       - If it has only one answer: remove the whole pair (removing the
         answer would leave a zero-answer question, which isn't valid).

   ADD process:
       Sample k in {1, 2, 3} (uniform) foreign QA pairs to inject.
       For each:
         - Prefer sampling the foreign QA from a DIFFERENT predicate in the
           SAME sentence. If the sentence has no other predicate, fall back
           to sampling globally (any sentence, any other predicate).
         - Adapt the question to target the CURRENT predicate by editing the
           VERB slot (index 3) of `question_slots` to the current group's
           `verb_form`, then re-rendering the question string from the
           edited slots. This is deterministic and doesn't rely on string
           matching against the rendered question text.
         - The answer text/span is taken VERBATIM from the foreign source --
           never modified. See rationale below.
         - If the (adapted) question string already matches an existing
           question in this group's gold set, the foreign answer is appended
           to that question's answer list instead of creating a new pair.
         - `maybe_llm_fix_question_grammar` is called on every adapted
           question right after the deterministic slot-swap, but stays a
           no-op passthrough here on purpose -- the actual LLM call runs
           afterwards, as its own step, in the separate `llm_fix_grammar.py`
           script (needs a GPU; this script doesn't). That script scans this
           script's output JSON for every `is_foreign=True` pair (i.e. every
           Add-step injection, not only the ones flagged `needs_review`) and
           asks an LLM to clean up tense/agreement issues, writing the
           result to a new file. See `maybe_llm_fix_question_grammar`'s
           docstring below for the exact two-step workflow.

4. Every constructed example carries full provenance metadata so you can
   audit, filter, or re-roll any example without re-running the whole script.

============================================================================
WHY THE ANSWER SPAN IS NEVER MODIFIED (only the question is adapted)
============================================================================
The evaluation metric in this project is unlabeled F1 by answer span only
(two QA pairs are "the same" if their answers occupy the same char range,
regardless of question wording). Given that:

  - If we hand-picked a "plausible" new span for the borrowed question, we'd
    be authoring a *correct* answer -- that's not a rejection, that's just
    more gold data.
  - If we left the question unadapted (still referencing the donor
    predicate's verb), the model could learn to flag rejections by surface
    verb mismatch alone, never learning the actual argument-attachment
    skill.
  - Adapting the question's verb slot but keeping the donor's original span
    produces a pair that is grammatically well-formed for the CURRENT
    predicate, but whose answer is genuinely wrong (it answers a different
    predicate's role). This is the realistic failure mode (argument
    misattribution), and because the span belongs to a different
    predicate's argument, it will almost never coincide with any gold span
    for the current predicate -- so it reliably scores as wrong under the
    project's own F1 metric.

============================================================================
RENDERING A QUESTION STRING FROM question_slots
============================================================================
`render_question_from_slots` joins the 8 slots with single spaces, skipping
"_" placeholders, and re-attaches the trailing "?" directly (no space before
it) -- matching the format of `detokenized.question` (e.g.
"what did someone post?"). This is a straightforward, fully deterministic
operation since every dev.json row carries `question_slots` -- there is no
"slots missing" fallback path needed for the adaptation step itself.
"""

import json
import logging
import os
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

DEV_URL = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/dev.json"
OUTPUT_PATH = os.environ.get("QASRL_DPO_PAIRS_OUT",
                             str(pathlib.Path(__file__).resolve().parent / "dpo_pairs.json"))
RANDOM_SEED = 42  # set to None for non-reproducible runs

VERB_SLOT_INDEX = 3  # position of the verb lemma within question_slots
AUX_SLOT_INDEX = 1   # position of the auxiliary within question_slots

ADD_COUNT_CHOICES = [1, 2, 3]          # uniform unless you want weights


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------

@dataclass
class QAPair:
    """One question with its (possibly multiple) gold answers."""
    question: str
    answers: list[str] = field(default_factory=list)
    answer_start_chars: list[Optional[int]] = field(default_factory=list)
    answer_end_chars: list[Optional[int]] = field(default_factory=list)
    question_slots: list[str] = field(default_factory=list)  # positional 8-slot template
    verb_form: str = ""           # bare lemma for this pair's ORIGINAL predicate
    source_sentence_id: Optional[str] = None  # set when this pair was borrowed from elsewhere
    source_predicate: Optional[str] = None
    source_question: Optional[str] = None     # the donor's ORIGINAL question, before verb-slot adaptation
    source_sentence_text: Optional[str] = None  # the donor's full sentence (answers are spans into THIS text, not the current group's sentence)
    is_foreign: bool = False        # True if this pair was injected by the Add step
    adapted_automatically: bool = False
    needs_review: bool = False
    review_note: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Group:
    """All gold QA pairs for one (sentence_id, predicate)."""
    sentence_id: str
    predicate: str
    verb_form: str
    sentence_text: str
    qa_pairs: list[QAPair]


# ----------------------------------------------------------------------------
# Step 1: download + group
# ----------------------------------------------------------------------------

def download_dev_set(url: str = DEV_URL) -> list[dict]:
    log.info(f"Downloading dev set from {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    examples = list(data.values()) if isinstance(data, dict) else list(data)
    log.info(f"Downloaded {len(examples)} raw QA rows.")
    return examples


def build_groups(raw_examples: list[dict]) -> dict[tuple[str, str], Group]:
    """
    Groups raw QA rows into (sentence_id, predicate) -> Group.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sentence_text_by_id: dict[str, str] = {}
    verb_form_by_key: dict[tuple[str, str], str] = {}

    for ex in raw_examples:
        s_id = ex.get("sentence_id")
        pred = ex.get("predicate")
        if not s_id or not pred:
            continue
        det = ex.get("detokenized", {}) or {}
        grouped[(s_id, pred)].append(ex)
        sent_text = det.get("sentence")
        if sent_text and s_id not in sentence_text_by_id:
            sentence_text_by_id[s_id] = sent_text
        if (s_id, pred) not in verb_form_by_key and ex.get("verb_form"):
            verb_form_by_key[(s_id, pred)] = ex["verb_form"]

    groups: dict[tuple[str, str], Group] = {}
    for (s_id, pred), rows in grouped.items():
        qa_pairs = []
        for row in rows:
            det = row.get("detokenized", {}) or {}
            question = det.get("question")
            if question is None:
                continue
            answers_field = det.get("answers", {}) or {}
            answer_texts = list(answers_field.get("text", []) or [])
            start_chars = list(answers_field.get("answer_start_char", []) or [])
            while len(start_chars) < len(answer_texts):
                start_chars.append(None)
            end_chars = [
                (sc + len(txt)) if sc is not None else None
                for sc, txt in zip(start_chars, answer_texts)
            ]

            qa_pairs.append(QAPair(
                question=capitalize_question(question),
                answers=answer_texts,
                answer_start_chars=start_chars,
                answer_end_chars=end_chars,
                question_slots=list(row.get("question_slots", []) or []),
                verb_form=row.get("verb_form", ""),
            ))

        groups[(s_id, pred)] = Group(
            sentence_id=s_id,
            predicate=pred,
            verb_form=verb_form_by_key.get((s_id, pred), ""),
            sentence_text=sentence_text_by_id.get(s_id, ""),
            qa_pairs=qa_pairs,
        )

    log.info(f"Built {len(groups)} (sentence, predicate) groups.")
    has_slots = sum(1 for g in groups.values() for qa in g.qa_pairs if qa.question_slots)
    total_qa = sum(len(g.qa_pairs) for g in groups.values())
    log.info(f"QA pairs carrying question_slots: {has_slots} / {total_qa}.")
    return groups


# ----------------------------------------------------------------------------
# Step 2: question adaptation (deterministic verb-slot swap)
# ----------------------------------------------------------------------------

def capitalize_question(question: str) -> str:
    """
    Capitalizes the first alphabetic character of a question string,
    leaving everything else untouched -- e.g. "why did someone leave?" ->
    "Why did someone leave?". Safe on empty strings and strings that don't
    start with a letter (shouldn't happen for WH-questions, but no reason
    to crash if it does).
    """
    for i, ch in enumerate(question):
        if ch.isalpha():
            return question[:i] + ch.upper() + question[i + 1:]
    return question


def render_question_from_slots(slots: list[str]) -> str:
    """
    Renders a question string from the positional 8-slot template:
    [WH, AUX, SUBJ, VERB, OBJ, PREP, OBJ2, "?"]. "_" placeholders are
    skipped; the trailing "?" is attached without a preceding space.
    The WH-word is capitalized to match standard question casing (dev.json's
    raw questions come in lowercase, e.g. "what did someone post?").
    """
    words = [s for s in slots[:-1] if s and s != "_"]
    q_mark = slots[-1] if slots else "?"
    return capitalize_question(" ".join(words) + q_mark)


def adapt_question_to_predicate(qa: QAPair, target_verb_form: str) -> tuple[str, list[str], bool, str]:
    """
    Rewrites qa's question to target a different predicate by swapping the
    verb slot (index VERB_SLOT_INDEX) to target_verb_form and re-rendering.

    Returns (new_question, new_slots, adapted_automatically, note).
    """
    if not qa.question_slots or len(qa.question_slots) <= VERB_SLOT_INDEX:
        return qa.question, list(qa.question_slots), False, (
            "question_slots missing or malformed -- needs manual adaptation."
        )
    if not target_verb_form:
        return qa.question, list(qa.question_slots), False, (
            "Target predicate has no verb_form on record -- needs manual adaptation."
        )

    new_slots = list(qa.question_slots)
    original_verb_slot = new_slots[VERB_SLOT_INDEX]
    new_slots[VERB_SLOT_INDEX] = target_verb_form
    new_question = render_question_from_slots(new_slots)

    aux_slot = qa.question_slots[AUX_SLOT_INDEX] if len(qa.question_slots) > AUX_SLOT_INDEX else "_"
    if aux_slot == "_" or not aux_slot:
        # No auxiliary carries the tense in this question -- the original verb
        # slot ("posted") was doing double duty as both lemma and conjugation.
        # Swapping in a bare lemma here often produces an unconjugated,
        # ungrammatical result (e.g. "who posted something?" -> "who explain
        # something?"). We still perform the swap (it's still useful as a
        # rejected example -- a malformed question is itself a a valid
        # negative), but mark it for manual review rather than claiming
        # full confidence.
        note = (
            f"Auto-swapped verb slot '{original_verb_slot}' -> '{target_verb_form}', but "
            f"the donor question has no auxiliary (AUX slot is empty), meaning the "
            f"original verb was carrying its own tense/conjugation. The bare lemma "
            f"'{target_verb_form}' was substituted as-is and is LIKELY UNCONJUGATED "
            f"(e.g. 'who posted something?' -> 'who explain something?'). Please fix "
            f"the conjugation by hand."
        )
        return new_question, new_slots, False, note

    note = (
        f"Auto-swapped verb slot '{original_verb_slot}' -> '{target_verb_form}' "
        f"and re-rendered the question. Tense/auxiliary agreement (the AUX slot) "
        f"was NOT changed and may now read oddly -- please sanity check."
    )
    return new_question, new_slots, True, note


# ----------------------------------------------------------------------------
# Step 3: Truncate process
# ----------------------------------------------------------------------------

def apply_truncate(group: Group, rng: random.Random) -> tuple[list[QAPair], list[str]]:
    """
    New, simplified truncate process (per user's latest spec):

      1. Choose ONE QA pair from the group at random.
      2. If that pair has multiple answers: flip a coin between
           (a) removing the whole pair, or
           (b) removing one of its answers.
      3. If that pair has only one answer: no choice to make -- removing
         the answer would leave a question with zero answers, which isn't
         valid, so we remove the whole pair.

    This function is never called on a group with exactly 1 QA pair in the
    new pipeline (those groups are always routed to Add instead -- see
    `assign_processes` / `main`), so we don't need a single-pair fallback
    here anymore. The assertion below documents that assumption loudly
    rather than silently doing something unexpected if it's ever violated.
    """
    assert len(group.qa_pairs) > 1, (
        f"apply_truncate called on a group with {len(group.qa_pairs)} QA pair(s) "
        f"({group.sentence_id}, {group.predicate}) -- truncate should never be "
        f"selected for single-pair groups under the current pipeline rules."
    )

    pairs = [QAPair(**asdict(qa)) for qa in group.qa_pairs]
    notes = []

    target_idx = rng.randrange(len(pairs))
    target = pairs[target_idx]
    notes.append(f"Truncate: selected QA pair at random: '{target.question}' (answers: {target.answers}).")

    if len(target.answers) > 1:
        action = rng.choice(["remove_pair", "remove_answer"])
    else:
        action = "remove_pair"
        notes.append("Selected pair has only one answer -- 'remove answer' is not a valid "
                     "option here (would leave a question with zero answers), so the whole "
                     "pair is removed instead.")

    if action == "remove_pair":
        removed_question = pairs[target_idx].question
        pairs = [qa for i, qa in enumerate(pairs) if i != target_idx]
        notes.append(f"Action: removed the whole QA pair ('{removed_question}').")
    else:
        removed_pos = rng.randrange(len(target.answers))
        removed_answer = target.answers.pop(removed_pos)
        if len(target.answer_start_chars) > removed_pos:
            target.answer_start_chars.pop(removed_pos)
        if len(target.answer_end_chars) > removed_pos:
            target.answer_end_chars.pop(removed_pos)
        notes.append(f"Action: removed one answer ('{removed_answer}') from '{target.question}', "
                     f"leaving {len(target.answers)} answer(s) on that question.")

    return pairs, notes


# ----------------------------------------------------------------------------
# Step 4: Add process
# ----------------------------------------------------------------------------

def pick_foreign_qa(
    group: Group,
    all_groups: dict[tuple[str, str], Group],
    same_sentence_other_predicates: list[tuple[str, str]],
    rng: random.Random,
) -> tuple[QAPair, str, str, str, str]:
    """
    Picks one QA pair from a different predicate, preferring the same
    sentence, falling back to a global random choice.

    Returns (qa_pair_copy, source_sentence_id, source_predicate, source_note,
             source_sentence_text).

    IMPORTANT: the global fallback excludes every group whose verb_form
    matches the CURRENT group's verb_form -- not just the exact
    (sentence_id, predicate) pair. The same lemma ("post"/"posted") recurs
    across many different sentences in the corpus, and if the fallback only
    excluded the current sentence, it could land on a different sentence
    that happens to share the same verb -- producing a no-op "swap" (e.g.
    'post' -> 'post') that isn't a real negative at all, just a relabeled
    copy of a same-meaning question. Filtering on verb_form (not predicate
    string) also correctly excludes inflectional variants of the same verb.
    """
    if same_sentence_other_predicates:
        src_key = rng.choice(same_sentence_other_predicates)
        source_note = "same-sentence"
    else:
        all_keys = [
            k for k, g in all_groups.items()
            if g.qa_pairs and g.verb_form != group.verb_form
        ]
        if not all_keys:
            # Degenerate corpus-wide case: every other group happens to share
            # this verb_form. Fall back to excluding only the exact group so
            # the Add step can still produce *something*, but this should be
            # rare to nonexistent on a real corpus with >1 distinct verb.
            all_keys = [k for k, g in all_groups.items()
                        if k != (group.sentence_id, group.predicate) and g.qa_pairs]
        src_key = rng.choice(all_keys)
        source_note = "global-fallback (sentence had no other predicate)"

    src_group = all_groups[src_key]
    donor_qa = rng.choice(src_group.qa_pairs)
    donor_copy = QAPair(**asdict(donor_qa))
    return donor_copy, src_key[0], src_key[1], source_note, src_group.sentence_text


def maybe_llm_fix_question_grammar(
    question: str,
    needs_review: bool,
    review_note: str,
    group: Group,
    donor_sentence_text: str,
) -> tuple[str, bool, str]:
    """
    Intentionally STILL a no-op here. This is not where the LLM call
    happens -- it happens in the separate `llm_fix_grammar.py` script, run
    as its own step AFTER this one. That split is deliberate (see the
    module docstring's PIPELINE OVERVIEW and `llm_fix_grammar.py`'s own
    module docstring for the full rationale): this script needs to keep
    running on a plain CPU node with no GPU/vLLM dependency, so it can't
    make the LLM call itself.

    How the two scripts actually connect, end to end:
      1. This script (`apply_add`, right below) tags every Add-step pair
         with `is_foreign=True`, and additionally flags `needs_review=True`
         on the specific cases its deterministic verb-slot swap suspects
         are ungrammatical (see `adapt_question_to_predicate`). Both fields
         are written straight into the output JSON -- no separate flagging
         step needed, it's already part of every record.
      2. Run `llm_fix_grammar.py --input dpo_pairs.json` (via
         `sbatch run_grammar_fix.sbatch` on this cluster) as a second job.
         It scans that JSON for every `is_foreign=True` pair -- ALL of
         them, not just the `needs_review` subset -- and asks an LLM to
         clean up the grammar (or leave it alone if it's already fine).
      3. It writes a NEW file, `dpo_pairs.grammar_fixed.json`, with fixed
         questions merged back in and `needs_review` cleared wherever a fix
         was successfully applied. `dpo_pairs.json` itself is never
         touched.

    So there's nothing to wire up inside this function -- it stays a no-op,
    and `apply_add` already calls it on every foreign pair right after the
    deterministic adaptation, which is exactly what tags each pair with the
    `is_foreign`/`needs_review` metadata `llm_fix_grammar.py` reads.
    """
    return question, needs_review, review_note


def answer_is_accidentally_correct(donor_answers: list[str], group: "Group") -> bool:
    """
    True if any of the donor's answer texts already appears as a gold answer
    SOMEWHERE in this group's TRUE (accepted) QA pairs -- i.e. under a
    different question, but for the same predicate.

    This matters because QASRL answers routinely repeat across questions for
    the same predicate (and even across predicates in a sentence) -- e.g.
    "who posted?" -> "Clark" and "who commented?" -> "Clark" can both be
    genuinely true. If a donor's answer text happens to coincide with one of
    THIS predicate's real gold answers, injecting it as a "wrong" answer
    would mislabel a correct answer as a rejection -- exactly the case we
    need to filter out before adding a foreign pair.

    Only compared against group.qa_pairs (the true/accepted set) -- NOT
    against previously-injected foreign pairs from earlier in the same Add
    loop, since those aren't gold and shouldn't count as "already correct".
    """
    true_answers = {a.strip() for qa in group.qa_pairs for a in qa.answers}
    return any(a.strip() in true_answers for a in donor_answers)


# Prepositions QASRL gold keeps INSIDE the argument span. Measured on
# evaluation/data/gold/gold_updated_passive_filled_slots.csv: 1,343 of 8,719
# gold spans (15.4%) begin with one of these ("In 1917", "with his family",
# "to his home"), so a stored span that is exactly PREPOSITION + answer is
# following the gold convention, not a mis-set offset.
_GOLD_LEADING_PREPS = (
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "into", "onto",
    "over", "under", "about", "after", "before", "during", "through",
    "between", "against", "across", "around", "among", "as", "as of",
)


def _is_gold_style_prep_span(ref: str, s, e, ans: str) -> bool:
    """True if [s,e) is exactly `<leading preposition> + ans` -- keep as-is."""
    if not (isinstance(s, int) and isinstance(e, int)):
        return False
    span = ref[s:e]
    if not span.endswith(ans) or span == ans:
        return False
    prefix = span[: len(span) - len(ans)].strip().lower()
    return prefix in _GOLD_LEADING_PREPS


def normalize_answer_offsets(records: list[dict]) -> int:
    """
    Re-align answer_start_chars / answer_end_chars so each offset pair actually
    slices to its own answer text, and return how many were corrected.

    REFERENCE SENTENCE. An answer's offsets index the sentence the answer was
    taken FROM, which is not always this record's sentence:
      * is_foreign=True (Add-step, borrowed from another predicate) -> the
        offsets index source_sentence_text, the donor's sentence.
      * everything else -> they index this record's sentence_text.
    Re-aligning a borrowed span against the TARGET sentence would silently
    relocate a deliberately wrong-role argument into the local sentence and
    destroy the corruption the Add step exists to create, so the donor
    sentence is always used for foreign pairs.

    RULE (conservative). An offset pair is rewritten only when the current one
    does not slice to the answer AND the answer text occurs EXACTLY ONCE in
    the reference sentence, which makes the correction unambiguous. If the
    text is absent, ambiguous, or the reference sentence is unknown, the
    existing offsets are left untouched -- a wrong offset is preferable to a
    guessed one. Spans that already slice correctly are never touched, so a
    gold-style span carrying a leading preposition (QASRL gold keeps them:
    "in 1917", "from the fans") survives as-is.

    These offsets are PROVENANCE ONLY. Nothing downstream reads them --
    build_dpo_training_data.render_completion() renders question + answer TEXT
    -- so this affects no training signal and no reported metric.
    """
    n_fixed = 0
    for rec in records:
        for side in ("accepted", "rejected"):
            for qa in rec.get(side, []):
                ref = (qa.get("source_sentence_text") if qa.get("is_foreign")
                       else rec.get("sentence_text"))
                answers = qa.get("answers") or []
                starts = qa.get("answer_start_chars") or []
                ends = qa.get("answer_end_chars") or []
                # keep the three lists index-aligned; short lists caused the
                # original drift (two answers sharing one offset pair)
                starts += [None] * (len(answers) - len(starts))
                ends += [None] * (len(answers) - len(ends))
                del starts[len(answers):]
                del ends[len(answers):]
                for i, ans in enumerate(answers):
                    if not ref or not ans:
                        continue
                    s, e = starts[i], ends[i]
                    if isinstance(s, int) and isinstance(e, int) and ref[s:e] == ans:
                        continue                      # already correct
                    if _is_gold_style_prep_span(ref, s, e, ans):
                        continue                      # gold convention -> keep
                    if ref.count(ans) != 1:
                        continue                      # absent or ambiguous -> keep
                    j = ref.find(ans)
                    starts[i], ends[i] = j, j + len(ans)
                    n_fixed += 1
                qa["answer_start_chars"] = starts
                qa["answer_end_chars"] = ends
    return n_fixed


def dedupe_new_answers(
    existing_answers: list[str],
    new_answers: list[str],
    new_start_chars: list[Optional[int]],
    new_end_chars: list[Optional[int]],
) -> tuple[list[str], list[Optional[int]], list[Optional[int]]]:
    """
    Filters (new_answers, new_start_chars, new_end_chars) down to only the
    answers that aren't already present (compared strip()'d, case-sensitive
    otherwise -- QASRL answers are verbatim spans, so exact text match is
    the right notion of "same answer") in existing_answers. Also de-dupes
    WITHIN new_answers itself, so two answers being added in the same call
    can't be literal copies of each other either.

    This is what guarantees a single question never ends up with the same
    answer text listed twice -- whether that's because the exact same donor
    QA pair got sampled twice in one Add pass, or because two different
    donors just happen to share answer text.
    """
    seen = {a.strip() for a in existing_answers}
    kept_answers, kept_starts, kept_ends = [], [], []
    for ans, start, end in zip(new_answers, new_start_chars, new_end_chars):
        key = ans.strip()
        if key in seen:
            continue
        seen.add(key)
        kept_answers.append(ans)
        kept_starts.append(start)
        kept_ends.append(end)
    return kept_answers, kept_starts, kept_ends


def apply_add(
    group: Group,
    base_pairs: list[QAPair],
    all_groups: dict[tuple[str, str], Group],
    sentence_predicate_index: dict[str, list[str]],
    rng: random.Random,
) -> tuple[list[QAPair], list[str]]:
    """
    Injects k in {1,2,3} foreign QA pairs into base_pairs, adapting each
    question's verb slot to the current predicate's verb_form.

    Two safeguards keep injected answers from ever being literal duplicates
    of one another (see `dedupe_new_answers`):
      1. Donor tracking: within one call, the same (sentence, predicate,
         question) donor is never sampled twice -- re-rolls if it is.
      2. Belt-and-suspenders answer dedup: even if two DIFFERENT donors
         happen to share answer text, that text is only added to a given
         question once.
    """
    pairs = [QAPair(**asdict(qa)) for qa in base_pairs]
    notes = []

    # Filter by verb_form, not predicate string -- two different predicate
    # annotations in the same sentence could (rarely) share a verb_form (e.g.
    # a repeated/coordinated verb), and a same-verb donor would produce a
    # no-op "swap" that isn't a real negative. We don't have direct access to
    # other groups' verb_forms here without a lookup, so build one.
    other_predicates_here = [
        p for p in sentence_predicate_index.get(group.sentence_id, [])
        if p != group.predicate and all_groups.get((group.sentence_id, p)) is not None
        and all_groups[(group.sentence_id, p)].verb_form != group.verb_form
    ]
    same_sentence_keys = [(group.sentence_id, p) for p in other_predicates_here]

    k = rng.choice(ADD_COUNT_CHOICES)
    notes.append(f"Add sub-choice: injecting k={k} foreign QA pair(s).")

    MAX_DONOR_RETRIES = 10  # guards against pathological corpora; see note below

    # Tracks (source_sentence_id, source_predicate, donor_question) for every
    # donor actually injected so far in THIS call. Without this, k>1 could
    # sample the exact same donor QA pair twice; the second draw would then
    # collide on the same adapted question and just re-append its own answer
    # onto itself -- a literal duplicate. Re-rolling on a repeat donor stops
    # that at the source, rather than relying only on the answer-level dedup
    # below to clean it up after the fact.
    used_donor_signatures: set[tuple[str, str, str]] = set()

    for i in range(k):
        # Pick a donor, re-rolling (up to MAX_DONOR_RETRIES times) whenever
        # its answer text turns out to already be a TRUE gold answer for
        # this predicate -- see answer_is_accidentally_correct's docstring --
        # or it's a donor we've already injected earlier in this same loop.
        # The forced_no_swap degenerate case (every other group shares this
        # verb_form) can't be re-rolled since there's no alternative donor --
        # but we still check it below, once, and skip the injection outright
        # rather than let a possibly-true answer through unflagged.
        accidental_correct_retries = 0
        donor_signature = None
        while True:
            donor_qa, src_sent, src_pred, source_note, src_sentence_text = pick_foreign_qa(
                group, all_groups, same_sentence_keys, rng
            )

            # Defensive guard: pick_foreign_qa should never return a same-verb_form
            # donor, but if it somehow does (e.g. the degenerate corpus-wide
            # fallback in pick_foreign_qa), don't silently inject a no-op swap.
            # Re-roll once globally excluding this donor; if that's not possible
            # either, flag the resulting pair loudly rather than passing it off
            # as a real adaptation.
            donor_verb_form = donor_qa.verb_form
            forced_no_swap = False
            if donor_verb_form == group.verb_form:
                other_keys = [
                    k2 for k2, g2 in all_groups.items()
                    if g2.qa_pairs and g2.verb_form != group.verb_form
                ]
                if other_keys:
                    retry_key = rng.choice(other_keys)
                    retry_group = all_groups[retry_key]
                    donor_qa = QAPair(**asdict(rng.choice(retry_group.qa_pairs)))
                    src_sent, src_pred = retry_key[0], retry_key[1]
                    src_sentence_text = retry_group.sentence_text
                    source_note = "global-fallback (re-rolled to avoid same-verb donor)"
                else:
                    forced_no_swap = True

            donor_signature = (src_sent, src_pred, donor_qa.question)

            if forced_no_swap:
                if answer_is_accidentally_correct(donor_qa.answers, group):
                    notes.append(
                        f"  [{i+1}/{k}] Skipped injection: forced_no_swap donor's answer "
                        f"({donor_qa.answers}) is ALSO already a TRUE gold answer for this "
                        f"predicate, and there was no alternative donor to re-roll to -- "
                        f"refusing to inject a pair that could just be a correct answer "
                        f"mislabeled as a rejection."
                    )
                    donor_qa = None
                break

            is_repeat_donor = donor_signature in used_donor_signatures
            if not is_repeat_donor and not answer_is_accidentally_correct(donor_qa.answers, group):
                break

            accidental_correct_retries += 1
            if is_repeat_donor and accidental_correct_retries < MAX_DONOR_RETRIES:
                # Same (sentence, predicate, question) as an earlier injection
                # in this loop -- re-roll rather than let the second copy
                # collide on the same adapted question and duplicate the
                # first one's answer onto itself.
                continue
            if accidental_correct_retries >= MAX_DONOR_RETRIES:
                if is_repeat_donor:
                    notes.append(
                        f"  [{i+1}/{k}] Skipped injection: after {MAX_DONOR_RETRIES} donor "
                        f"re-rolls, kept landing on donors already injected earlier in this "
                        f"Add pass ({donor_signature}) -- couldn't find a fresh donor to borrow."
                    )
                else:
                    notes.append(
                        f"  [{i+1}/{k}] Skipped injection: after {MAX_DONOR_RETRIES} donor re-rolls, "
                        f"every candidate's answer ({donor_qa.answers}) already appears as a TRUE gold "
                        f"answer for this predicate -- couldn't find a genuinely wrong answer to borrow."
                    )
                donor_qa = None
                break

        if donor_qa is None:
            continue  # this injection slot produced no valid negative; k effectively reduced by 1

        used_donor_signatures.add(donor_signature)

        new_question, new_slots, adapted_auto, adapt_note = adapt_question_to_predicate(
            donor_qa, group.verb_form
        )

        if forced_no_swap:
            adapted_auto = False
            adapt_note = (
                f"Every other group in this dataset shares verb_form "
                f"'{group.verb_form}' -- could not find a genuinely different "
                f"verb to borrow from. This 'foreign' pair is NOT a real "
                f"negative and should be removed or replaced by hand."
            )

        # EXTENSION POINT: future LLM grammar-fixing pass plugs in here.
        # Currently a no-op -- see maybe_llm_fix_question_grammar's docstring
        # for the intended wiring.
        new_question, needs_review_after_llm, adapt_note = maybe_llm_fix_question_grammar(
            new_question, not adapted_auto, adapt_note, group, src_sentence_text
        )

        # De-dupe the donor's own answers against themselves first (belt and
        # suspenders -- gold data shouldn't carry internal duplicates, but
        # cost nothing to guard against it here too).
        donor_answers, donor_starts, donor_ends = dedupe_new_answers(
            [], list(donor_qa.answers), list(donor_qa.answer_start_chars), list(donor_qa.answer_end_chars)
        )

        foreign_pair = QAPair(
            question=new_question,
            answers=donor_answers,       # verbatim -- see module docstring
            answer_start_chars=donor_starts,
            answer_end_chars=donor_ends,
            question_slots=new_slots,
            verb_form=group.verb_form,
            source_sentence_id=src_sent,
            source_predicate=src_pred,
            source_question=donor_qa.question,          # ORIGINAL question before adaptation
            source_sentence_text=src_sentence_text,      # donor's sentence (answers are spans into THIS)
            is_foreign=True,
            adapted_automatically=adapted_auto,
            needs_review=needs_review_after_llm,
            review_note=adapt_note,
        )

        existing_match = next((qa for qa in pairs if qa.question == new_question), None)
        if existing_match is not None:
            # Only merge in answers that aren't already sitting on this
            # question (whether from the original gold pair or an earlier
            # injection this same loop) -- never append a literal copy.
            unique_answers, unique_starts, unique_ends = dedupe_new_answers(
                existing_match.answers, foreign_pair.answers,
                foreign_pair.answer_start_chars, foreign_pair.answer_end_chars,
            )
            if not unique_answers:
                notes.append(
                    f"  [{i+1}/{k}] Collision: adapted question '{new_question}' already has "
                    f"this exact answer (donor: {src_sent}/{src_pred}, {source_note}) -- "
                    f"skipped as a pure duplicate, nothing added."
                )
            else:
                existing_match.answers.extend(unique_answers)
                existing_match.answer_start_chars.extend(unique_starts)
                existing_match.answer_end_chars.extend(unique_ends)
                existing_match.needs_review = existing_match.needs_review or foreign_pair.needs_review
                existing_match.review_note = (existing_match.review_note + " | " if existing_match.review_note else "") + \
                    f"Appended foreign answer(s) {unique_answers} from {src_sent}/{src_pred} " \
                    f"because adapted question collided with an existing gold question ({source_note})."
                dup_note = ""
                if len(unique_answers) < len(foreign_pair.answers):
                    dup_note = (
                        f" ({len(foreign_pair.answers) - len(unique_answers)} duplicate "
                        f"answer(s) already present were skipped.)"
                    )
                notes.append(f"  [{i+1}/{k}] Collision: merged foreign answer into existing question "
                             f"'{new_question}' (donor: {src_sent}/{src_pred}, {source_note}).{dup_note}")
        else:
            pairs.append(foreign_pair)
            notes.append(f"  [{i+1}/{k}] Injected new foreign question '{new_question}' "
                         f"(donor: {src_sent}/{src_pred}, {source_note}, "
                         f"auto-adapted={adapted_auto}).")

    return pairs, notes


# ----------------------------------------------------------------------------
# Step 5: top-level orchestration
# ----------------------------------------------------------------------------

def assign_processes(
    groups: dict[tuple[str, str], "Group"],
    rng: random.Random,
) -> dict[tuple[str, str], str]:
    """
    Decides, for every group, whether it goes through "add" or "truncate"
    (never both, never neither -- this replaces the old 3-way top-level
    choice). Rule, per the user's latest spec:

      1. Every group with EXACTLY 1 QA pair is forced to "add" (truncate is
         either impossible or destructive on a single pair: removing the
         pair would leave the rejected set empty, and "remove an answer"
         on a single-answer pair would leave a zero-answer question -- so
         singles always go to add, full stop).
      2. Every other group (>1 QA pair) is split between "add" and
         "truncate" so that, ACROSS THE WHOLE DATASET (including the
         forced-add singles from step 1), the overall split is 50/50.

    Worked example: if 10% of all groups have exactly 1 pair (forced add)
    and 90% have >1 pair, hitting 50% add overall means the >1-pair groups
    need (50% - 10%) / 90% ≈ 44.4% add and 55.6% truncate.

    We don't do this as an independent per-group coin flip at that
    probability, because independent flips only converge to the target
    ratio in expectation -- with thousands of groups the deviation would be
    small but nonzero, and there's no reason to accept that imprecision
    when an exact split is easy: shuffle the eligible (>1-pair) groups and
    assign the first N of them to "add", where N is computed exactly from
    the formula above (rounded to the nearest integer).

    Returns {(sentence_id, predicate): "add" | "truncate"}.
    """
    single_pair_keys = [k for k, g in groups.items() if len(g.qa_pairs) == 1]
    multi_pair_keys = [k for k, g in groups.items() if len(g.qa_pairs) > 1]

    n_total = len(single_pair_keys) + len(multi_pair_keys)
    n_single = len(single_pair_keys)
    n_multi = len(multi_pair_keys)

    assignment: dict[tuple[str, str], str] = {}
    for k in single_pair_keys:
        assignment[k] = "add"

    if n_multi == 0:
        log.info(f"All {n_single} groups have exactly 1 QA pair -- nothing to rebalance, "
                 f"everything goes through Add.")
        return assignment

    target_add_total = round(0.5 * n_total)
    add_needed_from_multi = target_add_total - n_single
    # Clamp: can't ask for negative adds (if singles alone already exceed 50%)
    # or more adds than there are multi-pair groups to give.
    add_needed_from_multi = max(0, min(n_multi, add_needed_from_multi))

    shuffled_multi = multi_pair_keys.copy()
    rng.shuffle(shuffled_multi)
    add_from_multi = set(shuffled_multi[:add_needed_from_multi])

    for k in multi_pair_keys:
        assignment[k] = "add" if k in add_from_multi else "truncate"

    actual_add_total = n_single + add_needed_from_multi
    log.info(
        f"Process split: {n_single} single-pair groups (all forced add) + "
        f"{n_multi} multi-pair groups, of which {add_needed_from_multi} ({add_needed_from_multi/n_multi:.1%}) "
        f"assigned to add and {n_multi - add_needed_from_multi} to truncate. "
        f"Overall: {actual_add_total}/{n_total} = {actual_add_total/n_total:.1%} add "
        f"(target was 50.0%)."
    )

    return assignment


def build_rejected_for_group(
    group: Group,
    process: str,
    all_groups: dict[tuple[str, str], Group],
    sentence_predicate_index: dict[str, list[str]],
    rng: random.Random,
) -> dict:
    """
    Runs ONE process (add or truncate, never both) for one group and returns
    a record with accepted, rejected, and full provenance metadata.
    """
    notes = [f"Top-level process: {process}."]

    if process == "add":
        rejected_pairs, sub_notes = apply_add(
            group, group.qa_pairs, all_groups, sentence_predicate_index, rng
        )
    elif process == "truncate":
        rejected_pairs, sub_notes = apply_truncate(group, rng)
    else:
        raise ValueError(f"Unknown process '{process}' for group {group.sentence_id}/{group.predicate}")

    notes.extend(sub_notes)
    needs_review = any(qa.needs_review for qa in rejected_pairs)

    return {
        "sentence_id": group.sentence_id,
        "predicate": group.predicate,
        "verb_form": group.verb_form,
        "sentence_text": group.sentence_text,
        "process": process,
        "accepted": [qa.to_dict() for qa in group.qa_pairs],
        "rejected": [qa.to_dict() for qa in rejected_pairs],
        "construction_log": notes,
        "needs_review": needs_review,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)

    raw_examples = download_dev_set()
    groups = build_groups(raw_examples)

    sentence_predicate_index: dict[str, list[str]] = defaultdict(list)
    for (s_id, pred) in groups.keys():
        sentence_predicate_index[s_id].append(pred)

    process_assignment = assign_processes(groups, rng)

    records = []
    for key, group in groups.items():
        if not group.qa_pairs:
            continue
        process = process_assignment[key]
        record = build_rejected_for_group(group, process, groups, sentence_predicate_index, rng)
        records.append(record)

    n_realigned = normalize_answer_offsets(records)
    if n_realigned:
        log.info(f"Re-aligned {n_realigned} answer offset(s) to their own reference sentence.")

    n_flagged = sum(1 for r in records if r["needs_review"])
    n_add = sum(1 for r in records if r["process"] == "add")
    n_truncate = sum(1 for r in records if r["process"] == "truncate")
    log.info(f"Built {len(records)} DPO records: {n_add} add ({n_add/len(records):.1%}), "
             f"{n_truncate} truncate ({n_truncate/len(records):.1%}). "
             f"{n_flagged} flagged for manual review ({n_flagged / len(records):.1%}).")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote {OUTPUT_PATH}. Open it in review_tool.html to inspect/edit.")


if __name__ == "__main__":
    main()