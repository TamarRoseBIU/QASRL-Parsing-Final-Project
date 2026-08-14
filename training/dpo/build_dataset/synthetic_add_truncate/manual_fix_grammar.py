# ==========================================================================
# Applies hand-authored corrections on top of the LLM grammar pass, producing
# dpo_pairs.grammar_fixed.manual_fixed.json (the dataset shipped in this directory).
#
# Part of the SUPERSEDED add/truncate DPO arm -- kept for the record only.
# The reported DPO result (79.59) came from the on-policy pairs in ../ .
# ==========================================================================
"""
manual_fix_grammar.py

Interactive, CPU-only manual review for the handful of questions
llm_fix_grammar.py could not fix automatically (status PARSE_FAILED or
VALIDATION_FAILED, still needs_review == True).

No torch, no vLLM, no GPU -- just JSON in, your typed corrections, JSON out.
Safe to run on a login node.

USAGE
    # Just see how many are left and what they are, don't write anything:
    python3 manual_fix_grammar.py --input dpo_pairs.grammar_fixed.json --dry-run

    # Interactive fix session:
    python3 manual_fix_grammar.py --input dpo_pairs.grammar_fixed.json

Writes a NEW file (default '<input>.manual_fixed.json'), never overwrites
--input. You can quit mid-session (Ctrl+C or 'q') and re-run later -- items
you already fixed in a previous partial run are skipped automatically if you
pass that run's output back in as --input. You can also drop a pair
entirely ('d') if the sentence/question was a bad template match with no
sensible fix -- it is removed from that record's "rejected" list.
"""

import argparse
import json
import sys
from dataclasses import dataclass


@dataclass
class FailedQuestion:
    record_idx: int
    qa_idx: int
    sentence_id: str
    original_question: str
    answers: list
    sentence_text: str
    verb_form: str
    llm_status: str
    llm_detail: str


def find_failed_questions(records: list[dict]) -> list[FailedQuestion]:
    """
    Same traversal shape as llm_fix_grammar.py's load_flagged_questions,
    but narrowed to items the LLM pass already tried and failed on:
    needs_review is still True AND llm_grammar_fix.status != "OK".

    Items that were never attempted by the LLM pass (no llm_grammar_fix
    field at all) are NOT included -- those are a different bucket, not
    "failed", and manual_fix_grammar.py is only for the failures.
    """
    failed = []
    for r_idx, record in enumerate(records):
        sentence_text = record.get("sentence_text", "")
        sentence_id = record.get("sentence_id", "")
        for qa_idx, qa in enumerate(record.get("rejected", [])):
            if not qa.get("needs_review"):
                continue
            fix_info = qa.get("llm_grammar_fix")
            if fix_info is None:
                continue  # never attempted by the LLM pass -- not our bucket
            if fix_info.get("status") == "OK":
                continue  # shouldn't happen (OK clears needs_review), but be safe
            failed.append(FailedQuestion(
                record_idx=r_idx,
                qa_idx=qa_idx,
                sentence_id=sentence_id,
                original_question=qa.get("question", ""),
                answers=qa.get("answers", []) or [],
                sentence_text=sentence_text,
                verb_form=qa.get("verb_form", ""),
                llm_status=fix_info.get("status", "UNKNOWN"),
                llm_detail=fix_info.get("detail", ""),
            ))
    return failed


QUIT = object()   # sentinel: user typed 'q'
DROP = object()   # sentinel: user typed 'd' -- remove the pair entirely


def prompt_for_fix(fq: FailedQuestion, idx: int, total: int):
    """
    Shows one failed item's context and asks for a corrected question.
    Returns (new_question, changed) normally, where changed is False if the
    user kept the original as-is (empty input) or typed 's' to skip.
    Returns (QUIT, False) if the user typed 'q' -- caller stops the loop but
    keeps everything fixed so far.
    Returns (DROP, False) if the user typed 'd' -- caller removes this pair
    from the dataset entirely.

    Special inputs:
      <empty>  -- keep the original question unchanged, mark reviewed
      s        -- skip entirely, leave needs_review True, revisit later
      d        -- drop this pair from the dataset (e.g. bad template match --
                  no sensible question exists for this sentence)
      q        -- quit the session; already-fixed items are still saved
    """
    print("-" * 70)
    print(f"[{idx}/{total}]  record {fq.record_idx}, qa {fq.qa_idx}")
    print(f"sentence_id: {fq.sentence_id}")
    print(f"sentence:    {fq.sentence_text}")
    print(f"verb_form:   {fq.verb_form}")
    print(f"llm status:  {fq.llm_status}  ({fq.llm_detail})")
    print(f"question:    {fq.original_question}")
    print(f"answer(s):   {fq.answers}")
    print()
    raw = input("fixed question (Enter=keep as-is, s=skip, d=drop pair, q=quit): ").strip()

    if raw.lower() == "q":
        return QUIT, False
    if raw.lower() == "d":
        return DROP, False
    if raw.lower() == "s":
        return fq.original_question, False
    if raw == "":
        return fq.original_question, True  # keep text, but mark as reviewed
    return raw, True


def apply_manual_fixes(records: list[dict], failed: list[FailedQuestion], dry_run: bool) -> dict:
    import copy
    updated = copy.deepcopy(records)
    counts = {"fixed": 0, "kept_unchanged": 0, "skipped": 0, "dropped": 0}

    total = len(failed)
    print(f"{total} sentence(s) still need review after the LLM pass.\n")

    if dry_run or total == 0:
        for fq in failed:
            print(f"  record {fq.record_idx}, qa {fq.qa_idx} [sentence_id={fq.sentence_id}]: "
                  f"\"{fq.original_question}\" -> {fq.answers} "
                  f"[{fq.llm_status}: {fq.llm_detail}]")
        return updated, counts

    to_drop = []  # (record_idx, qa_idx) pairs -- deleted after the loop so
                   # earlier qa_idx values in the same record stay valid
                   # while we're still iterating over `failed`.

    for i, fq in enumerate(failed, start=1):
        new_question, changed = prompt_for_fix(fq, i, total)

        if new_question is QUIT:
            print("\nQuitting early. Everything fixed so far will still be written to --output.")
            counts["skipped"] += (total - i + 1)  # this one plus everything not yet reached
            break

        if new_question is DROP:
            to_drop.append((fq.record_idx, fq.qa_idx))
            counts["dropped"] += 1
            continue

        qa = updated[fq.record_idx]["rejected"][fq.qa_idx]

        if not changed:
            counts["skipped"] += 1
            continue

        qa["question"] = new_question
        qa["needs_review"] = False
        qa["manual_grammar_fix"] = {
            "original_question": fq.original_question,
            "applied_question": new_question,
            "was_changed": new_question != fq.original_question,
            "previous_llm_status": fq.llm_status,
        }
        if new_question != fq.original_question:
            counts["fixed"] += 1
        else:
            counts["kept_unchanged"] += 1

    # Delete dropped pairs now, in reverse qa_idx order per record, so
    # removing one doesn't shift the index of another still-pending delete
    # in the same record's "rejected" list.
    by_record: dict[int, list[int]] = {}
    for r_idx, qa_idx in to_drop:
        by_record.setdefault(r_idx, []).append(qa_idx)
    for r_idx, qa_idxs in by_record.items():
        for qa_idx in sorted(qa_idxs, reverse=True):
            del updated[r_idx]["rejected"][qa_idx]

    return updated, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help="Path to llm_fix_grammar.py's output "
                              "(e.g. dpo_pairs.grammar_fixed.json).")
    parser.add_argument("--output", default=None,
                         help="Path to write the manually-fixed JSON. "
                              "Defaults to '<input>.manual_fixed.json'. Never overwrites --input.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Just print the count and list of still-failing sentences, "
                              "don't prompt for fixes or write any file.")
    args = parser.parse_args()

    if args.output is None:
        base = args.input[:-5] if args.input.endswith(".json") else args.input
        args.output = base + ".manual_fixed.json"
    if args.output == args.input:
        raise SystemExit("Refusing to run: --output would overwrite --input. Pick a different output path.")

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    failed = find_failed_questions(records)

    if not failed:
        print("Nothing to do -- no leftover failed questions found. Exiting without writing any file.")
        return

    updated_records, counts = apply_manual_fixes(records, failed, args.dry_run)

    if args.dry_run:
        return

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done. fixed={counts['fixed']}  kept_unchanged={counts['kept_unchanged']}  "
          f"dropped={counts['dropped']}  skipped={counts['skipped']}")
    print(f"Wrote {args.output}.")
    if counts["skipped"]:
        print("Note: skipped items still have needs_review=True -- re-run this script on "
              f"{args.output} to pick up where you left off.")


if __name__ == "__main__":
    main()