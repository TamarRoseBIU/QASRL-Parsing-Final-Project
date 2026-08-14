# ==========================================================================
# Read-only inspection helper: prints a record's original sentence alongside its
# accepted/rejected pairs, for eyeballing individual corruptions.
#
# Part of the SUPERSEDED add/truncate DPO arm -- kept for the record only.
# The reported DPO result (79.59) came from the on-policy pairs in ../ .
# ==========================================================================
"""
print_sentence_qas.py

Given a sentence_id, prints the ORIGINAL sentence and ALL of its gold QA
pairs exactly as they come from dev.json -- the same raw input
build_dpo_data.py starts from, before grouping, ADD/TRUNCATE, or any later
grammar-fix pass touches anything.

Useful for sanity-checking a record you saw flagged in the manual review
step: pull up every predicate/question/answer dev.json ever had for that
sentence, not just the one pair that happened to fail.

USAGE
    # First run: downloads dev.json and caches it locally so you don't
    # re-download every time you look up a sentence.
    python3 print_sentence_qas.py --sentence-id "Wiki1k:wikinews:1002218:0:0"

    # Subsequent runs reuse the cache automatically:
    python3 print_sentence_qas.py --sentence-id "Wiki1k:wikinews:1002218:0:0"

    # Force a fresh download:
    python3 print_sentence_qas.py --sentence-id "..." --refresh-cache

    # Point at an already-downloaded dev.json instead of fetching at all:
    python3 print_sentence_qas.py --sentence-id "..." --dev-json /path/to/dev.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

DEV_URL = "https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/dev.json"
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev.json.cache")


def _load_json_or_jsonl(path: str) -> list[dict]:
    """
    Handles three shapes people call ".jsonl":
      1. A real JSON file (dict-of-rows or list-of-rows) -- just named .jsonl.
      2. True JSONL: one complete JSON object per line.
      3. A single JSON array, but pretty-printed across many lines (so it
         LOOKS like multi-line JSONL but each individual line isn't valid
         JSON on its own).

    Strategy: try parsing the whole file as one JSON document first (covers
    #1 and #3). Only fall back to line-by-line parsing (#2) if that fails.
    If the whole-file parse succeeds but hands back something that isn't a
    flat list of row-dicts (e.g. a list containing one big list, from a
    single-line-array file), flatten one level.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        rows = list(data.values()) if isinstance(data, dict) else list(data)
    except json.JSONDecodeError:
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Flatten one level if rows turned out to be [[...]] instead of [...]
    if len(rows) == 1 and isinstance(rows[0], list):
        rows = rows[0]

    return rows


def load_raw_examples(dev_json_path: Optional[str], cache_path: str, refresh_cache: bool) -> list[dict]:
    """
    Same loading logic as build_dpo_data.download_dev_set (dict-of-rows or
    list-of-rows), but with an optional local cache so repeated lookups
    don't re-download the whole dev set every time.
    """
    if dev_json_path:
        return _load_json_or_jsonl(dev_json_path)

    if os.path.exists(cache_path) and not refresh_cache:
        print(f"(using cached dev.json at {cache_path} -- pass --refresh-cache to re-download)")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.values()) if isinstance(data, dict) else list(data)

    import requests  # only needed on the download path
    print(f"Downloading dev set from {DEV_URL} ...")
    response = requests.get(DEV_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Cached to {cache_path} for next time.")
    return list(data.values()) if isinstance(data, dict) else list(data)


def print_sentence_qas(raw_examples: list[dict], sentence_id: str) -> None:
    matches = [ex for ex in raw_examples if ex.get("sentence_id") == sentence_id]

    if not matches:
        print(f"No rows found for sentence_id={sentence_id!r}. "
              f"Double check the id (it must match exactly, including the trailing :0:0 etc.)")
        return

    sentence_text = None
    by_predicate: dict[str, list[dict]] = defaultdict(list)
    for ex in matches:
        det = ex.get("detokenized", {}) or {}
        if sentence_text is None and det.get("sentence"):
            sentence_text = det["sentence"]
        by_predicate[ex.get("predicate", "?")].append(ex)

    print("=" * 70)
    print(f"sentence_id: {sentence_id}")
    print(f"sentence:    {sentence_text}")
    print(f"predicates:  {len(by_predicate)}   total QA rows: {len(matches)}")

    for predicate, rows in by_predicate.items():
        verb_form = rows[0].get("verb_form", "")
        print("-" * 70)
        print(f"predicate: {predicate!r}   verb_form: {verb_form!r}")
        for row in rows:
            det = row.get("detokenized", {}) or {}
            question = det.get("question", "")
            answers_field = det.get("answers", {}) or {}
            answer_texts = list(answers_field.get("text", []) or [])
            print(f"  Q: {question}")
            if answer_texts:
                for a in answer_texts:
                    print(f"     A: {a}")
            else:
                print("     A: (no answers listed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-id", required=True,
                         help='e.g. "Wiki1k:wikinews:1002218:0:0"')
    parser.add_argument("--dev-json", default=None,
                         help="Path to an already-downloaded dev.json. If given, skips "
                              "download/cache entirely and reads straight from this file.")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH,
                         help=f"Where to cache a downloaded dev.json (default: {DEFAULT_CACHE_PATH}).")
    parser.add_argument("--refresh-cache", action="store_true",
                         help="Re-download dev.json even if a cache already exists.")
    args = parser.parse_args()

    raw_examples = load_raw_examples(args.dev_json, args.cache, args.refresh_cache)
    print_sentence_qas(raw_examples, args.sentence_id)


if __name__ == "__main__":
    main()