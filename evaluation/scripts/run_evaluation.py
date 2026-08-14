"""
QASRL Model Evaluation Script

Usage:
    python scripts/run_evaluation.py <system_predictions.csv> <ground_truth.csv> [--sentences sentences.csv]

Example:
    python scripts/run_evaluation.py \\
        data/model_output_filled_slots/t5_small_qaset_loss/passive_red_output_QASetLoss_filled_slots.csv \\
        data/gold/passive_red_test_gold_detokenized_normalized.csv
"""

import io
import os
import sys
import datetime
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from evaluate_dataset import main as evaluate_main


# ============================================================================
# Output file naming
# ============================================================================
def build_output_filename(system_path: str) -> str:
    """
    Build output filename from the system path.
    e.g. data/model_output_filled_slots/t5_small_qaset_loss/passive_red_output_QASetLoss_filled_slots.csv
      -> t5_small_qaset_loss~passive_red_output_QASetLoss_filled_slots
    """
    p = Path(system_path)
    model_name = p.parent.name          # t5_small_qaset_loss
    file_stem  = p.stem                 # passive_red_output_QASetLoss_filled_slots
    return f"{model_name}~{file_stem}"


def get_results_dir() -> Path:
    """
    Always save results to QSRL_evaluate/results/,
    regardless of where the script is called from.
    """
    script_dir  = Path(__file__).resolve().parent   # .../QSRL_evaluate/scripts
    project_dir = script_dir.parent                 # .../QSRL_evaluate
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


# ============================================================================
# Aesthetic report builder
# ============================================================================
def build_report(
    system_path: str,
    ground_truth_path: str,
    sentences_path: str,
    eval_output: str,
    timestamp: str,
) -> str:
    """Wrap the raw evaluation output in a readable report."""
    w = 70  # line width

    def header(title: str) -> str:
        pad = (w - len(title) - 2) // 2
        return f"{'═' * pad} {title} {'═' * (w - pad - len(title) - 2)}"

    lines = [
        header("QA-SRL EVALUATION REPORT"),
        "",
        f"  Timestamp  : {timestamp}",
        f"  Model      : {Path(system_path).parent.name}",
        f"  Predictions: {system_path}",
        f"  Gold       : {ground_truth_path}",
    ]
    if sentences_path:
        lines.append(f"  Sentences  : {sentences_path}")
    lines += [
        "",
        "─" * w,
        "",
        eval_output.strip(),
        "",
        "═" * w,
        "  Evaluation complete.",
        "═" * w,
    ]
    return "\n".join(lines)


# ============================================================================
# Core evaluation runner
# ============================================================================
def run_evaluation(system_path: str, ground_truth_path: str, sentences_path: str = None):
    """
    Evaluate a QA-SRL model's predictions against ground truth.
    Prints results to terminal AND saves an aesthetic report to results/.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Terminal header ──────────────────────────────────────────────────────
    print("=" * 70)
    print("QA-SRL Model Evaluation")
    print("=" * 70)
    print(f"\n  System predictions : {system_path}")
    print(f"  Ground truth       : {ground_truth_path}")
    if sentences_path:
        print(f"  Sentences          : {sentences_path}")
    print("\n" + "-" * 70 + "\n")

    # ── Capture stdout from evaluate_main ───────────────────────────────────
    buffer = io.StringIO()
    tee    = _TeeStream(sys.stdout, buffer)   # write to both terminal & buffer

    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        evaluate_main(system_path, ground_truth_path, sentences_path)
    finally:
        sys.stdout = original_stdout

    captured = buffer.getvalue()

    # Strip "found paraphrase" lines (and their content) from the saved file
    filtered = "\n".join(
        line for line in captured.splitlines()
        if not line.strip().startswith("found paraphrase")
    )

    # ── Terminal footer ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)

    # ── Save report ─────────────────────────────────────────────────────────
    results_dir  = get_results_dir()
    output_name  = build_output_filename(system_path)
    output_path  = results_dir / f"{output_name}.txt"

    report = build_report(system_path, ground_truth_path, sentences_path,
                          filtered, timestamp)

    output_path.write_text(report, encoding="utf-8")
    print(f"\n  Results saved to: {output_path}")


# ============================================================================
# Helper: write to two streams simultaneously
# ============================================================================
class _TeeStream:
    """Forwards writes to two file-like objects at once (terminal + buffer)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ============================================================================
# CLI
# ============================================================================
from summarize_results import main as update_summary
if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate QA-SRL model predictions")
    parser.add_argument("system_path",
                        help="Path to CSV file with system predictions")
    parser.add_argument("ground_truth_path",
                        help="Path to CSV file with ground truth annotations")
    parser.add_argument("-s", "--sentences_path",
                        required=False,
                        default=None,
                        help="Optional: Path to CSV file with sentences")

    args = parser.parse_args()

    # 1. Run the actual evaluation
    run_evaluation(args.system_path, args.ground_truth_path, args.sentences_path)

    # 2. Automatically update the comparison table
    print("\nRefreshing summary table...")
    try:
        update_summary()
    except Exception as e:
        print(f"Note: Evaluation succeeded but summary update failed: {e}")