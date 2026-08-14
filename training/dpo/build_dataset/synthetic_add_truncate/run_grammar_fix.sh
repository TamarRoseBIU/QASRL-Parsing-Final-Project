#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --job-name=qwen3_grammar_fix
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --gres=gpu:1                # Single GPU — script pins to device 0
#SBATCH --mem=64G
#SBATCH --time=01:00:00             # Grammar-fix pass over ~350 short prompts is fast; bump up if --sample shows otherwise

# ── Log Paths ─────────────────────────────────────────────────────────────────
#SBATCH --output=../logs/Stage_DPO/grammar_fix/grammar_fix_job_%j.out
#SBATCH --error=../logs/Stage_DPO/grammar_fix/grammar_fix_job_%j.err

#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --mail-type=END,FAIL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="${QASRL_CONDA_ENV:?set QASRL_CONDA_ENV to your train_qwen3 env prefix}/bin:$PATH"

PYTHON_BIN="${QASRL_CONDA_ENV:?set QASRL_CONDA_ENV to your train_qwen3 env prefix}/bin/python"
DATASET_DIR="${QASRL_DATASET_DIR:-$SCRIPT_DIR}"
DIFF_DIR="$DATASET_DIR/grammar_fix_diffs"
SCRIPT="$SCRIPT_DIR/llm_fix_grammar.py"

echo "Testing Python: $PYTHON_BIN"
$PYTHON_BIN -c "import torch; print('SUCCESS: Torch loaded!'); print('GPU:', torch.cuda.is_available())"
$PYTHON_BIN -c "import vllm; print('SUCCESS: vLLM loaded! Version:', vllm.__version__)"

# ── Sanity Checks ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU(s)      : $CUDA_VISIBLE_DEVICES"
echo "Start time  : $(date)"
echo "=========================================="
nvidia-smi

mkdir -p "$DATASET_DIR" "$DIFF_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
# Full pass over every flagged (foreign + needs_review) question. Writes
# DATASET_DIR/dpo_pairs.grammar_fixed.json -- never overwrites the input file.
# Also writes a changes log (.changes.txt) and a timestamped .diff file into
# DIFF_DIR -- if you re-run this job after tweaking the prompt or model, you
# can compare two runs directly with:
#     diff $DIFF_DIR/changes_<run_a>.diff $DIFF_DIR/changes_<run_b>.diff
#
# IMPORTANT: run the --sample version of this job FIRST (see the commented
# command below) and read through the printed before/after pairs before
# submitting this full run -- the full run costs real GPU time on a shared
# cluster, and there's no reason to spend it before checking quality on a
# small sample.
#
#   To sample instead, change the line below to:
#     $PYTHON_BIN "$SCRIPT" --input "$DATASET_DIR/dpo_pairs.json" --sample 30
#
JOB_START=$(date +%s)

$PYTHON_BIN "$SCRIPT" \
    --input "$DATASET_DIR/dpo_pairs.json" \
    --output "$DATASET_DIR/dpo_pairs.grammar_fixed.json" \
    --diff-dir "$DIFF_DIR"

# ── Timing Report ─────────────────────────────────────────────────────────────
JOB_END=$(date +%s)
JOB_SECS=$(( JOB_END - JOB_START ))
JOB_HRS=$(( JOB_SECS / 3600 ))
JOB_MINS=$(( (JOB_SECS % 3600) / 60 ))
JOB_SECS_REM=$(( JOB_SECS % 60 ))

echo "=========================================="
echo "Job finished      : $(date)"
echo "Job duration      : ${JOB_HRS}h ${JOB_MINS}m ${JOB_SECS_REM}s"
echo "=========================================="
