#!/usr/bin/env python
"""
End-to-end pipeline driver: SFT -> RL -> inference -> evaluation.

This is an ORCHESTRATOR ONLY. It contains no training, inference or scoring
logic of its own: every stage is the existing entry point, launched as a
subprocess with the environment each stage already documents in its own
config.yaml. Hyperparameters are READ FROM those config.yaml files rather than
restated here, so editing a config still changes the run.

    python run_full_pipeline.py --sft_data TRAIN --rl_method GRPO

Stage chaining
--------------
Each stage announces where it saved its adapter, e.g.

    Stage_CE_Instruct_DEV.py  -> "Training complete. Run '<name>' saved to <dir>"
    Stage_GRPO_Instruct_DEV.py-> "GRPO stage complete. Adapter saved to <dir>"
    Stage_DPO_Instruct_DEV.py -> "DPO stage complete. Final adapter saved to <dir>"

We parse that line to learn the output directory instead of re-deriving each
stage's RUN_NAME (which depends on the split, the arm tag and the reward
settings). If the line is missing we fall back to the newest adapter directory
under that stage's models_save_baseline/<STAGE>/ tree, and fail loudly if
neither yields a directory containing adapter_config.json.

The SFT adapter is handed to the RL stage through QASRL_SFT_ADAPTER, which both
RL trainers already honour; the selected RL adapter is handed to inference via
--model_path.

Scope
-----
Nothing under evaluation/ is modified: the gold files, the model-input work list
and the scoring scripts are used exactly as they ship. Inference writes new
files tagged with --tag, and results/ is left alone (summarize_results.py is not
invoked).

Checkpoint selection
--------------------
The reported RL results come from a MID-RUN checkpoint, not the final adapter
(GRPO ~checkpoint-3600; DPO checkpoint-150), so selection is ON BY DEFAULT: the
existing eval_on_val.py is run over every checkpoint plus the final adapter, and
the winner on the dev holdout is what gets evaluated. eval_on_val.py is a general
PEFT selector and its holdout is a deterministic, model-independent split of dev,
so it applies to a GRPO run as well as a DPO one.

Use --rl_checkpoint N to name a checkpoint directly, or --no_select to evaluate
the final adapter (faster, but not comparable to the reported numbers).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not logic
    sys.exit("PyYAML is required to read the stage config.yaml files "
             "(`pip install pyyaml`, or run this with the `eval` conda env's python).")

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
EVAL_DIR = REPO_ROOT / "evaluation"

log = logging.getLogger("pipeline")


class StageError(RuntimeError):
    """Raised when a pipeline stage fails; carries the stage name."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


# ---------------------------------------------------------------------------
# Config / environment helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    if not path.is_file():
        raise StageError("setup", f"missing config file: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_python(env_name: str, override: str | None) -> str:
    """
    Find the interpreter for a named conda env.

    Order: explicit --python_* override, then QASRL_PYTHON_<ENV>, then the
    conda base reported by `conda info --base`, then the current interpreter
    (with a warning -- correct only if it already has the right packages).
    """
    if override:
        return override

    env_var = f"QASRL_PYTHON_{env_name.upper()}"
    if os.environ.get(env_var):
        return os.environ[env_var]

    conda = shutil.which("conda")
    if conda:
        try:
            base = subprocess.run([conda, "info", "--base"], capture_output=True,
                                  text=True, timeout=60).stdout.strip()
            candidate = Path(base) / "envs" / env_name / "bin" / "python"
            if candidate.is_file():
                return str(candidate)
        except (subprocess.SubprocessError, OSError):
            pass

    log.warning("could not locate conda env %r; falling back to %s. "
                "Set %s to point at the right interpreter.",
                env_name, sys.executable, env_var)
    return sys.executable


def stage_env(base_dir: Path, extra: dict | None = None,
              pythonpath: Path | None = None) -> dict:
    """Build a subprocess environment: inherited + QASRL_BASE_DIR + stage vars."""
    env = os.environ.copy()
    env["QASRL_BASE_DIR"] = str(base_dir)
    if pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{existing}" if existing else str(pythonpath)
    for key, value in (extra or {}).items():
        env[str(key)] = str(value)
    return env


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------
def run_stage(stage: str, cmd: list[str], cwd: Path, env: dict,
              log_path: Path, dry_run: bool) -> str:
    """
    Run one stage, streaming output to the console and to log_path.

    Returns the captured output. Raises StageError on a non-zero exit.
    """
    printable = " ".join(str(c) for c in cmd)
    log.info("[%s] cwd=%s", stage, cwd)
    log.info("[%s] cmd: %s", stage, printable)
    log.info("[%s] log: %s", stage, log_path)

    if dry_run:
        log.info("[%s] DRY RUN - not executed", stage)
        return ""

    if not cwd.is_dir():
        raise StageError(stage, f"working directory does not exist: {cwd}")

    started = _dt.datetime.now()
    captured: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_fh:
        log_fh.write(f"# stage   : {stage}\n# cwd     : {cwd}\n"
                     f"# command : {printable}\n# started : {started}\n\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
            captured.append(line)
        returncode = proc.wait()

    elapsed = _dt.datetime.now() - started
    output = "".join(captured)
    if returncode != 0:
        raise StageError(stage, f"exit code {returncode} after {elapsed}. "
                                f"Full output: {log_path}")
    log.info("[%s] OK (%s)", stage, elapsed)
    return output


# ---------------------------------------------------------------------------
# Adapter resolution
# ---------------------------------------------------------------------------
def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file()


def resolve_adapter(stage: str, output: str, pattern: str,
                    fallback_root: Path, dry_run: bool) -> Path:
    """
    Learn a stage's output adapter from the line it printed; fall back to the
    newest adapter directory under fallback_root.
    """
    if dry_run:
        return fallback_root / "<RUN_NAME>"

    match = None
    for match in re.finditer(pattern, output):
        pass  # keep the last occurrence
    if match:
        candidate = Path(match.group("dir").strip().rstrip("."))
        if is_adapter_dir(candidate):
            log.info("[%s] adapter (from stage output): %s", stage, candidate)
            return candidate
        log.warning("[%s] announced %s but it has no adapter_config.json; "
                    "falling back to a directory scan", stage, candidate)
    else:
        log.warning("[%s] stage printed no recognisable save path; "
                    "falling back to a directory scan under %s", stage, fallback_root)

    if not fallback_root.is_dir():
        raise StageError(stage, f"cannot locate the output adapter: {fallback_root} "
                                f"does not exist")
    candidates = [d for d in fallback_root.iterdir() if d.is_dir() and is_adapter_dir(d)]
    if not candidates:
        raise StageError(stage, f"cannot locate the output adapter: no directory with "
                                f"adapter_config.json under {fallback_root}")
    newest = max(candidates, key=lambda d: d.stat().st_mtime)
    log.info("[%s] adapter (newest under %s): %s", stage, fallback_root, newest)
    return newest


def checkpoint_parent(adapter_dir: Path) -> Path:
    """Map models_save_baseline/<stage>/<run> -> trainer_runs_baseline/<stage>/<run>."""
    parts = list(adapter_dir.parts)
    for i, part in enumerate(parts):
        if part == "models_save_baseline":
            parts[i] = "trainer_runs_baseline"
            return Path(*parts)
    return adapter_dir


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def run_sft(args, base_dir: Path, python: str, log_dir: Path) -> Path:
    stage = "sft"
    cfg = load_config(TRAINING_DIR / "sft" / "config.yaml")
    cwd = TRAINING_DIR / "sft"

    pipeline_key = "train" if args.sft_data == "TRAIN" else "dev"
    pipelines = cfg.get("pipelines", {})
    entrypoint = pipelines.get(pipeline_key, {}).get("entrypoint", "Stage_CE_Instruct_DEV.py")

    # QASRL_SFT_TRAIN_ON is what the script actually reads; the config records it too.
    env_extra = dict(pipelines.get(pipeline_key, {}).get("env", {}))
    env_extra["QASRL_SFT_TRAIN_ON"] = args.sft_data.lower()

    log.info("[%s] split=%s entrypoint=%s", stage, args.sft_data, entrypoint)
    output = run_stage(stage, [python, entrypoint], cwd,
                       stage_env(base_dir, env_extra),
                       log_dir / "01_sft.log", args.dry_run)

    return resolve_adapter(
        stage, output, r"saved to (?P<dir>\S+)",
        base_dir / "models_save_baseline" / "Stage_CE", args.dry_run)


def run_rl(args, base_dir: Path, python: str, sft_adapter: Path, log_dir: Path) -> Path:
    method = args.rl_method
    stage = method.lower()
    cwd = TRAINING_DIR / stage
    cfg = load_config(cwd / "config.yaml")

    if method == "GRPO":
        entrypoint = cfg.get("entrypoint", "Stage_GRPO_Instruct_DEV.py")
        env_extra = dict(cfg.get("env", {}))
        save_stage = "Stage_GRPO_Instruct_DEV"
    else:  # DPO
        train_cfg = cfg.get("train", {})
        entrypoint = train_cfg.get("entrypoint", "Stage_DPO_Instruct_DEV.py")
        env_extra = dict(train_cfg.get("env", {}))
        data_dir = train_cfg.get("data_dir", "existing_dataset/pairs_onpolicy_recall")
        if not (cwd / data_dir).is_dir() and not args.dry_run:
            raise StageError(stage, f"preference pairs not found: {cwd / data_dir}. "
                                    f"Build them with training/dpo/build_dataset/ first "
                                    f"(see training/dpo/config.yaml, step 0).")
        env_extra["DPO_DATA_DIR"] = data_dir
        save_stage = "Stage_DPO_Instruct_DEV"

    # The chaining hook: both RL trainers resolve their warm start from this.
    env_extra["QASRL_SFT_ADAPTER"] = str(sft_adapter)

    log.info("[%s] warm start: %s", stage, sft_adapter)
    log.info("[%s] config env: %s", stage,
             ", ".join(f"{k}={v}" for k, v in sorted(env_extra.items())) or "(none)")

    output = run_stage(stage, [python, entrypoint], cwd,
                       stage_env(base_dir, env_extra, pythonpath=TRAINING_DIR / "shared"),
                       log_dir / f"02_{stage}.log", args.dry_run)

    return resolve_adapter(
        stage, output, r"(?:[Aa]dapter saved to|saved to) (?P<dir>\S+)",
        base_dir / "models_save_baseline" / save_stage, args.dry_run)


def select_checkpoint(args, base_dir: Path, python: str, rl_adapter: Path,
                      log_dir: Path) -> Path:
    """Rank checkpoints with the existing eval_on_val.py and return the winner."""
    stage = "select"
    cwd = TRAINING_DIR / "dpo"          # eval_on_val.py is a general PEFT selector
    cfg = load_config(cwd / "config.yaml")
    sel = cfg.get("select", {})
    entrypoint = sel.get("entrypoint", "eval_on_val.py")
    val = sel.get("val_holdout", "existing_dataset/pairs_onpolicy_recall/dpo_val.jsonl")
    out_json = log_dir / "selection.json"

    ckpt_parent = checkpoint_parent(rl_adapter)
    ckpts = sorted(ckpt_parent.glob("checkpoint-*")) if ckpt_parent.is_dir() else []
    if not ckpts and not args.dry_run:
        log.warning("[%s] no checkpoint-* under %s; keeping the final adapter",
                    stage, ckpt_parent)
        return rl_adapter

    cmd = [python, entrypoint, "--ckpts", *[str(c) for c in ckpts], str(rl_adapter),
           "--val", val, "--out", str(out_json)]
    run_stage(stage, cmd, cwd,
              stage_env(base_dir, pythonpath=TRAINING_DIR / "shared"),
              log_dir / f"03_{stage}.log", args.dry_run)

    if args.dry_run:
        return rl_adapter
    try:
        import json
        with open(out_json) as fh:
            result = json.load(fh)

        # eval_on_val.py writes a LIST of per-checkpoint records already sorted by
        # descending val_micro_f1, so element 0 is the winner. Dict shapes are also
        # accepted so a hand-written selection file still works.
        best = None
        if isinstance(result, list) and result:
            best = result[0]
        elif isinstance(result, dict):
            best = (result.get("best") or result.get("best_ckpt")
                    or result.get("best_checkpoint"))
        if isinstance(best, dict):
            best = best.get("ckpt") or best.get("path")

        if best:
            # Recorded selection files may carry a literal $QASRL_BASE_DIR prefix.
            best = os.path.expandvars(str(best))
            log.info("[%s] selected: %s", stage, best)
            return Path(best)
        log.warning("[%s] %s holds no usable checkpoint entry; keeping the final adapter",
                    stage, out_json)
    except (OSError, ValueError, AttributeError, KeyError, IndexError) as exc:
        log.warning("[%s] could not read %s (%s); keeping the final adapter",
                    stage, out_json, exc)
    return rl_adapter


def run_inference_and_eval(args, python_infer: str, python_score: str,
                           adapter: Path, log_dir: Path) -> Path:
    """Inference -> dummy slots -> scoring, exactly as evaluation/config.yaml documents."""
    cfg = load_config(EVAL_DIR / "config.yaml")
    paths = cfg.get("paths", {})
    model_input = paths.get("model_input", "./data/model_input/passive_red.model_inputs.csv")
    gold = paths.get("gold", "./data/gold/gold_updated_passive_filled_slots.csv")
    raw_dir = paths.get("raw_output",
                        "./data/model_output/Qwen3-30B-A3B-Instruct-2507/")
    filled_dir = paths.get("filled_slots",
                           "./data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/")

    raw_csv = f"{raw_dir.rstrip('/')}/pipeline_{args.tag}.csv"
    filled_csv = f"{filled_dir.rstrip('/')}/pipeline_{args.tag}_filled_slots.csv"

    for rel in (raw_csv, filled_csv):
        (EVAL_DIR / rel).parent.mkdir(parents=True, exist_ok=True)

    # Step 1 - GPU inference
    run_stage("inference",
              [python_infer, "scripts/run_qwen3_instruct_inference.py",
               "--model_path", str(adapter), "--input", model_input,
               "--output", raw_csv],
              EVAL_DIR, stage_env(Path(os.environ.get("QASRL_BASE_DIR", REPO_ROOT / "runs"))),
              log_dir / "04_inference.log", args.dry_run)

    # Step 2 - fill dummy slot columns (unlabelled metric only)
    run_stage("dummy_slots",
              [python_score, "scripts/add_dummy_slots.py", raw_csv, filled_csv],
              EVAL_DIR, os.environ.copy(),
              log_dir / "05_dummy_slots.log", args.dry_run)

    # Step 3 - score against the shipped gold (never modified)
    run_stage("evaluation",
              [python_score, "scripts/evaluate_dataset.py", filled_csv, gold],
              EVAL_DIR, os.environ.copy(),
              log_dir / "06_evaluation.log", args.dry_run)

    return EVAL_DIR / filled_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run SFT -> RL -> inference -> evaluation end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  python run_full_pipeline.py --sft_data TRAIN --rl_method GRPO")
    p.add_argument("--sft_data", required=True, choices=["TRAIN", "DEV"],
                   help="SFT training split. DEV is the headline baseline; "
                        "TRAIN is the full-train-split comparison row.")
    p.add_argument("--rl_method", required=True, choices=["GRPO", "DPO"],
                   help="RL stage to run after SFT.")
    p.add_argument("--tag", default=None,
                   help="Label for this run's inference outputs "
                        "(default: <rl_method>_<sft_data>_<timestamp>).")
    p.add_argument("--base_dir", default=None,
                   help="Storage root for adapters/checkpoints/logs "
                        "(QASRL_BASE_DIR; default <repo>/runs).")
    p.add_argument("--sft_adapter", default=None,
                   help="Skip SFT and warm-start the RL stage from this adapter.")
    p.add_argument("--rl_adapter", default=None,
                   help="Skip SFT and RL and evaluate this adapter directly.")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--select", dest="select", action="store_true", default=True,
                     help="Rank RL checkpoints with eval_on_val.py and evaluate the "
                          "winner. ON BY DEFAULT -- this is how the reported numbers "
                          "were produced.")
    sel.add_argument("--no_select", dest="select", action="store_false",
                     help="Skip selection and evaluate the final RL adapter, which is "
                          "generally NOT the best checkpoint.")
    p.add_argument("--rl_checkpoint", default=None,
                   help="Evaluate this checkpoint instead of the final adapter: a "
                        "step number (e.g. 3600) or a path.")
    p.add_argument("--python_train", default=None,
                   help="Interpreter for the training/inference env (default: the "
                        "conda env named in the configs).")
    p.add_argument("--python_eval", default=None,
                   help="Interpreter for the CPU scoring env.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print each stage's command without running it.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.tag is None:
        args.tag = (f"{args.rl_method.lower()}_{args.sft_data.lower()}_"
                    f"{_dt.datetime.now():%Y%m%d_%H%M%S}")

    base_dir = Path(args.base_dir).resolve() if args.base_dir else REPO_ROOT / "runs"
    log_dir = base_dir / "pipeline_logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "00_pipeline.log", encoding="utf-8")])

    sft_cfg = load_config(TRAINING_DIR / "sft" / "config.yaml")
    eval_cfg = load_config(EVAL_DIR / "config.yaml")
    python_train = resolve_python(sft_cfg.get("conda_env", "train_qwen3"), args.python_train)
    python_score = resolve_python(eval_cfg.get("envs", {}).get("scoring", "eval"),
                                  args.python_eval)
    python_infer = resolve_python(eval_cfg.get("envs", {}).get("inference", "train_qwen3"),
                                  args.python_train)

    log.info("=" * 72)
    log.info("QA-SRL full pipeline | sft_data=%s rl_method=%s tag=%s",
             args.sft_data, args.rl_method, args.tag)
    log.info("base_dir=%s", base_dir)
    log.info("logs=%s", log_dir)
    log.info("train/inference python=%s", python_train)
    log.info("scoring python=%s", python_score)
    log.info("=" * 72)

    started = _dt.datetime.now()
    try:
        # ---- SFT ------------------------------------------------------
        if args.rl_adapter:
            rl_adapter = Path(args.rl_adapter)
            log.info("[sft] skipped (--rl_adapter given)")
            log.info("[%s] skipped (--rl_adapter given)", args.rl_method.lower())
        else:
            if args.sft_adapter:
                sft_adapter = Path(args.sft_adapter)
                log.info("[sft] skipped; using %s", sft_adapter)
                if not args.dry_run and not is_adapter_dir(sft_adapter):
                    raise StageError("sft", f"--sft_adapter has no adapter_config.json: "
                                            f"{sft_adapter}")
            else:
                sft_adapter = run_sft(args, base_dir, python_train, log_dir)

            # ---- RL ---------------------------------------------------
            rl_adapter = run_rl(args, base_dir, python_train, sft_adapter, log_dir)

        # ---- checkpoint choice ----------------------------------------
        if args.rl_checkpoint:
            candidate = Path(args.rl_checkpoint)
            if not candidate.is_absolute() and not candidate.exists():
                candidate = checkpoint_parent(rl_adapter) / f"checkpoint-{args.rl_checkpoint}"
            if not args.dry_run and not is_adapter_dir(candidate):
                raise StageError("select", f"--rl_checkpoint does not resolve to an "
                                           f"adapter directory: {candidate}")
            log.info("[select] using requested checkpoint: %s", candidate)
            eval_adapter = candidate
        elif args.select:
            eval_adapter = select_checkpoint(args, base_dir, python_train,
                                             rl_adapter, log_dir)
        else:
            eval_adapter = rl_adapter
            log.warning("--no_select: evaluating the FINAL %s adapter. The reported "
                        "results come from a mid-run checkpoint (GRPO ~3600, DPO 150), "
                        "so this number is not comparable to them.", args.rl_method)

        # ---- inference + evaluation -----------------------------------
        filled = run_inference_and_eval(args, python_infer, python_score,
                                        eval_adapter, log_dir)

    except StageError as exc:
        log.error("=" * 72)
        log.error("PIPELINE FAILED at stage: %s", exc.stage.upper())
        log.error("%s", exc)
        log.error("stage logs: %s", log_dir)
        log.error("=" * 72)
        return 1
    except KeyboardInterrupt:
        log.error("interrupted by user; partial outputs are under %s", log_dir)
        return 130

    log.info("=" * 72)
    log.info("PIPELINE COMPLETE in %s", _dt.datetime.now() - started)
    log.info("evaluated adapter : %s", eval_adapter)
    log.info("scored predictions: %s", filled)
    log.info("stage logs        : %s", log_dir)
    log.info("Unlabelled Argument F1 is printed above by evaluate_dataset.py "
             "(also in %s).", log_dir / "06_evaluation.log")
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
