# SFT LoRA adapters

Two adapters from the SFT stage, 37 MB each. They differ only in **how the best
checkpoint was chosen** — same base model, same recipe (LoRA r=8, α=16, dropout 0.05,
LR 2e-4, 5 epochs), same `dev` training data.

| Directory | Selection | Unlab. Arg F1 | Role |
| --------- | --------- | :-----------: | ---- |
| `sft_dev_selected_on_test/` | trained on all of `dev` (2,406 ex.), checkpoint selected on `test` | **78.90** | the reported SFT baseline, and the checkpoint the reported GRPO and DPO runs warm-start from. **The default warm start.** |
| `sft_dev_val_heldout/` | trained on 90% of `dev`, checkpoint selected on a held-out `dev_val` slice (grouped by `sentence_id`, seed 42) → checkpoint-600 | 79.42 | a later validation check, kept for reference. It confirms the original test-based selection did not inflate the baseline (79.42 > 78.90). **Not part of the reported RL chain.** |

Each directory is self-contained: the LoRA weights plus the tokenizer the stages load
from it. The base model, `Qwen/Qwen3-30B-A3B-Instruct-2507`, is resolved separately.

## Using them

GRPO and DPO pick `sft_dev_selected_on_test/` automatically. To use a different adapter:

```bash
export QASRL_SFT_ADAPTER=$PWD/training/sft/adapters/sft_dev_val_heldout
```

To score one directly, see `evaluation/config.yaml`. The predictions behind both numbers
already ship, so the F1s above re-derive on a CPU without running inference:

```bash
conda activate eval
cd evaluation
python scripts/evaluate_dataset.py \
    data/model_output_filled_slots/Qwen3-30B-A3B-Instruct-2507/passive_red_output_SFT_dev_selected_on_test_filled_slots.csv \
    data/gold/gold_updated_passive_filled_slots.csv
```

## Reproducing them

`training/sft/Stage_CE_Instruct_DEV.py` writes `sft_dev_selected_on_test` by default and
`sft_dev_val_heldout` with `QASRL_SFT_SELECT_ON=dev_val`. See `../config.yaml`.
