---
base_model: Qwen/Qwen3-30B-A3B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-30B-A3B-Instruct-2507
- lora
- transformers
license: mit
---

# QA-SRL SFT adapter — `sft_dev_val_heldout`

LoRA adapter over **Qwen/Qwen3-30B-A3B-Instruct-2507** for QA-SRL parsing: given a
sentence and a predicate, generate the question–answer pairs describing that predicate's
arguments.

- **Trained on:** 90% of `dev`, best checkpoint (600) chosen on a held-out 10% slice of `dev`; `test` never loaded
- **Unlabelled Argument F1** on the `passive_red` test split: **79.42**
- **Role in the project:** a later validation check confirming that the original test-based selection did not inflate the baseline. Not part of the reported RL chain
- **LoRA:** r=8, alpha=16, dropout=0.05 on q/k/v/o/gate/up/down projections; LR 2e-4, 5 epochs

Part of [QASRL-Parsing-Final-Project](https://github.com/TamarRoseBIU/QASRL-Parsing-Final-Project);
see `training/sft/adapters/README.md` there for both adapters and how to use them.
