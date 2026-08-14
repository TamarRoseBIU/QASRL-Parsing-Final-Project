import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# now you can use absolute imports

import torch
from torch.nn.functional import cross_entropy as CrossEntropyLoss, pad
from transformers import Seq2SeqTrainer

from .computeQASetValidationMetrics import *
from .defaultValues import *


# subclass trainer
class QASetLossTrainer(Seq2SeqTrainer):

    def __init__(self, lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2, qa_sep=DEFAULT_QA_SEP_TOKENS,
                 q_sep=DEFAULT_Q_SEP_TOKENS, a_sep=DEFAULT_A_SEP_TOKENS, padding_idx=DEFAULT_PADDING_IDX, use_device=DEFAULT_DEVICE, *args, **kwargs):

        # else assign default value
        if 'lambda1' in kwargs:
            self.LAMBDA1 = kwargs['lambda1']
        else:
            self.LAMBDA1 = lambda1 if lambda1 is not None else DEFAULT_LAMBDA1

        if 'lambda2' in kwargs:
            self.LAMBDA2 = kwargs['lambda2']
            del kwargs['lambda2']
        else:
            self.LAMBDA2 = lambda2 if lambda2 is not None else DEFAULT_LAMBDA2

        if 'qa_sep' in kwargs:
            self.QA_sep_tokens_tensor = kwargs['qa_sep']
            del kwargs['qa_sep']
        else:
            self.QA_sep_tokens_tensor = qa_sep if qa_sep is not None else DEFAULT_QA_SEP_TOKENS

        self.QA_sep_length = len(self.QA_sep_tokens_tensor)

        if 'a_sep' in kwargs:
            self.A_sep_tokens_tensor = kwargs['a_sep']
            del kwargs['a_sep']
        else:
            self.A_sep_tokens_tensor = a_sep if a_sep is not None else DEFAULT_A_SEP_TOKENS

        self.A_sep_length = len(self.A_sep_tokens_tensor)

        if 'q_sep' in kwargs:
            self.Q_sep_tokens_tensor = kwargs['q_sep']
            del kwargs['q_sep']
        else:
            self.Q_sep_tokens_tensor = q_sep if qa_sep is not None else DEFAULT_Q_SEP_TOKENS

        self.Q_sep_length = len(self.Q_sep_tokens_tensor)

        if 'padding_idx' in kwargs:
            self.PADDING_IDX = kwargs['padding_idx']
            del kwargs['padding_idx']
        else:
            self.PADDING_IDX = padding_idx if padding_idx is not None else DEFAULT_PADDING_IDX

        if 'use_device' in kwargs:
            self.DEVICE = kwargs['use_device']
            del kwargs['use_device']
        else:
            self.DEVICE = use_device if use_device is not None else DEFAULT_DEVICE
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, num_items_in_batch=16, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if not hasattr(self, '_loss_log_step'):
            self._loss_log_step = 0

        # --- Causal LM shift ---
        # logits[t] predicts token at position t+1, so align:
        #   shift_logits[:, t] <-> shift_labels[:, t] = labels[:, t+1]
        # Without this, logits[t] is compared to labels[t] (off by 1):
        # the first answer token's loss hits a -100 (ignored), and QA boundary
        # parsing in custom_loss_function sees misaligned IDs -> returns 0.
        shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()        # [B, L-1]

        # ------------------------------------------------------------------ #
        #  DEBUG BLOCK — first 3 steps + every 50 steps                      #
        # ------------------------------------------------------------------ #
        if self._loss_log_step < 3 or self._loss_log_step % 50 == 0:
            print(f"\n{'='*60}")
            print(f"[DEBUG step={self._loss_log_step}]")
            print(f"  QA_sep token IDs : {self.QA_sep_tokens_tensor.tolist()}")
            print(f"  Q_sep  token IDs : {self.Q_sep_tokens_tensor.tolist()}")
            print(f"  A_sep  token IDs : {self.A_sep_tokens_tensor.tolist()}")

            row0 = shift_labels[0]
            answer_tokens0 = row0[row0 != -100].cpu().tolist()
            print(f"  Batch[0] answer token count : {len(answer_tokens0)}")
            print(f"  Batch[0] answer token IDs   : {answer_tokens0[:60]}{'...' if len(answer_tokens0)>60 else ''}")
            try:
                tok = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
                if tok:
                    print(f"  Batch[0] answer decoded     : {repr(tok.decode(answer_tokens0, skip_special_tokens=False)[:200])}")
            except Exception as e:
                print(f"  (Could not decode: {e})")

            qa_sep_cpu = self.QA_sep_tokens_tensor.cpu()
            q_sep_cpu  = self.Q_sep_tokens_tensor.cpu()
            a_sep_cpu  = self.A_sep_tokens_tensor.cpu()
            t0 = torch.tensor(answer_tokens0)
            print(f"  find_subsequence <QA> hits : {find_subsequence_occurrences(t0, qa_sep_cpu)}")
            print(f"  find_subsequence ?    hits : {find_subsequence_occurrences(t0, q_sep_cpu)}")
            print(f"  find_subsequence <A>  hits : {find_subsequence_occurrences(t0, a_sep_cpu)}")
            print(f"  (find_subsequence now uses int64 unfold — no float32 precision issues)")
            print('='*60)
        # ------------------------------------------------------------------ #

        try:
            custom_loss = self.custom_loss_function(shift_logits, shift_labels)
        except Exception as e:
            import traceback
            print("\n" + "="*60)
            print("ERROR inside custom_loss_function:")
            traceback.print_exc()
            print("="*60 + "\n")
            custom_loss = torch.tensor(0.0, device=self.DEVICE, requires_grad=True)

        ce_loss = CrossEntropyLoss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        loss = self.LAMBDA1 * ce_loss + self.LAMBDA2 * custom_loss

        if self._loss_log_step % 10 == 0:
            print(f"  [step {self._loss_log_step}] ce_loss={ce_loss.item():.4f}  "
                  f"custom_loss={custom_loss.item():.4f}  total={loss.item():.4f}")
        self._loss_log_step += 1

        return (loss, outputs) if return_outputs else loss

    def custom_loss_function(self, logits, targets):

        # --- Causal LM + CUDA fix ---
        # For Qwen3 (causal LM), labels are [-100, -100, ..., real_tokens].
        # We strip -100 prompt positions so only answer tokens are processed.
        #
        # CRITICAL: clean_targets and clean_predictions must be moved to CPU.
        # The original code was written for T5 where all tensors were on CPU.
        # With Qwen3 and device_map="auto", predictions are CUDA tensors.
        # find_subsequence_occurrences uses conv1d: with CUDA float32 and Qwen3
        # token IDs up to 151,936, the squared dot-product comparison loses
        # float32 precision (only ~7 significant digits), producing NaN/Inf
        # that triggers SIGFPE in the CUDA kernel -- uncatchable by Python.
        # clean_logits stays on GPU since it feeds the final CrossEntropyLoss.
        predictions = torch.max(logits, dim=-1).indices
        clean_targets = []
        clean_predictions = []
        clean_logits = []
        for label_row, pred_row, logit_row in zip(targets, predictions, logits):
            answer_mask = label_row != -100
            clean_targets.append(label_row[answer_mask].cpu())      # CPU for parsing
            clean_predictions.append(pred_row[answer_mask].cpu())   # CPU for parsing
            clean_logits.append(logit_row[answer_mask])             # GPU for loss

        # Move separator tensors to CPU to match the CPU token sequences above
        qa_sep_cpu = self.QA_sep_tokens_tensor.cpu()
        q_sep_cpu  = self.Q_sep_tokens_tensor.cpu()
        a_sep_cpu  = self.A_sep_tokens_tensor.cpu()

        pred_split  = [split_QAs(pred,  qa_sep_cpu, q_sep_cpu, a_sep_cpu) for pred  in clean_predictions]
        label_split = [split_QAs(label, qa_sep_cpu, q_sep_cpu, a_sep_cpu) for label in clean_targets]

        qa_start_indices   = [find_subsequence_occurrences(pred,  qa_sep_cpu) for pred  in clean_predictions]
        gold_start_indices = [find_subsequence_occurrences(label, qa_sep_cpu) for label in clean_targets]

        total_gold, total_pred = 0, 0
        pred_stack, gold_stack = [], []
        for b, (label_qas, pred_qas) in enumerate(zip(label_split, pred_split)):
            label_answers = []
            pred_answers  = []
            label_ans_ind, pred_ans_ind = 0, 0
            pred_ans_ind_to_qa  = {}
            label_ans_ind_to_qa = {}

            for qa_i, (question, answers) in enumerate(label_qas):
                label_answers.extend(answers)
                # Pass CPU target and CPU sep tensors; returned tensors are CPU
                q, answers = self.get_question_and_answer_from_indices(
                    gold_start_indices[b], clean_targets[b], qa_i,
                    qa_sep=qa_sep_cpu, q_sep=q_sep_cpu, a_sep=a_sep_cpu)
                for ans in answers:
                    label_ans_ind_to_qa[label_ans_ind] = torch.cat((q, ans))
                    label_ans_ind += 1

            for i, (question, answers) in enumerate(pred_qas):
                pred_answers.extend(answers)
                # clean_logits[b] is GPU; clean_predictions[b] is CPU for index finding
                q, answers = self.get_question_and_answer_from_indices(
                    qa_start_indices[b], clean_logits[b], i,
                    is_logits=True, logits_pred=clean_predictions[b],
                    qa_sep=qa_sep_cpu, q_sep=q_sep_cpu, a_sep=a_sep_cpu)
                for ans in answers:
                    pred_ans_ind_to_qa[pred_ans_ind] = torch.cat((q, ans))
                    pred_ans_ind += 1

            total_gold += label_ans_ind
            total_pred += pred_ans_ind

            pred_answers, label_answers = pad_lists_with_empty(pred_answers, label_answers)

            # Guard: linear_sum_assignment on a (0x0) matrix causes SIGFPE in scipy's
            # C extension -- also uncatchable by Python try/except.
            if len(pred_answers) == 0 and len(label_answers) == 0:
                continue

            IOU_matrix = build_IOU_metrix(pred_answers, label_answers)
            row_ind, col_ind = linear_sum_assignment(IOU_matrix, maximize=True)

            for r, c in zip(row_ind, col_ind):
                # Compute loss per matched pair immediately on CPU rather than stacking.
                # Stacking logit slices into pred_stack was fine for T5 (vocab ~32k),
                # but Qwen3's vocab of 151,936 makes each answer slice ~6MB in bfloat16.
                # Accumulating a batch worth of those and moving to GPU causes OOM.
                if r in pred_ans_ind_to_qa and c in label_ans_ind_to_qa:
                    pred_t = pred_ans_ind_to_qa[r].cpu().float()   # [seq, vocab]
                    gold_t = label_ans_ind_to_qa[c].cpu().long()   # [seq]
                    # Align lengths by truncating to the shorter of the two
                    min_len = min(pred_t.size(0), gold_t.size(0))
                    pred_stack.append(pred_t[:min_len])
                    gold_stack.append(gold_t[:min_len])
                elif r in pred_ans_ind_to_qa:
                    pred_t = pred_ans_ind_to_qa[r].cpu().float()
                    gold_t = torch.full((pred_t.size(0),), self.PADDING_IDX, dtype=torch.long)
                    pred_stack.append(pred_t)
                    gold_stack.append(gold_t)
                elif c in label_ans_ind_to_qa:
                    gold_t = label_ans_ind_to_qa[c].cpu().long()
                    pred_t = torch.zeros(gold_t.size(0), vocab_size, dtype=torch.float)
                    pred_stack.append(pred_t)
                    gold_stack.append(gold_t)

        if len(gold_stack) == 0 or len(pred_stack) == 0:
            print("  [QASetLoss] WARNING: gold_stack or pred_stack is empty — "
                  "custom_loss returning 0. Check that <QA>/<A>/? separators "
                  "appear in the answer tokens after the -100 mask is removed.")
            return torch.tensor(0.0, device=self.DEVICE, requires_grad=True)

        # Concatenate on CPU (avoids the giant VRAM allocation)
        all_pred = torch.cat([p.view(-1, vocab_size) for p in pred_stack], dim=0)  # [N, vocab]
        all_gold = torch.cat([g.view(-1)             for g in gold_stack], dim=0)  # [N]

        # CrossEntropyLoss on CPU, then move scalar result to GPU for backward
        loss = CrossEntropyLoss(all_pred, all_gold, ignore_index=-100)
        return loss.to(self.DEVICE)

    def get_question_and_answer_from_indices(self, QA_start_indices, original_values, QA_index, is_logits=False,
                                             logits_pred=None, qa_sep=None, q_sep=None, a_sep=None):
        # Allow callers to pass CPU versions of sep tensors (needed for Qwen3/CUDA)
        qa_sep_t = qa_sep if qa_sep is not None else self.QA_sep_tokens_tensor
        q_sep_t  = q_sep  if q_sep  is not None else self.Q_sep_tokens_tensor
        a_sep_t  = a_sep  if a_sep  is not None else self.A_sep_tokens_tensor

        QA_indices = self.get_QA_indices(QA_start_indices, QA_index, len(original_values))

        if len(QA_indices) == 0:
            return torch.tensor([], device=original_values.device, dtype=original_values.dtype), []

        # For is_logits=True: logits_pred is CPU token IDs used only for index finding;
        # original_values is the GPU logits tensor used for the returned slices.
        if is_logits:
            if logits_pred is None:
                raise ValueError("logits_pred must be provided if is_logits is True")
            values = logits_pred[QA_indices]   # CPU token IDs for subsequence search
        else:
            values = original_values[QA_indices]

        # Find the question mark index (values is CPU here, q_sep_t is CPU)
        qm_index = find_subsequence_occurrences(values, q_sep_t)
        # If no question mark is found, return the full tensor as question
        if not qm_index:
            return torch.tensor(values, device=self.DEVICE, dtype=original_values.dtype), []

        # Compute question indices
        start_index = 0 if QA_index == 0 else self.QA_sep_length
        question_indices = QA_indices[start_index:qm_index[0] + 1]  # +1 to include the question mark in the question

        # Find answer start indices
        ans_start_indices = find_subsequence_occurrences(values, a_sep_t)
        ans_start_indices = [qm_index[0] + 1] + ans_start_indices  # Adjust to start after question

        # Compute answer indices using list comprehension (optimized slicing)
        answer_indices = []
        ans_indices_length = len(ans_start_indices)
        for ans_index in range(ans_indices_length):
            if ans_index >= ans_indices_length:
                answer_indices.append([])
                print(
                    f"Unexpected scenario in get_question_and_answer_from_indices when computing answer indices. Received index = {ans_index} when ans_start_indices are:\n{ans_start_indices}")
                continue

            start_index = ans_start_indices[ans_index]
            if ans_index > 0:
                start_index += self.A_sep_length

            if ans_index < ans_indices_length - 1:
                answer_indices.append(QA_indices[start_index:ans_start_indices[ans_index + 1]])
            else:  # ans_index == ans_indices_length - 1
                answer_indices.append(QA_indices[start_index:])

        question_ret = original_values[question_indices]
        answer_ret = [original_values[indices] for indices in answer_indices]

        return question_ret, answer_ret

    @staticmethod
    def get_QA_indices(start_indices, index, max_len, start_index=0):
        if index < 0 or index > len(start_indices):
            return range(0)  # Empty range
        if index == 0:
            end_index = start_indices[0] if start_indices else max_len
            return range(start_index, end_index)
        if index == len(start_indices):
            return range(start_indices[-1], max_len)
        return range(start_indices[index - 1], start_indices[index])


# Add at the bottom of QASetLossTrainer.py
from transformers import Trainer

class QASetLossTrainerCausalLM(Trainer):

    def __init__(self, lambda1=None, lambda2=None, qa_sep=None,
                 q_sep=None, a_sep=None, padding_idx=None, use_device=None, *args, **kwargs):
        
        self.LAMBDA1 = lambda1 if lambda1 is not None else DEFAULT_LAMBDA1
        self.LAMBDA2 = lambda2 if lambda2 is not None else DEFAULT_LAMBDA2
        self.QA_sep_tokens_tensor = qa_sep if qa_sep is not None else DEFAULT_QA_SEP_TOKENS
        self.QA_sep_length = len(self.QA_sep_tokens_tensor)
        self.A_sep_tokens_tensor = a_sep if a_sep is not None else DEFAULT_A_SEP_TOKENS
        self.A_sep_length = len(self.A_sep_tokens_tensor)
        self.Q_sep_tokens_tensor = q_sep if q_sep is not None else DEFAULT_Q_SEP_TOKENS
        self.Q_sep_length = len(self.Q_sep_tokens_tensor)
        self.PADDING_IDX = padding_idx if padding_idx is not None else DEFAULT_PADDING_IDX
        self.DEVICE = use_device if use_device is not None else DEFAULT_DEVICE

        super().__init__(*args, **kwargs)  # calls Trainer.__init__ cleanly

    # Reuse all the logic from the updated CausalLM-aware version of these methods
    compute_loss                         = QASetLossTrainer.compute_loss
    custom_loss_function                 = QASetLossTrainer.custom_loss_function
    get_question_and_answer_from_indices = QASetLossTrainer.get_question_and_answer_from_indices
    get_QA_indices                       = QASetLossTrainer.get_QA_indices