import numpy as np
import torch
from torch.nn.functional import pad
from scipy.optimize import linear_sum_assignment
from .defaultValues import *


def compute_match_result(trg_str_list, pred_str_list, match_type='exact'):
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)

    is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
    for trg_idx, trg_word_list in enumerate(trg_str_list):  # for each target answer
        joined_trg_word_list = ' '.join(str(x) for x in trg_word_list) if isinstance(trg_word_list,
                                                                                     (list, np.ndarray)) else str(
            trg_word_list)  # concatenate the answer tokens indexes
        for pred_idx, pred_word_list in enumerate(pred_str_list):  # for each prediction answer
            joined_pred_word_list = ' '.join(str(x) for x in pred_word_list) if isinstance(pred_word_list,
                                                                                           (list, np.ndarray)) else str(
                pred_word_list)  # concatenate the answer tokens indexes
            if match_type == 'exact':
                if joined_pred_word_list == joined_trg_word_list:
                    is_match[trg_idx][pred_idx] = True
            elif match_type == 'sub':
                if joined_pred_word_list in joined_trg_word_list:
                    is_match[trg_idx][pred_idx] = True

            # if we already found a match for the target answer, we don't need to check the rest of the predictions
            # because we don't want to count twice the same label if two predictions are the same
            if is_match[trg_idx][pred_idx]:
                break

    return is_match


def IOU(list1, list2):
    # input: two lists of answers, each answer is a list of indexes of the tokens in the answer
    # output: the IOU score of the two lists

    is_match_matrix = compute_match_result(list1, list2)
    intersection = np.sum(is_match_matrix)
    union = len(list1) + len(list2) - intersection

    return intersection / union if union > 0 else 0


def find_subsequence_occurrences(tokenized_text, sub_sequence):
    """
    Find all start indices of sub_sequence in tokenized_text.

    Uses an integer sliding-window (torch.unfold + exact int64 equality),
    NOT conv1d. The original conv1d approach computed a float32 dot-product
    for comparison, which silently loses precision for large token IDs:
    float32 can represent integers exactly only up to 2^24 (~16.7M), but
    Qwen3's vocab has IDs up to 151,936 giving dot products ~2.3e9, well
    beyond that limit. The result was that separator tokens were never found,
    making custom_loss always return 0 regardless of the input.

    Parameters:
    - tokenized_text (torch.Tensor | np.ndarray | list): The tokenized text.
    - sub_sequence (torch.Tensor): The sub-sequence to find.

    Returns:
    - List[int]: List of starting indices where sub_sequence is found.
    """
    if isinstance(tokenized_text, (np.ndarray, list)):
        tokenized_text = torch.tensor(tokenized_text)

    # Work entirely in int64 on CPU - no float arithmetic involved
    tokenized_text = tokenized_text.cpu().long()
    sub_sequence   = sub_sequence.cpu().long()

    text_len = tokenized_text.size(0)
    sub_len  = sub_sequence.size(0)

    if sub_len == 0 or text_len < sub_len:
        return []

    # unfold produces shape [text_len - sub_len + 1, sub_len]
    # where windows[i] == tokenized_text[i : i + sub_len]
    windows = tokenized_text.unfold(0, sub_len, 1)               # [N, sub_len]
    matches = (windows == sub_sequence.unsqueeze(0)).all(dim=1)  # [N] bool
    return matches.nonzero(as_tuple=True)[0].tolist()


def pad_lists_with_empty(pred_split, label_split):
    if len(pred_split) > len(label_split):
        label_split += [[]] * (len(pred_split) - len(label_split))
    elif len(pred_split) < len(label_split):
        pred_split += [[]] * (len(label_split) - len(pred_split))

    return pred_split, label_split


def build_IOU_metrix(pred_split, label_split):
    max_len = max(len(pred_split), len(label_split))
    # we want the matrix to be NxN, so we will pad the shorter list with zeros
    # (like adding empty elements to the shorter list, like mentioned in the article)
    IOU_matrix = np.zeros((max_len, max_len))
    for i, single_answer in enumerate(pred_split):
        for j, label_answer in enumerate(label_split):
            IOU_matrix[i, j] = IOU(single_answer, label_answer)

    return IOU_matrix


def split_QAs(text,
              qa_sep_tokens_tensor=DEFAULT_QA_SEP_TOKENS,
              q_sep_tokens_tensor=DEFAULT_Q_SEP_TOKENS,
              a_sep_tokens_tensor=DEFAULT_A_SEP_TOKENS):
    # split the text to be list of answers.
    # each answer is a list of indexes of the tokens in the answer
    # we will split the text first by <QA> token, then by ? token and lastly by the <A> token
    # input: text - np array of indexes of the tokens in the text
    # QA_sep_tokens_tensor - tensor of indexes of the tokens that separate between questions and answers
    # q_sep_tokens_tensor - tensor of indexes of the tokens ? that separate between the question and it's answers
    # A_sep_tokens_tensor - tensor of indexes of the tokens that separate between answers
    # output: list of tuples, each tuple is a question and a list of answers

    qa_split = split_tokenized_text(text, qa_sep_tokens_tensor)
    questions_and_answers = []
    for qa in qa_split:
        q_split = split_tokenized_text(qa, q_sep_tokens_tensor,
                                       include_sub_sequence=True)  # split the question from the answers. should be only one question
        if len(q_split) == 1:
            question, answers = q_split[0], []
        else:
            question, answers = q_split[0], q_split[1]
        answers = split_tokenized_text(answers, a_sep_tokens_tensor)
        questions_and_answers.append((question, answers))

    return questions_and_answers


def split_tokenized_text(tokenized_text, sub_sequence, include_sub_sequence=False):
    """
    Split tokenized_text into chunks separated by sub_sequence.

    Parameters:
    - tokenized_text (list of int): The tokenized text.
    - sub_sequence (list of int): The sub-sequence to split around.

    Returns:
    - list of list of int: A list of sub-sequences from the original text.
    """

    start_indices = find_subsequence_occurrences(tokenized_text, sub_sequence)
    if len(start_indices) == 0:
        return [tokenized_text]

    sub_seqs = []
    last_index = 0

    for start_index in start_indices:
        if last_index != start_index:
            if include_sub_sequence:
                sub_seqs.append(tokenized_text[last_index:start_index + len(sub_sequence)])
            else:
                sub_seqs.append(tokenized_text[last_index:start_index])
        last_index = start_index + len(sub_sequence)

    # Append the remaining part after the last occurrence of the sub-sequence
    if last_index < len(tokenized_text):
        sub_seqs.append(tokenized_text[last_index:])

    return sub_seqs


def calculate_metrics(tp, fp, fn):
    accuracy = tp / (tp + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, recall, precision, f1


def compute_metrics(eval_pred):
    global DEFAULT_QA_SEP_TOKENS, DEFAULT_Q_SEP_TOKENS, DEFAULT_A_SEP_TOKENS

    predictions, labels = eval_pred

    pred_split = [split_QAs(prediction, DEFAULT_QA_SEP_TOKENS, DEFAULT_Q_SEP_TOKENS, DEFAULT_A_SEP_TOKENS) for prediction in predictions]  # split the predictions to questions and answers
    label_split = [split_QAs(label, DEFAULT_QA_SEP_TOKENS, DEFAULT_Q_SEP_TOKENS, DEFAULT_A_SEP_TOKENS) for label in labels]  # split the labels to questions and answers

    total_tp, total_fp, total_fn = 0, 0, 0
    # for each example, calculate the IOU score between the predictions and the labels
    for i, (label_qas, pred_qas) in enumerate(
            zip(label_split, pred_split)):  # for each sentence & predicat we'll take the target and predicted qa

        label_answers = []
        pred_answers = []
        for question, answers in label_qas:
            label_answers.extend(answers)
        for question, answers in pred_qas:
            pred_answers.extend(answers)

        original_label_answers_len = len(label_answers)  # number of label qas for the same sentence and predict
        original_pred_answers_len = len(pred_answers)  # number of pred qas the same sentence and predict

        # add empty rows or columns to the splits if needed
        pred_answers, label_answers = pad_lists_with_empty(pred_answers, label_answers)
        IOU_matrix = build_IOU_metrix(pred_answers,
                                      label_answers)  # matrix to compare between groups of answers; if one group is larger it's like adding more empty lists
        try:

            row_ind, col_ind = linear_sum_assignment(IOU_matrix, maximize=True)

            tp = (IOU_matrix[row_ind, col_ind] >= 0.5).sum()
            fp = original_pred_answers_len - tp
            fn = original_label_answers_len - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn
        except ValueError:
            print(IOU_matrix)
            print(pred_answers)
            print(label_answers)

    accuracy, recall, precision, f1 = calculate_metrics(total_tp, total_fp, total_fn)

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}