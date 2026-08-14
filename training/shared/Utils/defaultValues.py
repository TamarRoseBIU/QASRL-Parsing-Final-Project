import torch

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LAMBDA1 = 0.7
DEFAULT_LAMBDA2 = 0.3
DEFAULT_QA_SEP_TOKENS = torch.tensor([2, 23008, 3155]).to(DEFAULT_DEVICE)
DEFAULT_Q_SEP_TOKENS = torch.tensor([58]).to(DEFAULT_DEVICE)
DEFAULT_A_SEP_TOKENS = torch.tensor([2, 188, 3155]).to(DEFAULT_DEVICE)
DEFAULT_PADDING_IDX = 0

def set_global_default_values(tokenizer):
    global DEFAULT_DEVICE, DEFAULT_QA_SEP_TOKENS, DEFAULT_Q_SEP_TOKENS, DEFAULT_A_SEP_TOKENS, DEFAULT_PADDING_IDX

    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QA_sep_tokens = tokenizer(' <QA>')['input_ids'][1:-1]  # without the start and end of line flags
    print(f'QA_sep_tokens = {QA_sep_tokens}')
    DEFAULT_QA_SEP_TOKENS = torch.tensor(QA_sep_tokens).to(DEFAULT_DEVICE)
    print(DEFAULT_QA_SEP_TOKENS)

    A_sep_tokens = tokenizer(' <A>')['input_ids'][1:-1]
    print(f'A_sep_tokens = {A_sep_tokens}')
    DEFAULT_A_SEP_TOKENS = torch.tensor(A_sep_tokens).to(DEFAULT_DEVICE)
    print(DEFAULT_A_SEP_TOKENS)

    q_sep_tokens = tokenizer('?')['input_ids'][1:-1]
    print(f'q_sep_tokens = {q_sep_tokens}')
    DEFAULT_Q_SEP_TOKENS = torch.tensor(q_sep_tokens).to(DEFAULT_DEVICE)
    print(DEFAULT_Q_SEP_TOKENS)

    DEFAULT_PADDING_IDX = tokenizer.pad_token_id