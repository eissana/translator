import torch
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score

from preprocess import Preprocessor


def get_loss(logits, y, ignore_index):
    """
    Computes cross-entropy loss, given logits and labels.
    """
    B, T, C = logits.shape
    # F.cross_entropy expects size C, (B, C), or (B, C, ...)
    # logits shape is (B, T, C), so we flatten the first two dimensions.
    return F.cross_entropy(
        logits.view(B * T, C), y.reshape(B * T), ignore_index=ignore_index
    )
    # criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_index)
    # return criterion(logits.view(B*T, C), y.reshape(B*T))


def text2tokens(text, tokenizer):
    tokens = [Preprocessor.INIT_TOKEN]
    tokens.extend([t.text.lower() for t in tokenizer(text)])
    tokens.append(Preprocessor.EOS_TOKEN)
    return tokens


def src2target(tokens, model, preprocessor, block_size, device):
    """
    Gets source language tokens, calls model to translate them, and returns
    target tokens.
    """
    token_ids = [preprocessor.src_field.vocab.stoi[token] for token in tokens]

    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    sos = preprocessor.tgt_field.vocab.stoi[Preprocessor.INIT_TOKEN]
    eos = preprocessor.tgt_field.vocab.stoi[Preprocessor.EOS_TOKEN]

    y = torch.tensor([[sos]], dtype=torch.long, device=device)

    for _ in range(block_size):
        with torch.no_grad():
            logits = model(x, y)

        logits = logits[:, -1, :]

        scores = F.softmax(logits, dim=-1)
        next_token = scores.multinomial(1)
        # next_token = logits.argmax(dim=-1).unsqueeze(0)

        if next_token.item() == eos:
            break

        y = torch.cat((y, next_token), dim=-1)

    y = y.view(-1)
    y = [preprocessor.tgt_field.vocab.itos[t] for t in y]

    return y[1:]


def translate(text, model, preprocessor, block_size, device):
    src_tokens = text2tokens(text, preprocessor.src_spacy.tokenizer)
    tgt_tokens = src2target(src_tokens, model, preprocessor, block_size, device)
    return " ".join(tgt_tokens)


def bleu(data, model, preprocessor, block_size, device):
    targets = []
    outputs = []

    for example in data:
        src = example.src
        trg = example.trg
        prediction = src2target(src, model, preprocessor, block_size, device)

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)
