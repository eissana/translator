import torch
import torch.nn as nn
from torchtext.data import BucketIterator
import numpy as np

from preprocess import Preprocessor
import util
from transformer import CustomTransformer


def main():
    import yaml
    import warnings

    warnings.filterwarnings("ignore")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config_pp = config["preprocessor"]
    pp = Preprocessor(
        spacy_names=config_pp["spacy_names"],
        exts=config_pp["exts"],
        data_root=config_pp["data_root"],
        min_freq=config_pp["min_freq"],
        max_size=config_pp["max_size"],
    )

    src_vocab_size = len(pp.src_field.vocab)
    tgt_vocab_size = len(pp.tgt_field.vocab)

    src_pad_idx = pp.src_field.vocab.stoi[pp.src_field.pad_token]
    tgt_pad_idx = pp.tgt_field.vocab.stoi[pp.tgt_field.pad_token]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}")

    config_data = config["data"]
    train_iter, val_iter, test_iter = BucketIterator.splits(
        datasets=(pp.train, pp.val, pp.test),
        batch_size=config_data["batch_size"],
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    params = {
        "epochs": 10,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "embedding_dim": 512,
        "nhead": 8,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dropout": 0.1,
        "block_size": 100,
        "dim_feedforward": 4,
    }

    # model = Transformer(
    #     src_vocab_size=src_vocab_size,
    #     tgt_vocab_size=tgt_vocab_size,
    #     num_encoder_layers=params["num_encoder_layers"],
    #     num_decoder_layers=params["num_decoder_layers"],
    #     src_pad_idx=src_pad_idx,
    #     tgt_pad_idx=tgt_pad_idx,
    #     block_size=params["block_size"],
    #     device=device,
    #     embedding_dim=params["embedding_dim"],
    #     nhead=params["nhead"],
    #     dim_feedforward=params["dim_feedforward"],
    #     dropout=params["dropout"],
    # ).to(device)

    model = CustomTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        params=params,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10
    )

    if config["load_model"]:
        state = torch.load(config["model_filename"], map_location=torch.device(device))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    num_params = sum([p.nelement() for p in model.parameters()])
    print(f"\nmodel parameters: {num_params}")
    print(f"\n{config=}")
    print(f"\nexample text to trasnlate: {config['example_text']}")

    answer = input("\nwould you like to proceed? (y/n): ")

    if answer.lower() in {"y", "yes"}:
        for epoch in range(params["epochs"]):
            print(f"epoch {epoch} / {params['epochs']}")

            model.eval()
            with torch.no_grad():
                translated_text = util.translate(
                    config["example_text"],
                    model,
                    pp,
                    params["block_size"],
                    device,
                )
                print(f"translated example text:\n {translated_text}")

            model.train()
            losses = []
            for batch in train_iter:
                src = batch.src.T.to(device)
                tgt = batch.trg.T.to(device)

                logits = model(src, tgt[:, :-1])
                loss = util.get_loss(
                    logits,
                    tgt[:, 1:],
                    ignore_index=pp.tgt_field.vocab.stoi[pp.tgt_field.pad_token],
                )
                losses.append(loss.item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            if config["save_model"]:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(checkpoint, config["model_filename"])

            scheduler.step(np.mean(losses))

    print(f"\ncomputing bleu score...")
    score = util.bleu(pp.test[:100], model, pp, params["block_size"], device) * 100
    print(f"bleu score: {score:0.2f}%")


if __name__ == "__main__":
    main()
