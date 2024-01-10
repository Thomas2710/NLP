from utils import *
from torch.utils.data import DataLoader
from functools import partial
import os
from functions import *


if __name__ == "__main__":
    train_raw = read_file(
        os.path.join(os.path.dirname(__file__), "dataset", "ptb.train.txt")
    )
    dev_raw = read_file(
        os.path.join(os.path.dirname(__file__), "dataset", "ptb.valid.txt")
    )
    test_raw = read_file(
        os.path.join(os.path.dirname(__file__), "dataset", "ptb.test.txt")
    )
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=1024,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )

    optimizer_params = {
        "optimizer": "SGD",
        "lr": 0.1,
        "momentum": 0.95,
        "dampening": 0,
        "nesterov": True,
        "wd": 1e-4,
    }

    run_experiment(train_loader, dev_loader, test_loader, lang, optimizer_params, 1)

    emb_dropout = 0.2
    out_dropout = 0.5
    optimizer_params["momentum"] = 0.99

    run_experiment(
        train_loader,
        dev_loader,
        test_loader,
        lang,
        optimizer_params,
        2,
        emb_dropout,
        out_dropout,
    )

    optimizer_params["optimizer"] = "AdamW"
    optimizer_params["lr"] = 1e-3
    optimizer_params["wd"] = 1e-3
    run_experiment(
        train_loader,
        dev_loader,
        test_loader,
        lang,
        optimizer_params,
        3,
        emb_dropout,
        out_dropout,
    )
