from utils import *
from torch.utils.data import DataLoader
from functools import partial

from functions import *

if __name__ == "__main__":
    train_raw = read_file(os.path.dirname(__file__) + "/dataset/ptb.train.txt")
    dev_raw = read_file(os.path.dirname(__file__) + "/dataset/ptb.valid.txt")
    test_raw = read_file(os.path.dirname(__file__) + "/dataset/ptb.test.txt")
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
        "lr": 1,
        "momentum": 0.95,
        "dampening": 0,
        "nesterov": True,
        "wd": 1e-4,
    }

    emb_dropout = 0.1
    out_dropout = 0.4

    weight_tying = True
    hidden_dropout = 0.3
    path = os.path.dirname(__file__) + "/bin/best_model_1.pt"
    run_experiment(
        path,
        train_loader,
        dev_loader,
        test_loader,
        lang,
        optimizer_params,
        emb_dropout,
        out_dropout,
        hidden_dropout,
        weight_tying,
    )

    variational = True
    path = os.path.dirname(__file__) + "/bin/best_model_2.pt"
    run_experiment(
        path,
        train_loader,
        dev_loader,
        test_loader,
        lang,
        optimizer_params,
        emb_dropout,
        out_dropout,
        hidden_dropout,
        weight_tying,
        variational,
    )

    optimizer_params["optimizer"] = "NT-ASGD"
    logging_interval = 1
    path = os.path.dirname(__file__) + "/bin/best_model_3.pt"
    run_experiment(
        path,
        train_loader,
        dev_loader,
        test_loader,
        lang,
        optimizer_params,
        emb_dropout,
        out_dropout,
        hidden_dropout,
        weight_tying,
        variational,
        logging_interval,
    )
