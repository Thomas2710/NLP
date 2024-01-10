from utils import *
from functions import *
from model import *

import os
from tqdm import tqdm
import torch
from torch import nn as nn
from torch import optim as optim
import numpy as np
from torch.utils.data import DataLoader


roccabruna = True
PATH = os.path.dirname(__file__) + "/bin/best_model.pt"

if __name__ == "__main__":
    tmp_train_raw = load_data(
        os.path.join(os.path.dirname(__file__), "dataset", "train.json")
    )
    test_raw = load_data(
        os.path.join(os.path.dirname(__file__), "dataset", "test.json")
    )

    train_raw, dev_raw, intents, y_test = create_splits(tmp_train_raw, test_raw)

    words = sum(
        [x["utterance"].split() for x in train_raw], []
    )  # No set() since we want to compute the cutoff
    corpus = (
        train_raw + dev_raw + test_raw
    )  # We do not wat unk labels, however this depends on the research purpose
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(
        train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    hid_size = 200
    emb_size = 300

    bidirectional = True
    dropout_prob = 0.3
    lr = 0.001  # learning rate
    clip = 5  # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    runs = 5
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = ModelIAS(
            hid_size,
            out_slot,
            out_int,
            emb_size,
            vocab_len,
            dropout_prob=dropout_prob,
            bidirectional=bidirectional,
            pad_index=PAD_TOKEN,
        ).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        if not roccabruna:
            n_epochs = 200
            patience = 3
            losses_train = []
            losses_dev = []
            sampled_epochs = []
            best_f1 = 0
            for x in range(0, n_epochs):
                loss = train_loop(
                    train_loader, optimizer, criterion_slots, criterion_intents, model
                )
                if x % 5 == 0:
                    sampled_epochs.append(x)
                    losses_train.append(np.asarray(loss).mean())
                    results_dev, intent_res, loss_dev = eval_loop(
                        dev_loader, criterion_slots, criterion_intents, model, lang
                    )
                    losses_dev.append(np.asarray(loss_dev).mean())
                    f1 = results_dev["total"]["f"]

                    if f1 > best_f1:
                        best_f1 = f1
                        torch.save(model.state_dict(), PATH)
                    else:
                        patience -= 1
                    if patience <= 0:
                        print("early stopping")  # Early stoping with patient
                        break  # Not nice but it keeps the code clean

        best_model = torch.load(PATH)
        model.load_state_dict(best_model)
        best_model = model
        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang
        )
        intent_acc.append(intent_test["accuracy"])
        slot_f1s.append(results_test["total"]["f"])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print("")
    print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
    print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(slot_f1s.std(), 3))
