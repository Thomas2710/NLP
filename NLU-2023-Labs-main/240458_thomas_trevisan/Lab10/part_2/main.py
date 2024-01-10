import utils
from utils import *
from functions import *
from model import *
import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm

BERT_MODEL = "bert-base-uncased"
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
        train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    dropout_prob = 0.2
    lr = 5e-5  # learning rate
    eps = 1e-8  # avoid division by zero
    runs = 10
    best_f1 = 0

    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = JointIntentAndSlotFillingModel(
            out_slot, out_int, dropout_prob=dropout_prob, model_name=BERT_MODEL
        ).to(device)
        # Check if model params are correctly given to optimizer
        # list_params(model)

        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        criterion_intents = nn.CrossEntropyLoss()

        # Train model
        if not roccabruna:
            optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
            n_epochs = 10
            patience = 5
            losses_train = []
            losses_dev = []
            sampled_epochs = []
            for x in range(0, n_epochs):
                loss = train_loop(
                    train_loader, optimizer, criterion_slots, criterion_intents, model
                )
                if x % 1 == 0:
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
                    if patience <= 0:  # Early stoping with patience
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
