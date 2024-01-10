import torch
import os
import sys


import numpy as np
from collections import Counter
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
from tqdm import tqdm
import transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    clip = 5
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(sample["utterances"], sample["attention_masks"])
        slots = slots[:, :, 1:-1]
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["utterances"], sample["attention_masks"])
            slots = slots[:, :, 1:-1]
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            # Get the highest probable class
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            # print("\n\n\n\n")
            # print(f"hyp_intents are: {hyp_intents}")
            # print("\n\n\n\n")
            # print(f"ref_intents are: {ref_intents}")
            # Slot inference
            # ids of predicted slots
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                # We get the slots for which the length is the one of the original utterance
                # So, no padding slots
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][1 : length + 1].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                to_decode = seq[:length].tolist()
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
            # print(f"hyp_slots are: {hyp_slots}")
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


def list_params(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print("The BERT model has {:} different named parameters.\n".format(len(params)))

    print("==== Embedding Layer ====\n")

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n==== First Transformer ====\n")

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print("\n==== Output Layer ====\n")

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
