import numpy as np
import torch
import math
import torch.nn as nn
from tqdm import tqdm
import copy
import torch.optim as optim
import os
from model import *


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def run_experiment(
    PATH,
    train_loader,
    dev_loader,
    test_loader,
    lang,
    optimizer_params,
    emb_dropout=0,
    out_dropout=0,
    hidden_dropout=0,
    weight_tying=False,
    variational=False,
    logging_interval=1,
):
    roccabruna = True
    hid_size = 500
    emb_size = 500
    clip = 5  # Clip the gradient

    # As described in the paper
    non_monotone_interval = 5
    # t = len(best_val_loss)
    # logs are equal to best_val_loss

    vocab_len = len(lang.word2id)

    model = LM_LSTM(
        emb_size,
        hid_size,
        vocab_len,
        emb_dropout,
        out_dropout,
        hidden_dropout,
        weight_tying,
        variational,
        pad_index=lang.word2id["<pad>"],
    ).to(device)
    model.apply(init_weights)

    if optimizer_params["optimizer"] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_params["lr"],
            weight_decay=optimizer_params["wd"],
        )
    elif (
        optimizer_params["optimizer"] == "SGD"
        or optimizer_params["optimizer"] == "NT-ASGD"
    ):
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_params["lr"],
            weight_decay=optimizer_params["wd"],
            momentum=optimizer_params["momentum"],
            dampening=optimizer_params["dampening"],
            nesterov=optimizer_params["nesterov"],
        )
    else:
        print("weird optimizer setting")

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    if not roccabruna:
        n_epochs = 100
        patience = 15
        losses_train = []
        losses_dev = []
        ppls_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, n_epochs))
        # If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
                    # best_model = copy.deepcopy(model).to("cpu")
                    torch.save(model.state_dict(), PATH)
                    patience = 15
                else:
                    patience -= 1

                if (
                    epoch % logging_interval == 0
                    and optimizer.__class__.__name__ != "ASGD"
                ):
                    ppls_dev.append(ppl_dev)
                    # print(loss_dev)
                    if (
                        optimizer.__class__.__name__ == "SGD"
                        and "t0" not in optimizer.param_groups[0]
                        and (
                            len(losses_dev) > non_monotone_interval
                            and loss_dev > min(losses_dev[:-non_monotone_interval])
                        )
                        and optimizer_params["optimizer"] == "NT-ASGD"
                    ):
                        print("Switching to ASGD")
                        optimizer = torch.optim.ASGD(
                            model.parameters(),
                            optimizer_params["lr"],
                            weight_decay=optimizer_params["wd"],
                            t0=0,
                        )

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

    best_model = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(best_model)
    best_model = model
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print("Test ppl: ", final_ppl)
