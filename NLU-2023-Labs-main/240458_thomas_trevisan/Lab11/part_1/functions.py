from tqdm import tqdm
import torch
import nltk
from nltk.metrics import accuracy
from transformers import BertTokenizer
import os
from functools import partial
from torch import nn
from torch import optim
from utils import *
from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_predictions(sample, model):
    input_ids = sample["input_ids"].to(device)
    mask = sample["mask"].to(device)
    labels = sample["labels"].to(device)
    # Case where model is polarity model
    if "decoder_ids" in sample.keys():
        decoder_ids = sample["decoder_ids"].to(device)
    else:
        decoder_ids = None
    if "n_chunks" in sample.keys():
        n_chunks = sample["n_chunks"].to(device)
    else:
        n_chunks = None
    prediction = model(input_ids, mask, decoder_ids, n_chunks).squeeze()
    return prediction, labels


def train_loop(data, model, optimizer, criterion):
    model.train()
    # Iterate on batches and train model
    train_loss = []
    avg_loss = 0
    for sample in data:
        optimizer.zero_grad()

        # prediction.size() = [32] in case of subj
        # prediction.size() > [32] in case of pol
        prediction, labels = get_predictions(sample, model)
        # If polarity, we should average together the pieces before  computing the loss
        loss = criterion(prediction.float(), labels.float())
        train_loss.append(loss.item())

        loss.backward()

        optimizer.step()
        avg_loss = sum(train_loss) / len(train_loss)
    return avg_loss


def eval_loop(data, model, criterion):
    model.eval()
    # Iterate on batches and train model
    eval_loss = []
    accuracy_list = []
    with torch.no_grad():
        for sample in data:
            # prediction.size() = (32,1)
            prediction, labels = get_predictions(sample, model)

            loss = criterion(prediction.float(), labels.float())
            eval_loss.append(loss.item())

            binary_predictions = prediction >= 0.5
            accuracy_list.append(accuracy(labels, binary_predictions))

            avg_loss = sum(eval_loss) / len(eval_loss)
            avg_accuracy = sum(accuracy_list) / len(accuracy_list)
    return avg_loss, avg_accuracy


# BERT FOR SUBJECTIVITY DETECTION
def get_subjectivity_model(dataset, PATH, model_name, skf, train=False, dropout_prob=0):
    partial_collate_fn = partial(subjectivity_collate_fn, model_name=model_name)

    overall_acc = []
    best_accuracy = 0
    for train_idxs, test_idxs in tqdm(skf.split(dataset.data, dataset.labels)):
        model = SubjectivityBert(model_name, dropout_prob).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        # Get texts, attention masks, labels for this 'batch' of 9000 sentences
        train_dataset, test_dataset = [dataset[indx] for indx in train_idxs], [
            dataset[indx] for indx in test_idxs
        ]

        # Divide in batches because the entire split won't fit in GPU RAM
        batch_size = 32
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=partial_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, collate_fn=partial_collate_fn
        )
        epochs = 5
        best_eval_loss = 0
        train_losses = []
        if train:
            for epoch in range(epochs):
                train_loss = train_loop(train_loader, model, optimizer, criterion)
                train_losses.append(train_loss)

                eval_loss, running_accuracy = eval_loop(test_loader, model, criterion)
                overall_acc.append(running_accuracy)
                # print(f"running accuracy is {running_accuracy.item()}")
                if running_accuracy > best_accuracy:
                    best_accuracy = running_accuracy
                    torch.save(
                        model.state_dict(),
                        PATH,
                    )
        else:
            best_model = torch.load(PATH)
            model.load_state_dict(best_model)
            eval_loss, running_accuracy = eval_loop(test_loader, model, criterion)
            overall_acc.append(running_accuracy)
    # print(
    #    f"Accuracy of subjectivity detection, averaged across 10 splits/runs is {sum(overall_acc)/len(overall_acc)}"
    # )
    return model, sum(overall_acc) / len(overall_acc)


def get_best_subj_model(PATH, model_name, dropout_prob=0):
    # Comment till here
    model = SubjectivityBert(model_name, dropout_prob).to(device)
    best_model = torch.load(PATH)
    model.load_state_dict(best_model)
    return model


def get_polarity_model(dataset, PATH, model_name, skf, train, dropout_prob):
    partial_collate_fn = partial(polarity_collate_fn, model_name=model_name)
    overall_acc = []
    best_accuracy = 0
    for train_idxs, test_idxs in tqdm(skf.split(dataset.data, dataset.labels)):
        model = PolarityT5(model_name, dropout_prob).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        # Get texts, attention masks, labels for this 'batch' of 9000 sentences
        train_dataset, test_dataset = [dataset[indx] for indx in train_idxs], [
            dataset[indx] for indx in test_idxs
        ]
        # train_dataset[0] is tuple (list of words, numeric label)
        # Divide in batches because the entire split won't fit in GPU RAM
        # Create loaders for polarity training
        batch_size = 8
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=partial_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, collate_fn=partial_collate_fn
        )

        epochs = 5
        best_eval_loss = 0
        train_losses = []
        if train:
            for epoch in range(epochs):
                train_loss = train_loop(train_loader, model, optimizer, criterion)
                train_losses.append(train_loss)

                eval_loss, running_accuracy = eval_loop(test_loader, model, criterion)
                overall_acc.append(running_accuracy)
                if running_accuracy > best_accuracy:
                    best_accuracy = running_accuracy
                    torch.save(model.state_dict(), PATH)
        else:
            best_model = torch.load(PATH)
            model.load_state_dict(best_model)
            eval_loss, running_accuracy = eval_loop(test_loader, model, criterion)
            overall_acc.append(running_accuracy)
    # print(
    #    f"Accuracy of subjectivity detection, averaged across 10 splits/runs is {sum(overall_acc) / len(overall_acc)}"
    # )
    return model, sum(overall_acc) / len(overall_acc)


def get_best_pol_model(PATH, model_name, dropout_prob=0):
    # Comment till here
    model = PolarityT5(model_name, dropout_prob).to(device)
    best_model = torch.load(PATH)
    model.load_state_dict(best_model)
    return model
