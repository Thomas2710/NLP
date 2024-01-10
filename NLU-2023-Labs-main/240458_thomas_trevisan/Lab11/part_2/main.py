from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import math
from utils import *
from functions import *
from model import *

model_name = "bert-base-uncased"

# Usage
dev_file_path = os.path.join(
    os.path.dirname(__file__), "dataset", "conll_laptop14_dev.txt"
)
train_file_path = os.path.join(
    os.path.dirname(__file__), "dataset", "conll_laptop14_train.txt"
)
test_file_path = os.path.join(
    os.path.dirname(__file__), "dataset", "conll_laptop14_test.txt"
)

dev_data = read_conll_file(dev_file_path)
train_data = read_conll_file(train_file_path)
test_data = read_conll_file(test_file_path)

words = [word for sent in train_data for word, tag in sent]
aspects = [tag.split("-")[0] for sent in train_data for word, tag in sent]
polarities = [
    tag.split("-")[1] if tag != "O" else "O"
    for sent in train_data
    for word, tag in sent
]
lang = Lang(words, aspects, polarities, model_name, cutoff=1)

train_dataset = AspectBasedDataset(train_data, lang)
test_dataset = AspectBasedDataset(test_data, lang)
dev_dataset = AspectBasedDataset(dev_data, lang)

batch_size = 32
partial_collate_fn = partial(collate_fn, model_name=model_name)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=partial_collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=partial_collate_fn
)
dev_loader = DataLoader(
    dev_dataset, batch_size=batch_size, collate_fn=partial_collate_fn
)

# SPAN EXTRACTOR STUFF
train = False
SPAN_PATH = os.path.dirname(__file__) + "/bin/best_span_model.pt"
runs = 2
lr = 3e-5
epochs = 5
top_k = 1

best_test_loss = math.inf
best_dev_loss = math.inf
best_train_loss = math.inf
number_of_polarities = len(lang.polarity2id)

span_criterion = SpanLoss()
polarity_criterion = PolarityLoss()

# Loop over the runs
for run in tqdm(range(runs)):
    if train:
        model = SpanExtractor(model_name, number_of_polarities).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        best_loss = math.inf
        for epoch in range(epochs):
            span_loss_train = span_train_loop(
                train_loader,
                model,
                span_criterion,
                polarity_criterion,
                optimizer,
                top_k,
            )
            if span_loss_train < best_train_loss:
                best_train_loss = span_loss_train
            span_loss_dev, best_loss = span_dev_loop(
                SPAN_PATH,
                dev_loader,
                model,
                span_criterion,
                polarity_criterion,
                best_loss,
                top_k,
            )
            if span_loss_dev < best_dev_loss:
                best_dev_loss = span_loss_dev

        model = SpanExtractor(model_name, number_of_polarities).to(device)
        best_model = torch.load(SPAN_PATH)
        model.load_state_dict(best_model)
        span_loss_test = span_test_loop(
            test_loader, model, span_criterion, polarity_criterion, top_k
        )
        print(f"running loss {span_loss_test}")
        if span_loss_test < best_test_loss:
            best_test_loss = span_loss_test

# print(f"best test loss was {best_test_loss}")
