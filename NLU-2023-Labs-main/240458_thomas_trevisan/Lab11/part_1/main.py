import nltk

nltk.download("movie_reviews")
nltk.download("punkt")
import numpy as np
import pandas as pd
import transformers
import os
import torch
from torch import nn, optim

from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

nltk.download("subjectivity")


from functions import *
from utils import *
from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    # SUBJECTIVITY DETECTION
    train_subjectivity = False
    SUBJ_PATH = os.path.dirname(__file__) + "/bin/best-subjectivity-model.pt"
    subj_model_name = "bert-base-uncased"
    subj_dropout_prob = 0.2

    subj_dataset = get_subjectivity_data()
    subj_model, subj_acc = get_subjectivity_model(
        subj_dataset,
        SUBJ_PATH,
        subj_model_name,
        skf,
        train_subjectivity,
        subj_dropout_prob,
    )
    print(f"\n\n accuracy over 10 runs for subjectivity task is {subj_acc}\n\n ")

    # POLARITY DETECTION WITH ALL SENTENCES
    train_polarity = False
    POL_PATH = os.path.dirname(__file__) + "/bin/best-polarity-model1.pt"
    pol_model_name = "t5-small"
    pol_dropout_prob = 0.2

    mr = movie_reviews
    rev_neg = mr.paras(categories="neg")
    rev_pos = mr.paras(categories="pos")
    pol_dataset = get_polarity_data(rev_neg, rev_pos)

    pol_model, pol_acc = get_polarity_model(
        pol_dataset, POL_PATH, pol_model_name, skf, train_polarity, pol_dropout_prob
    )
    print(
        f"\n\n accuracy over 10 runs for polarity with also objective sentences is task is {pol_acc}\n\n "
    )
    # DO IT ONLY WITH SUBJECTIVE SENTENCES

    best_subjectivity_model = get_best_subj_model(
        SUBJ_PATH, subj_model_name, subj_dropout_prob
    )
    best_subjectivity_model.eval()
    neg_subjective_movie_revs = get_subjective_dataset(rev_neg, best_subjectivity_model)
    pos_subjective_movie_revs = get_subjective_dataset(rev_pos, best_subjectivity_model)
    subjective_movie_dataset = get_polarity_data(
        neg_subjective_movie_revs, pos_subjective_movie_revs
    )

    # Run again without objective sentences
    train_polarity = False
    POL_PATH = os.path.dirname(__file__) + "/bin/best-polarity-model2.pt"
    pol_model_name = "t5-small"
    pol_dropout_prob = 0.2
    pol_model, pol_acc = get_polarity_model(
        subjective_movie_dataset,
        POL_PATH,
        pol_model_name,
        skf,
        train_polarity,
        pol_dropout_prob,
    )
    print(
        f"\n\n accuracy over 10 runs for polarity with only subjective sentences is task is {pol_acc}\n\n"
    )
