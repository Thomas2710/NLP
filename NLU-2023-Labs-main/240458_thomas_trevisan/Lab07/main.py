import sklearn_crfsuite
from sklearn_crfsuite import CRF
from nltk.corpus import conll2002

# to import conll
import os
import sys

from conll import evaluate

# for nice tables
import pandas as pd
from functions import *


# let's get only word and iob-tag
trn_sents = [
    [(text, iob) for text, pos, iob in sent]
    for sent in conll2002.iob_sents("esp.train")
]
tst_sents = [
    [(text, iob) for text, pos, iob in sent]
    for sent in conll2002.iob_sents("esp.testa")
]


trn_label = [sent2labels(s) for s in trn_sents]

trn_feats_baseline = [sent2spacy_features(s) for s in trn_sents]
tst_feats_baseline = [sent2spacy_features(s) for s in tst_sents]


trn_feats_suffix = [sent2spacy_features(s, "suffix") for s in trn_sents]
tst_feats_suffix = [sent2spacy_features(s, "suffix") for s in tst_sents]


trn_feats_tutorial = [sent2spacy_features(s, "tutorial") for s in trn_sents]
tst_feats_tutorial = [sent2spacy_features(s, "tutorial") for s in tst_sents]


trn_feats_tutorial = [
    sent2spacy_features(
        s,
        "tutorial",
    )
    for s in trn_sents
]
tst_feats_tutorial = [
    sent2spacy_features(
        s,
        "tutorial",
    )
    for s in tst_sents
]


trn_feats_tutorial_context1 = [sent2spacy_features(s, "tutorial", 1) for s in trn_sents]
tst_feats_tutorial_context1 = [sent2spacy_features(s, "tutorial", 1) for s in tst_sents]


trn_feats_tutorial_context2 = [sent2spacy_features(s, "tutorial", 2) for s in trn_sents]
tst_feats_tutorial_context2 = [sent2spacy_features(s, "tutorial", 2) for s in tst_sents]


crf = CRF(
    algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True
)

print("Basic")
hyp = fit_predict(crf, trn_feats_baseline, trn_label, tst_feats_baseline)
result_table = evaluation(tst_sents, hyp)
print(result_table)

print("With suffix")
hyp = fit_predict(crf, trn_feats_suffix, trn_label, tst_feats_suffix)
result_table = evaluation(tst_sents, hyp)
print(result_table)

print("tutorial features")
hyp = fit_predict(crf, trn_feats_tutorial, trn_label, tst_feats_tutorial)
result_table = evaluation(tst_sents, hyp)
print(result_table)

print("tutorial features with window")
hyp = fit_predict(
    crf, trn_feats_tutorial_context1, trn_label, tst_feats_tutorial_context1
)
result_table = evaluation(tst_sents, hyp)
print(result_table)

print("tutorial features with larger window")
hyp = fit_predict(
    crf, trn_feats_tutorial_context2, trn_label, tst_feats_tutorial_context2
)
result_table = evaluation(tst_sents, hyp)
print(result_table)
