from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import senseval
from nltk.wsd import lesk

import nltk
import numpy as np
from pprint import pprint
from functions import *

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("senseval")
nltk.download("averaged_perceptron_tagger")

mapping = {
    "interest_1": "interest.n.01",
    "interest_2": "interest.n.03",
    "interest_3": "pastime.n.01",
    "interest_4": "sake.n.01",
    "interest_5": "interest.n.05",
    "interest_6": "interest.n.04",
}

data = [
    " ".join([t[0] for t in inst.context])
    for inst in senseval.instances("interest.pos")
]
lbls = [inst.senses[0] for inst in senseval.instances("interest.pos")]

# Supervised approach with BOW to solve word-sense disambiguation

vectorizer = CountVectorizer()
classifier = MultinomialNB()
lblencoder = LabelEncoder()

stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

vectors = vectorizer.fit_transform(data)

# encoding labels for multi-calss
lblencoder.fit(lbls)
labels = lblencoder.transform(lbls)


# Supervised approach using dictionary of features to solve word-sense disambiguation

data_col = [
    collocational_features(inst, NGRAM_WINDOW)
    for inst in senseval.instances("interest.pos")
]
dvectorizer = DictVectorizer(sparse=False)
dvectors = dvectorizer.fit_transform(data_col)

concatenated_vectors = np.concatenate((vectors.toarray(), dvectors), axis=1)

scores = cross_validate(
    classifier, concatenated_vectors, labels, cv=stratified_split, scoring=["f1_micro"]
)

print(
    f"f1 collocational features is {sum(scores['test_f1_micro']) / len(scores['test_f1_micro'])}"
)
print("")

# since WordNet defines more senses, let's restrict predictions
synsets = []
for ss in wordnet.synsets("interest", pos="n"):
    if ss.name() in mapping.values():
        defn = ss.definition()
        tags = preprocess(defn)
        toks = [l for w, l, p in tags]
        synsets.append((ss, toks))

run_experiment(data, lbls, stratified_split, mapping, synsets, "lesk")
run_experiment(data, lbls, stratified_split, mapping, synsets, "pedersen")
