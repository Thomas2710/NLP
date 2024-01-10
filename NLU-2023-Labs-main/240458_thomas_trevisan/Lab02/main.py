from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
import nltk
from functions import *
from nltk.corpus import stopwords

nltk.download("stopwords")

# parameters
strategy = "uniform"
n_splits = 10
C_count_vect = 0.00008
C_tf_idf = 1
binary = True
stop_words = stopwords.words("english")
max_df = 0.99995
min_df = 0.00005

# Fetching data
all_data = fetch_20newsgroups(subset="all")


# parameters to pass to function
params = {}
params["split"] = StratifiedKFold(n_splits=n_splits, shuffle=True)

# Dummy count vect
params["vectorizer"] = CountVectorizer()
params["clf"] = DummyClassifier(strategy=strategy)
run_experiment(params, "Dummy classifier with count vect", all_data)

# Count vectorizer binary
params["vectorizer"] = CountVectorizer(binary=binary)
params["clf"] = svm.LinearSVC(C=C_count_vect)
run_experiment(params, "Count vectorizer binary", all_data)

# TF-IDF vectorizer no params
params["vectorizer"] = TfidfVectorizer()
params["clf"] = svm.LinearSVC(C=C_tf_idf)
run_experiment(params, "TFIDF no params", all_data)

# TF-IDF vectorizer min-max cutoff
params["vectorizer"] = TfidfVectorizer(
    lowercase=False, max_df=max_df, min_df=min_df, stop_words=None
)
params["clf"] = svm.LinearSVC(C=C_tf_idf)
run_experiment(params, "TFIDF min-max cutoff", all_data)

# TF-IDF vectorizer without stopwords
params["vectorizer"] = TfidfVectorizer(
    lowercase=False, max_df=max_df, min_df=min_df, stop_words=stop_words
)
params["clf"] = svm.LinearSVC(C=C_tf_idf)
run_experiment(params, "TFIDF stopwords", all_data)

# TF-IDF vectorizer all params
params["vectorizer"] = TfidfVectorizer(
    lowercase=True, max_df=max_df, min_df=min_df, stop_words=stop_words
)
params["clf"] = svm.LinearSVC(C=C_tf_idf)
run_experiment(params, "TFIDF all params", all_data)
