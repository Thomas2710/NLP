import math
import spacy
import nltk

nltk.download("treebank")
nltk.download("universal_tagset")
from nltk.corpus import treebank
from nltk import NgramTagger
from nltk import DefaultTagger
from nltk.tag import RegexpTagger
from nltk.util import flatten
from nltk.metrics import accuracy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.tokenizer import Tokenizer
from functions import *
import spacy
import en_core_web_sm


ngram_order = [1, 2, 3]
cutoffs = [1, 10, 50]


##Generating train and test set
trn_data, tst_data, train_indx = load_data(treebank)
# print("Total: {}; Train: {}; Test: {}".format(total_size, len(trn_data), len(tst_data)))

# NLTK tagger

##Backoff tagger
##1)backoff_tagger = DefaultTagger('NN')
backoff_tagger = RegexpTagger(rules)
best_n, best_cutoff = compute_best_ngram_model(trn_data, tst_data, backoff_tagger)

nltk_tagger = nltk.NgramTagger(
    best_n,
    train=trn_data,
    model=None,
    backoff=backoff_tagger,
    cutoff=best_cutoff,
    verbose=False,
)


##Spacy tagger
nlp = en_core_web_sm.load()

# Method1
nlp.tokenizer = Tokenizer(nlp.vocab)
spacy_test_data = [" ".join(sent) for sent in treebank.sents()[train_indx:]]
doc1 = nlp(" ".join(spacy_test_data))

# Method2
# doc = Doc(nlp.vocab, flatten(list(treebank.sents()[train_indx:])))
# doc1 = nlp(doc, disable = ['tokenizer'])

# Creating the set of tuples to evaluate
result_set = [(token.text, mapping_spacy_to_NLTK[token.pos_]) for token in doc1]

# Creating the set of tuples as reference set from treebank_dataset
reference_tuples = [
    tagged_token
    for sent in treebank.tagged_sents(tagset="universal")[train_indx:]
    for tagged_token in sent
]
spacy_accuracy = accuracy(reference_tuples, result_set)
print(
    f"NLTK tagger accuracy with ngram order: {best_n}: {nltk_tagger.accuracy(tst_data):.3} | Spacy accuracy: {spacy_accuracy:.3}"
)
