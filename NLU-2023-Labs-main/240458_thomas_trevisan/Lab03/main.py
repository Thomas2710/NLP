from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import StupidBackoff
from nltk.lm import Vocabulary
from nltk.corpus import gutenberg
from nltk.lm import NgramCounter
from nltk.lm.preprocessing import flatten
from sklearn.model_selection import train_test_split
from functions import *

NGRAM_ORDER = 3

test_sents1 = []
test_sents2 = []

# NLTK StupidBackoff
# Dataset
# List of Lists of words
macbeth_sents = [
    [w.lower() for w in sent] for sent in gutenberg.sents("shakespeare-macbeth.txt")
]
train_sents, test_sents = train_test_split(
    macbeth_sents, test_size=0.001, random_state=42
)
# Preprocess data
macbeth_train_words = flatten(train_sents)
macbeth_test_words = flatten(test_sents)

# Compute vocab
lex = Vocabulary(macbeth_train_words, unk_cutoff=1)
# Handling OOV
# sent is a list of str(words)
macbeth_oov_sents = [list(lex.lookup(sent)) for sent in train_sents]
padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(
    NGRAM_ORDER, macbeth_oov_sents
)

# Creating model
lm = StupidBackoff(order=NGRAM_ORDER, alpha=0.4)
lm.fit(padded_ngrams_oov, flat_text_oov)

test_ngrams, test_flat_text = padded_everygram_pipeline(
    lm.order, [lex.lookup(sent) for sent in test_sents]
)
print("NLTK PPL")
# Compute PPL extracting ngram equals to lm_oov.order
seq = []
for gen in test_ngrams:
    for x in gen:
        if len(x) == lm.order:
            seq.append(x)

print(lm.perplexity(seq))


print("---------------------------------")

# Regenerating for MY stupid backoff
padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(
    NGRAM_ORDER, macbeth_oov_sents
)
my_lm = custom_StupidBackoff(NGRAM_ORDER, 0.4)
my_lm.fit(padded_ngrams_oov, flat_text_oov)

test_ngrams, test_flat_text = padded_everygram_pipeline(
    my_lm.order, [lex.lookup(sent) for sent in test_sents]
)
print("MY PPL")
# Compute PPL extracting ngram equals to my_lm_oov.order
seq = []
for gen in test_ngrams:
    for x in gen:
        if len(x) == lm.order:
            seq.append(x)

print(my_lm.perplexity(seq))
