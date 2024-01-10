# Spacy version

from nltk.corpus import dependency_treebank
import spacy
import spacy_conll
import en_core_web_sm
import itertools
from itertools import zip_longest, tee
import stanza
import spacy_stanza
from nltk.parse import DependencyEvaluator
from spacy.tokenizer import Tokenizer
from functions import *

N = 100

# Load the spacy model
spacy_nlp = spacy.load("en_core_web_sm")
# Set up the conll formatter
config = {
    "ext_names": {"conll_pd": "pandas"},
    "conversion_maps": {"DEPREL": {"nsubj": "subj"}},
}
# Add the formatter to the pipeline
spacy_nlp.add_pipe("conll_formatter", config=config, last=True)
# Split by white space
spacy_nlp.tokenizer = Tokenizer(spacy_nlp.vocab)

# Load the stanza model
stanza_nlp = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)
# Set up the conll formatter
config = {
    "ext_names": {"conll_pd": "pandas"},
    "conversion_maps": {"DEPREL": {"nsubj": "subj", "root": "ROOT"}},
}
# Add the formatter to the pipeline
stanza_nlp.add_pipe("conll_formatter", config=config, last=True)


stanza_dp_list, spacy_dp_list = get_dependency_tree_list(
    stanza_nlp, spacy_nlp, dependency_treebank
)

de = DependencyEvaluator(spacy_dp_list, dependency_treebank.parsed_sents()[-N:])
las, uas = de.eval()

print(f"spacy las: {las}")
print(f"spacy uas: {uas}\n")

de = DependencyEvaluator(stanza_dp_list, dependency_treebank.parsed_sents()[-N:])
las, uas = de.eval()

print(f"stanza las: {las}")
print(f"stanza uas: {uas}")
