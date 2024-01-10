import nltk
import numpy as np
import spacy
import en_core_web_sm
from functions import *
from collections import Counter

nltk.download("gutenberg")
nltk.download("punkt")

nltk.corpus.gutenberg.fileids()


corpus = "milton-paradise.txt"

corpus_chars = nltk.corpus.gutenberg.raw(corpus)
corpus_words = nltk.corpus.gutenberg.words(corpus)
corpus_sents = nltk.corpus.gutenberg.sents(corpus)

# number of rows to select from computed frequencies
N = 10

print("---------------------------------------------------\n")
(
    total_chars,
    total_words,
    total_sents,
    word_per_sent,
    char_per_word,
    char_per_sent,
    longest_sent,
    longest_word,
    shortest_sent,
    shortest_word,
) = my_statistics(corpus_chars, corpus_words, corpus_sents)
print("Total number of characters", total_chars)
print("Total number of words", total_words)
print("Total number of sents", total_sents)
print("Words per sentence", word_per_sent)
print("Chars per word", char_per_word)
print("Chars per sentence", char_per_sent)
print("Longest sentence", longest_sent)
print("Longest word", longest_word)
print("Shortest sentence", shortest_sent)
print("shortest word", shortest_word)


# Tokenization with spacy
print("---------------------------------------------------\n")
print("SPACY tokenization")
nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])
txt = corpus_chars
doc = nlp(txt)
spacy_corpus_chars = doc.text
spacy_corpus_words = [word.text for word in doc]
spacy_corpus_sents = list(doc.sents)


(
    spacy_total_chars,
    spacy_total_words,
    spacy_total_sents,
    spacy_word_per_sent,
    spacy_char_per_word,
    spacy_char_per_sent,
    spacy_longest_sent,
    spacy_longest_word,
    spacy_shortest_sent,
    spacy_shortest_word,
) = my_statistics(spacy_corpus_chars, spacy_corpus_words, spacy_corpus_sents)

print("Total number of characters", spacy_total_chars)
print("Total number of words", spacy_total_words)
print("Total number of sents", spacy_total_sents)
print("Words per sentence", spacy_word_per_sent)
print("Chars per word", spacy_char_per_word)
print("Chars per sentence", spacy_char_per_sent)
print("Longest sentence", spacy_longest_sent)
print("Longest word", spacy_longest_word)
print("Shortest sentence", spacy_shortest_sent)
print("shortest word", spacy_shortest_word)


# Tokenization with nltk
print("---------------------------------------------------\n")
print("NLTK tokenization")
nltk_corpus_words = nltk.word_tokenize(corpus_chars)
nltk_corpus_sents = nltk.sent_tokenize(corpus_chars)
(
    nltk_total_chars,
    nltk_total_words,
    nltk_total_sents,
    nltk_word_per_sent,
    nltk_char_per_word,
    nltk_char_per_sent,
    nltk_longest_sent,
    nltk_longest_word,
    nltk_shortest_sent,
    nltk_shortest_word,
) = my_statistics(corpus_chars, nltk_corpus_words, nltk_corpus_sents)

print("Total number of characters", nltk_total_chars)
print("Total number of words", nltk_total_words)
print("Total number of sents", nltk_total_sents)
print("Words per sentence", nltk_word_per_sent)
print("Chars per word", nltk_char_per_word)
print("Chars per sentence", nltk_char_per_sent)
print("Longest sentence", nltk_longest_sent)
print("Longest word", nltk_longest_word)
print("Shortest sentence", nltk_shortest_sent)
print("shortest word", nltk_shortest_word)
print("------------------------------------------------------------\n")


# ---------------LEXICON-----------------------
corpus_lexicon_set = set([w.lower() for w in corpus_words])
print("Reference corpus lowercase lexicon size: ", len(corpus_lexicon_set))


spacy_corpus_lexicon_set = set([str(w).lower() for w in spacy_corpus_words])
print("Spacy processed lowercase lexicon size: ", len(spacy_corpus_lexicon_set))

nltk_corpus_lexicon_set = set([w.lower() for w in nltk_corpus_words])
print("Nltk processed lowercase lexicon size: ", len(nltk_corpus_lexicon_set))

# ----------------N most frequent words------------
print("------------------------------------------------------------\n")
corpus_lexicon_list = [w.lower() for w in corpus_words]
corpus_lowercase_freq_list = Counter(corpus_lexicon_list)
print("Frequency table: ", nbest(corpus_lowercase_freq_list, n=N))


spacy_corpus_lexicon_list = [str(w).lower() for w in spacy_corpus_words]
spacy_corpus_lowercase_freq_list = Counter(spacy_corpus_lexicon_list)
print("Spacy frequency table: ", nbest(spacy_corpus_lowercase_freq_list, n=N))

nltk_corpus_lexicon_list = list([w.lower() for w in nltk_corpus_words])
nltk_corpus_lowercase_freq_list = Counter(nltk_corpus_lexicon_list)
print("nltk frequency table: ", nbest(nltk_corpus_lowercase_freq_list, n=N))

print("---------------------------------------------------------------")
