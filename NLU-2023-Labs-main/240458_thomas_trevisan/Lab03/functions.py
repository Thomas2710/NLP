import math
import numpy as np
from nltk.lm import Vocabulary
from nltk.lm import NgramCounter


# Custom StupidBackoff
class custom_StupidBackoff:
    def __init__(self, order=2, a=0.4):
        self.order = order
        self.a = a

    def fit(self, ngrams, text):
        self.vocab = Vocabulary(text, unk_cutoff=2)
        self.ngram_counts = NgramCounter(ngrams)
        self.ngram_probs = self.probability()

    def probability(self):
        prob_dict = {}
        prob = 0
        counter = 0
        total = 0
        for i in range(self.order):
            j = i + 2
            for ngram in self.ngram_counts[j]:
                if len(ngram) == 1:
                    numerator_counts = self.ngram_counts[ngram].N()
                    denominator_counts = (
                        self.ngram_counts.unigrams.N()
                    )  # Returns total number of words, with repetitions (2 occurences of the same word are counted as 2)
                else:
                    # Every time that context appears in an ngram of that order
                    denominator_counts = self.ngram_counts[len(ngram[:-1]) + 1][
                        ngram[:-1]
                    ].N()
                    # Every time that word appears after such context in an ngram of that order
                    numerator_counts = self.ngram_counts[len(ngram[:-1]) + 1][
                        ngram[:-1]
                    ][ngram[-1]]
                    # print(f"ngram is {ngram} and the denominator is computed using context {ngram[:j-2]} and word {ngram[j-2]}")
                prob = numerator_counts / denominator_counts
                prob_dict[ngram] = prob
            # print(f"prob is: {numerator_counts}/{denominator_counts}={numerator_counts/denominator_counts}")
        return prob_dict

    def score(self, ngram):
        if ngram in self.ngram_probs.keys():
            return self.ngram_probs[ngram]
        elif len(ngram) >= 1:
            return self.a * self.score(ngram[1:])
        else:
            return 0

    # Kinda taken from nltk docs
    def logscore(self, ngram):
        return math.log(self.score(ngram), 2)

    # Kinda taken from nltk docs
    def entropy(self, ngrams):
        return -1 * np.mean([self.logscore(ngram) for ngram in ngrams])

    # Kinda taken from nltk docs
    def perplexity(self, ngrams):
        return pow(2.0, self.entropy(ngrams))
