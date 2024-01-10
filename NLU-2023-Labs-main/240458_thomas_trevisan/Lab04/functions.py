import math
import nltk

# rules from NLTK adapted to Universal Tag Set & extended
rules = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'DET'),   # articles
    (r'.*able$', 'ADJ'),                # adjectives
    (r'.*ness$', 'NOUN'),               # nouns formed from adjectives
    (r'.*ly$', 'ADV'),                  # adverbs
    (r'.*s$', 'NOUN'),                  # plural nouns
    (r'.*ing$', 'VERB'),                # gerunds
    (r'.*ed$', 'VERB'),                 # past tense verbs
    (r'.*ed$', 'VERB'),                 # past tense verbs
    (r'[\.,!\?:;\'"]', '.'),            # punctuation (extension) 
    (r'(In|in|Among|among|Above|above|as|As)$', 'ADP'),   # prepositions
    (r'(to|To|well|Well|Up|up|Not|not|Now|now)$', 'PRT'),   # particles
    (r'(I|you|You|He|he|She|she|It|it|They|they|We|we)$', 'PRON'),   # pronouns
    (r'(and| or|But|but|while|since)$', 'CONJ'),# conjunctions
    (r'.*', 'NOUN'),                     # nouns (default)
]



# See above for further details
mapping_spacy_to_NLTK = {
    "ADJ": "ADJ",
    "ADP": "ADP",
    "ADV": "ADV",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "DET": "DET",
    "INTJ": "X",
    "NOUN": "NOUN",
    "NUM": "NUM",
    "PART": "PRT",
    "PRON": "PRON",
    "PROPN": "NOUN",
    "PUNCT": ".",
    "SCONJ": "CONJ",
    "SYM": "X",
    "VERB": "VERB",
    "X": "X"
}

def load_data(dataset):
    total_size = len(dataset.tagged_sents())
    train_indx = math.ceil(total_size * 0.9)
    trn_data = dataset.tagged_sents(tagset='universal')[:train_indx]
    tst_data = dataset.tagged_sents(tagset='universal')[train_indx:]
    return trn_data, tst_data, train_indx


def compute_best_ngram_model(trn_data, tst_data, backoff_tagger):
    ngram_order = [1,2,3]
    cutoffs = [1,10,50]
    best_accuracy = 0
    best_n = -1
    best_cutoff = 1
    
    for n in ngram_order:
        for cutoff in cutoffs:
            #Creating nltk tagger
            nltk_tagger = nltk.NgramTagger(n, train=trn_data, model=None, backoff=backoff_tagger, cutoff=cutoff, verbose=False)
            #Evaluating tagger
            current_accuracy =nltk_tagger.accuracy(tst_data)
            #print(f"NLTK tagger accuracy with ngram order {n}:{accuracy:.3}")
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_n = n
                best_cutoff = cutoff
    
    return best_n, best_cutoff