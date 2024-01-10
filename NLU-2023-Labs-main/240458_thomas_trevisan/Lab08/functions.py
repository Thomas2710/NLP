from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from collections import Counter
from nltk.corpus import wordnet_ic
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from nltk.util import ngrams
from nltk.corpus import stopwords

nltk.download("wordnet_ic")
semcor_ic = wordnet_ic.ic("ic-semcor.dat")
NGRAM_WINDOW = 3


def preprocess(text):
    mapping = {
        "NOUN": wordnet.NOUN,
        "VERB": wordnet.VERB,
        "ADJ": wordnet.ADJ,
        "ADV": wordnet.ADV,
    }
    sw_list = stopwords.words("english")

    lem = WordNetLemmatizer()

    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    # pos-tag
    tagged = nltk.pos_tag(tokens, tagset="universal")
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    # unique the list
    tagged = list(set(tagged))
    return tagged


def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)
    # let's get senses for each
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]

    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions


def get_top_sense(words, sense_list):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    val, sense = max(
        (len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list
    )
    return val, sense


def original_lesk(
    context_sentence, ambiguous_word, pos=None, synsets=None, majority=False
):
    context_senses = get_sense_definitions(
        set(context_sentence) - set([ambiguous_word])
    )
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []
    # print(synsets)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        # We remove 0 scores senses without overlapping
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense


##GRAPH BASED
def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense


def lesk_similarity(
    context_sentence,
    ambiguous_word,
    similarity="resnik",
    pos=None,
    synsets=None,
    majority=True,
):
    context_senses = get_sense_definitions(
        set(context_sentence) - set([ambiguous_word])
    )

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None

    scores = []

    # Here you may have some room for improvement
    # For instance instead of using all the definitions from the context
    # you pick the most common one of each word (i.e. the first)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)

    return best_sense


def pedersen(
    context_sentence,
    ambiguous_word,
    similarity="resnik",
    pos=None,
    synsets=None,
    threshold=0.1,
):
    context_senses = get_sense_definitions(
        set(context_sentence) - set([ambiguous_word])
    )

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None

    synsets_scores = {}
    for ss_tup in synsets:
        ss = ss_tup[0]
        if ss not in synsets_scores:
            synsets_scores[ss] = 0
        for senses in context_senses:
            scores = []
            for sense in senses[1]:
                if similarity == "path":
                    try:
                        scores.append((sense[0].path_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "lch":
                    try:
                        scores.append((sense[0].lch_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "wup":
                    try:
                        scores.append((sense[0].wup_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "resnik":
                    try:
                        scores.append((sense[0].res_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "lin":
                    try:
                        scores.append((sense[0].lin_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "jiang":
                    try:
                        scores.append((sense[0].jcn_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                else:
                    print("Similarity metric not found")
                    return None
            value, sense = max(scores)
            if value > threshold:
                synsets_scores[sense] = synsets_scores[sense] + value

    values = list(synsets_scores.values())
    senses = list(synsets_scores.keys())
    best_sense_id = values.index(max(values))
    return senses[best_sense_id]


def collocational_features(inst, ngram_window=NGRAM_WINDOW):
    p = inst.position
    feats_dict = {
        "w-2_word": "NULL" if p < 2 else inst.context[p - 2][0],
        "w-1_word": "NULL" if p < 1 else inst.context[p - 1][0],
        "w+1_word": "NULL" if len(inst.context) - 1 < p + 1 else inst.context[p + 1][0],
        "w+2_word": "NULL" if len(inst.context) - 1 < p + 2 else inst.context[p + 2][0],
        "POS-tags": inst.context[p][1],
    }
    # Computing raw string
    sent_before = [
        inst.context[p - i - 1][0]
        for i in reversed(range(ngram_window - 1))
        if p > i + 1
    ]
    sent_after = [
        inst.context[p + i + 1][0]
        for i in (range(ngram_window - 1))
        if len(inst.context) - 1 > p + 1
    ]
    word = [inst.context[p][0]]
    sent_for_ngrams = " ".join(sent_before + word + sent_after)
    add_dict = {}

    # Computing ngrams from raw string
    for i in range(ngram_window):
        values = []
        key_name = str(i + 1) + "-gram"
        value_with_tuples = ngrams(nltk.word_tokenize(sent_for_ngrams), i + 1)
        for item in value_with_tuples:
            value_str = " ".join(item)
            values.append(value_str)
        add_dict.update({key_name: values})

    # Updating the features dict with the ngram dictionary
    feats_dict.update(add_dict)
    return feats_dict


def run_experiment(data, lbls, stratified_split, mapping, synsets, method="pedersen"):
    exp_scores = {}
    exp_scores["precision"] = []
    exp_scores["accuracy"] = []
    exp_scores["recall"] = []
    exp_scores["f_measure"] = []

    for train_index, test_index in stratified_split.split(data, lbls):
        # print(test_index)
        refs, hyps, refs_list, hyps_list = get_hyps(
            test_index, data, lbls, mapping, synsets, method
        )

        acc = round(accuracy(refs_list, hyps_list), 3)
        exp_scores["accuracy"].append(acc)
        for cls in hyps.keys():
            if refs[cls] == set():
                refs[cls].add(-1)
            if hyps[cls] == set():
                hyps[cls].add(-1)
            p = round(precision(refs[cls], hyps[cls]), 3)
            r = round(recall(refs[cls], hyps[cls]), 3)
            f = round(f_measure(refs[cls], hyps[cls], alpha=1), 3)

            exp_scores["precision"].append(p)
            exp_scores["recall"].append(r)
            exp_scores["f_measure"].append(f)

    print(
        f"{method} precision: {sum(exp_scores['precision'])/len(exp_scores['precision'])}"
    )
    print(f"{method} recall: {sum(exp_scores['recall'])/len(exp_scores['recall'])}")
    print(
        f"{method} f_measure: {sum(exp_scores['f_measure'])/len(exp_scores['f_measure'])}"
    )
    print(
        f"{method} accuracy: {sum(exp_scores['accuracy'])/len(exp_scores['accuracy'])}"
    )
    print("")


def get_hyps(test_index, data, lbls, mapping, synsets, method):
    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []
    for index in test_index:
        if method == "pedersen":
            hyp = pedersen(
                data[index].split(), "interest", similarity="path", synsets=synsets
            ).name()
        elif method == "lesk":
            hyp = original_lesk(
                data[index].split(), "interest", synsets=synsets, majority=True
            ).name()
        else:
            print("specify another method, options: [pedersen, lesk]")

        ref = mapping[lbls[index]]

        # for precision, recall, f-measure
        refs[ref].add(index)
        hyps[hyp].add(index)

        # for accuracy
        refs_list.append(ref)
        hyps_list.append(hyp)

    return refs, hyps, refs_list, hyps_list
