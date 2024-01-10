from conll import evaluate
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
import es_core_news_sm

nlp = es_core_news_sm.load()
nlp.tokenizer = Tokenizer(
    nlp.vocab
)  # to use white space tokenization (generally a bad idea for unknown data)


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def sent2spacy_features(sent, mode=" baseline", context=0):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))

    feats = []
    for token in spacy_sent:
        token_feats = {
            "bias": 1.0,
            "word.lower()": token.lower_,
            "pos": token.pos_,
            "lemma": token.lemma_,
        }
        suffix_token_feats = {
            "token[-3:]": token.suffix_,
            "token[-3:]_hash": token.suffix,
        }
        tutorial_token_feats = {
            "word" + mode + ".isupper()": token.is_upper,
            "word.istitle()": token.is_title,
            "word.isdigit()": token.is_digit,
            "postag[:2]": token.pos_[:2],
        }
        if mode == "suffix":
            token_feats.update(suffix_token_feats)

        if mode == "conll_tutorial":
            token_feats.update(suffix_token_feats)
            token_feats.update(tutorial_token_feats)

            if context != 0:
                for i in range(context):
                    if not token.is_sent_start:
                        context_token = token.nbor(-(i + 1))
                        context_postag = context_token.pos_
                        token_feats.update(
                            {
                                "-" + i + ":word.lower()": context_token.lower_,
                                "-" + i + ":word.istitle()": context_token.is_title,
                                "-" + i + ":word.isupper()": context_token.is_upper,
                                "-" + i + ":postag": context_postag,
                                "-" + i + ":postag[:2]": context_postag[:2],
                            }
                        )
                    else:
                        token_feats["BOS"] = True

                    if not token.is_sent_end:
                        context_token = token.nbor(i + 1)
                        context_postag = context_token.pos_
                        token_feats.update(
                            {
                                "+" + i + ":word.lower()": context_token.lower_,
                                "+" + i + ":word.istitle()": context_token.is_title,
                                "+" + i + ":word.isupper()": context_token.is_upper,
                                "+" + i + ":postag": context_postag,
                                "+" + i + ":postag[:2]": context_postag[:2],
                            }
                        )
                    else:
                        token_feats["EOS"] = True

        feats.append(token_feats)
    return feats


def fit_predict(crf, trn_data, trn_label, tst_data):
    # workaround for scikit-learn 1.0
    try:
        crf.fit(trn_data, trn_label)
    except AttributeError:
        pass
    pred = crf.predict(tst_data)
    hyp = [
        [(tst_data[i][j], token) for j, token in enumerate(tokens)]
        for i, tokens in enumerate(pred)
    ]
    return hyp


def evaluation(tst_sents, hyp):
    results = evaluate(tst_sents, hyp)
    pd_tbl = pd.DataFrame().from_dict(results, orient="index")
    pd_tbl.round(decimals=3)
    return pd_tbl
