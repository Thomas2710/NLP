def my_statistics(chars, words, sents):
    total_chars = len(chars)
    total_words = len(words)
    total_sents = len(sents)
    word_lens = [len(word) for word in words]
    sent_lens = [len(sentence) for sentence in sents]
    chars_in_sents = [sum(len(word) for word in sent) for sent in sents]
    # chars_in_sents = [len(''.join(sent)) for sent in sents]

    word_per_sent = round(len(words) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    # char_per_sent = round(sum(word_lens) / len(sents))
    # Alternatively
    char_per_sent = round(sum(chars_in_sents) / len(sents))

    longest_sentence = max(sent_lens)
    longest_word = max(word_lens)
    shortest_sentence = min(sent_lens)
    shortest_word = min(word_lens)

    return (
        total_chars,
        total_words,
        total_sents,
        word_per_sent,
        char_per_word,
        char_per_sent,
        longest_sentence,
        longest_word,
        shortest_sentence,
        shortest_word,
    )


def nbest(d, n=1):
    """
    get n max values from a dict
    :param d: input dict (values are numbers, keys are stings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])
