import torch
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from functools import partial

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_conll_file(file_path):
    output = []
    sentence = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            tagged_word = tuple(line.split())
            if not tagged_word:
                output.append(sentence)
                sentence = []
            else:
                sentence.append(tagged_word)
    return output


class Lang:
    def __init__(
        self, words, aspects, polarities, model_name="bert-base-uncased", cutoff=0
    ):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.PAD_ID = self.tokenizer.pad_token_id
        self.UNK_TOKEN = self.tokenizer.unk_token
        self.UNK_ID = self.tokenizer.unk_token_id
        self.SEP_TOKEN = self.tokenizer.sep_token
        self.SEP_ID = self.tokenizer.sep_token_id
        self.CLS_TOKEN = self.tokenizer.cls_token
        self.CLS_ID = self.tokenizer.cls_token_id

        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.aspect2id = self.a2id(aspects)
        self.id2aspect = {v: k for k, v in self.aspect2id.items()}
        self.polarity2id = self.p2id(polarities)
        self.id2polarity = {v: k for k, v in self.polarity2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {self.PAD_TOKEN: self.PAD_ID}
        vocab[self.CLS_TOKEN] = self.CLS_ID
        vocab[self.SEP_TOKEN] = self.SEP_ID
        if unk:
            vocab[self.UNK_TOKEN] = self.UNK_ID
        for word in elements:
            vocab[word] = self.tokenizer.convert_tokens_to_ids(word)
        count = Counter(elements)
        for k, v in sorted(count.items()):
            if v > cutoff:
                vocab[k] = self.tokenizer.convert_tokens_to_ids(k)
        return vocab

    def a2id(self, elements):
        vocab = {"PAD": 0, "O": 1}
        for elem in sorted(set(elements)):
            vocab[elem] = len(vocab)
        return vocab

    def p2id(self, elements):
        vocab = {}
        for elem in sorted(set(elements)):
            vocab[elem] = len(vocab)
        return vocab


class AspectBasedDataset(Dataset):
    def __init__(self, data, lang):
        self.sents = [
            [
                lang.word2id[word] if word in lang.word2id else lang.UNK_ID
                for word, tag in sent
            ]
            for sent in data
        ]
        self.aspects_sent = [
            [
                lang.aspect2id[tag.split("-")[0]] if tag != "O" else lang.aspect2id["O"]
                for word, tag in sent
            ]
            for sent in data
        ]
        self.polarity_sent = [
            [
                lang.polarity2id[tag.split("-")[1]]
                if tag != "O"
                else lang.polarity2id["O"]
                for word, tag in sent
            ]
            for sent in data
        ]
        self.lang = lang

        self.one_hot_pol_labels = []
        for element in self.polarity_sent:
            default_hot = [0] * len(
                self.lang.polarity2id
            )  # T2 is the pad and 'O' tokens
            for j, word in enumerate(element):
                if word != "O":
                    default_hot[word] = 1
            self.one_hot_pol_labels.append(default_hot)

        self.spans = []
        self.starts = []
        self.ends = []
        # Computes for every sentence the list of span intervals,
        # The mask of the beginning of the intervals
        # The mask of the ends of the intervals
        empty_id = self.lang.aspect2id["O"]
        for sent in self.aspects_sent:
            tmp_span = []
            tmp_start = []
            tmp_end = []
            start_index = None
            for i, word in enumerate(sent):
                if word == empty_id and start_index is None:
                    # print('empty')
                    tmp_start.append(0)
                    tmp_end.append(0)
                elif word != empty_id and start_index is None:
                    # print('beginning')
                    start_index = i
                    tmp_start.append(1)
                    tmp_end.append(0)
                elif word != empty_id and start_index is not None:
                    # print(f'in the middle with aspect {word}')
                    tmp_start.append(0)
                    tmp_end.append(0)
                elif word == empty_id and start_index is not None:
                    # print('inserting end')
                    tmp_start.append(0)
                    tmp_end.append(0)
                    tmp_end[i - 1] = 1
                    tmp_span.append((start_index, i - 1))
                    start_index = None
            self.starts.append(tmp_start)
            self.ends.append(tmp_end)
            self.spans.append(tmp_span)

        """
        Doing by words, wrong
        self.words = [word for sent in data for word, tag in sent]
        self.aspects = [tag.split('-')[0] for sent in data for word, tag in sent]
        self.polarity = [tag.split('-')[1] if tag != 'O' else None for sent in data for word, tag in sent]
        """

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        utterance = self.lang.tokenizer.build_inputs_with_special_tokens(
            self.sents[idx]
        )
        aspect = self.lang.tokenizer.build_inputs_with_special_tokens(
            self.aspects_sent[idx]
        )
        polarity = self.lang.tokenizer.build_inputs_with_special_tokens(
            self.polarity_sent[idx]
        )
        one_hot = self.one_hot_pol_labels[idx]
        start = (
            [self.lang.aspect2id["O"]] + self.starts[idx] + [self.lang.aspect2id["O"]]
        )
        end = [self.lang.aspect2id["O"]] + self.ends[idx] + [self.lang.aspect2id["O"]]

        utterance = torch.tensor(utterance)
        aspect = torch.tensor(aspect)
        polarity = torch.tensor(polarity)
        start = torch.tensor(start)
        end = torch.tensor(end)
        one_hot = torch.tensor(one_hot)

        # Process the sample and return it
        return utterance, aspect, polarity, start, end, one_hot


def collate_fn(batch, model_name):
    def pad(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        attention_masks = torch.FloatTensor(
            [
                [1 for i in range(len(seq))] + [0 for i in range(max_len - len(seq))]
                for seq in sequences
            ]
        )
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(0)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
        return padded_seqs, attention_masks, lengths

    # Compute attention mask
    utterances = [
        utterance for utterance, aspect, polarity, start, end, one_hot in batch
    ]
    aspects = [aspect for utterance, aspect, polarity, start, end, one_hot in batch]
    polarities = [
        polarity for utterance, aspect, polarity, start, end, one_hot in batch
    ]
    starts = [start for utterance, aspect, polarity, start, end, one_hot in batch]
    ends = [end for utterance, aspect, polarity, start, end, one_hot in batch]
    one_hots = torch.stack(
        [one_hot for utterance, aspect, polarity, start, end, one_hot in batch]
    )

    utterances, utt_mask, utt_len = pad(utterances)
    starts, _, _ = pad(starts)
    ends, _, _ = pad(ends)

    utterances = utterances.to(device)
    utt_mask = utt_mask.to(device)
    utt_len = torch.tensor(utt_len).to(device)
    starts = starts.to(device)
    ends = ends.to(device)
    one_hots = one_hots.to(device)

    sample = {
        "utterance": utterances,
        "utt_mask": utt_mask,
        "utt_len": utt_len,
        "start": starts,
        "end": ends,
        "one_hot": one_hots,
    }
    return sample
