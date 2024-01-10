import torch
import math
import sentencepiece
import logging
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from itertools import chain

from transformers import T5Tokenizer
import random
from tqdm import tqdm
from nltk.corpus import subjectivity
from nltk.corpus import movie_reviews

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# SUBJECTIVITY STUFF
class SubjectivityData(Dataset):
    def __init__(self, data):
        self.data = [sent[0] for sent in data]
        self.labels = [sent[1] for sent in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = " ".join(self.data[idx])
        label = self.labels[idx]

        # Process the sample and return it
        return sample, label


def subjectivity_collate_fn(batch, model_name):
    # Tokenize and pad the texts in the batch
    lengths = [len(text.split()) for text, label in batch]
    sents = [sent for sent, label in batch]
    labels = [label for text, label in batch]
    # +2 is for the CLS and SEP tokens
    max_length = max(lengths) + 2
    tokenizer = BertTokenizer.from_pretrained(model_name)

    encoded_inputs = tokenizer.batch_encode_plus(
        sents, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    labels = torch.Tensor(labels)

    sample = {"input_ids": input_ids, "mask": attention_mask, "labels": labels}
    # Return the tokenized and padded batch
    return sample


def get_subjectivity_data():
    label_mapping = {"subj": 0, "obj": 1}

    subj_docs = [
        (sent, label_mapping["subj"])
        for sent in subjectivity.sents(categories="subj")[:]
    ]
    obj_docs = [
        (sent, label_mapping["obj"]) for sent in subjectivity.sents(categories="obj")[:]
    ]
    sents = subj_docs + obj_docs
    random.shuffle(sents)

    dataset = SubjectivityData(sents)

    return dataset


# POLARITY STUFF
# Custom Dataset class example
class PolarityData(Dataset):
    def __init__(self, data):
        self.data_sents = [sent[0] for sent in data]
        self.data = [list(chain.from_iterable(review)) for review in self.data_sents]
        self.labels = [sent[1] for sent in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data[idx]
        label = self.labels[idx]

        # Process the sample and return it
        return review, label


def polarity_collate_fn(batch, model_name):
    # Tokenize reviews

    # Get the labels
    sents = [" ".join(sent) for sent, label in batch]
    labels = [label for text, label in batch]

    # Get the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    eos_token = tokenizer.eos_token
    # Tokenize without adding special tokens, encoded inputs are longer than 512 tokens
    encoded_inputs = tokenizer.batch_encode_plus(sents, add_special_tokens=False)

    # Tokenize labels for decoder input ids, T5 is a encoder-decoder model
    neg_id = tokenizer("negative " + eos_token)["input_ids"]  # 2841
    pos_id = tokenizer("positive " + eos_token)["input_ids"]  # 1465
    mapping_for_T5 = {0: neg_id, 1: pos_id}
    decoder_ids = [mapping_for_T5[label] for label in labels]

    # Get input ids and attention mask, currently not padded so AM all 1s
    input_ids = encoded_inputs["input_ids"]
    attention_masks = encoded_inputs["attention_mask"]

    # Compute how many chunks per review
    chunks_len = 510
    number_of_chunks = [math.ceil(len(input) / chunks_len) for input in input_ids]
    # Compute the indices where a review begins and ends

    # Split reviews in chunks
    split_sents = [
        input[i : i + chunks_len]
        for input in input_ids
        for i in range(0, len(input), chunks_len)
    ]
    attention_masks = [
        mask[i : i + chunks_len]
        for mask in attention_masks
        for i in range(0, len(mask), chunks_len)
    ]
    new_split_sents = []
    new_attention_masks = []
    # Add special tokens and padding to last
    for i, split_sent in enumerate(split_sents):
        if len(split_sent) < 510:
            new_split_sents.append(split_sent + [pad_id] * (510 - len(split_sent)))
            new_attention_masks.append(
                attention_masks[i] + [0] * (510 - len(attention_masks[i]))
            )
        else:
            new_split_sents.append(split_sent)
            new_attention_masks.append(attention_masks[i])
        # Unlike Bert, T5 has only </s> token for end of sentences
        new_split_sents[i].insert(0, eos_id)
        new_split_sents[i].insert(511, eos_id)
        new_attention_masks[i].insert(0, 1)
        new_attention_masks[i].insert(511, 1)

    input_ids = torch.tensor(new_split_sents)
    attention_masks = torch.tensor(new_attention_masks)
    labels = torch.tensor(labels)
    decoder_ids = torch.LongTensor(
        [id for i, id in enumerate(decoder_ids) for _ in range(number_of_chunks[i])]
    )
    number_of_chunks = torch.tensor(number_of_chunks)
    sample = {
        "input_ids": input_ids,
        "mask": attention_masks,
        "labels": labels,
        "decoder_ids": decoder_ids,
        "n_chunks": number_of_chunks,
    }
    # Return the tokenized and padded batch
    return sample


def get_polarity_data(rev_neg, rev_pos):
    polarity_label_mapping = {"neg": 0, "pos": 1}
    pos_revs = [(sent, polarity_label_mapping["neg"]) for sent in rev_neg]
    neg_revs = [(sent, polarity_label_mapping["pos"]) for sent in rev_pos]
    revs = pos_revs + neg_revs
    random.shuffle(revs)

    movie_dataset = PolarityData(revs)

    return movie_dataset


def remove_objective_sentences(review, model):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()
    lengths = [len(sent) for sent in review]
    max_length = max(lengths)
    review = [" ".join(sentence) for sentence in review]
    with torch.no_grad():
        encoded_inputs = tokenizer.batch_encode_plus(
            review,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        new_rev = []
        # If close to 0 then subj, if close to 1 then obj
        subjectivity = model(input_ids, attention_mask).squeeze()

        # Problem on iterating over a 0-d tensor with a single value
        if len(review) != 1:
            for index, value in enumerate(subjectivity):
                if value < 0.5:
                    new_rev.append(review[index].split())
        else:
            if subjectivity.item() < 0.5:
                new_rev.append(review[0])
    return new_rev


def get_subjective_dataset(sents, detection_model):
    subjective_movie_reviews = []
    for rev in tqdm(sents):
        new_rev = remove_objective_sentences(rev, detection_model)
        if new_rev:
            subjective_movie_reviews.append(new_rev)
    # subjective_dataset = PolarityData(subjective_movie_reviews)
    return subjective_movie_reviews


# OLD, USED WHEN I HAD LSTM MODEL
"""
def polarity_collate_fn(batch , lang):

    def merge(batch):
      #sentences is a list of reviews(lists)
      lengths = [len(review) for review in batch]
      max_len = 1 if max(lengths)==0 else max(lengths)
      # Pad token is 100 in our case
      # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
      # batch_size X maximum length of a sequence
      padded_seqs = torch.LongTensor(len(batch),max_len).fill_(100)
      for i, review in enumerate(batch):
          end = lengths[i]
          padded_seqs[i, :end] = torch.LongTensor(review) # We copy each sequence into the matrix
      # print(padded_seqs)
      padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
      return padded_seqs, lengths

    def convert_to_ids(review, lang):
      tmp_review = []
      for word in review:
        tmp_review.append(lang.vocab[word])
      return tmp_review

    batch.sort(key = lambda x: len(x[0]),  reverse=True)
    labels = [label for review, label in batch]
    reviews = [review for review, label in batch]
    #batch is a list of lists of words
    #a review is a list of words
    reviews_ids = []
    for review in reviews:
      reviews_ids.append(convert_to_ids(review, lang))

    reviews, seq_len = merge(reviews_ids)

    seq_len = torch.LongTensor(seq_len).to(device)
    labels = torch.LongTensor(labels).to(device)
    sample = {'reviews':reviews , 'seq_len':seq_len, 'labels':labels}
    return sample

class Lang:
  def __init__(self, data, cutoff=0):
    words = list(chain.from_iterable(data.data))
    self.vocab = self.word2id(words, cutoff)
    self.id2word = {v:k for k, v in self.vocab.items()}

  def word2id(self, words, cutoff, unknown=True):
        vocab = {'PAD': 100}
        if unknown:
            vocab['UNK'] = 103
        count = Counter(words)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
"""
