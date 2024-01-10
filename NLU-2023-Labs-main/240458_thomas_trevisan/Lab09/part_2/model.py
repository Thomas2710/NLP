import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Variational_Dropout(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

    # Computes a mask for an element of x, and then use the same mask for all elements of x
    def forward(self, x):
        if self.training:
            m = torch.empty(1, x.size(1), x.size(2)).bernoulli(1 - self.dropout) / (
                1 - self.dropout
            )
            mask = m.expand_as(x).to(device)
            return mask * x
        else:
            return x


class LM_LSTM(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        emb_dropout,
        out_dropout,
        hidden_dropout,
        weight_tying,
        variational,
        pad_index=0,
        n_layers=3,
    ):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = torch.nn.LSTM(
            emb_size, hidden_size, num_layers=n_layers, dropout=hidden_dropout
        )
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        if variational:
            self.out_dropout = Variational_Dropout(out_dropout)
            self.emb_dropout = Variational_Dropout(emb_dropout)
        else:
            self.out_dropout = nn.Dropout(p=out_dropout)
            self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.hidden_dropout = nn.Dropout(p=hidden_dropout)

        if weight_tying:
            if emb_size != hidden_size:
                raise ValueError(
                    "When using the tied flag, hidden size must be equal to emb_size"
                )
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        tmp_emb = self.embedding(input_sequence)
        emb = self.emb_dropout(tmp_emb)
        tmp_output, (h_n, c_n) = self.lstm(emb)
        tmp_output = self.out_dropout(tmp_output)
        output = self.output(tmp_output).permute(0, 2, 1)
        return output

    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        # Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(self.cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return (indexes, top_scores)

    def cosine_similarity(self, v, w):
        v_norm = norm(v)
        w_norm = norm(w)
        denom = v_norm * w_norm
        num = np.dot(v, w)
        return num / denom
