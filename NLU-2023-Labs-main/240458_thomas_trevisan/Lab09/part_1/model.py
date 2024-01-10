import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine


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
        n_layers=1,
    ):
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = torch.nn.LSTM(
            emb_size, hidden_size, num_layers=n_layers, dropout=hidden_dropout
        )
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)
        self.out_dropout = nn.Dropout(p=out_dropout)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout)

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
