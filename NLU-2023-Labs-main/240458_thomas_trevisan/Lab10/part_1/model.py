import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):
    def __init__(
        self,
        hid_size,
        out_slot,
        out_int,
        emb_size,
        vocab_len,
        n_layer=2,
        dropout_prob=0,
        bidirectional=False,
        pad_index=0,
    ):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (ouput size for intent class)
        # emb_size = word embedding size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(
            emb_size, hid_size, n_layer, bidirectional=self.bidirectional
        )
        if bidirectional:
            self.slot_out = nn.Linear(2 * hid_size, out_slot)
            self.intent_out = nn.Linear(2 * hid_size, out_int)
        else:
            self.slot_out = nn.Linear(hid_size, out_slot)
            self.intent_out = nn.Linear(hid_size, out_int)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(
            utterance
        )  # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = self.dropout(utt_emb)
        utt_emb = utt_emb.permute(
            1, 0, 2
        )  # we need seq len first -> seq_len X batch_size X emb_size
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        # Apply dropout to utt_encoded
        utt_encoded = self.dropout(utt_encoded)
        # Get the last hidden state
        if self.bidirectional:
            last_hidden = torch.cat(
                (last_hidden[-1, :, :], last_hidden[-2, :, :]), dim=1
            )
        else:
            last_hidden = last_hidden[-1, :, :]
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: seq_len, batch size, classes
        slots = slots.permute(1, 2, 0)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
