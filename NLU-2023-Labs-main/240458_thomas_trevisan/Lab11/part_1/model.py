import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import T5ForConditionalGeneration, T5Config


class SubjectivityBert(nn.Module):
    def __init__(self, model_name, dropout_prob=0):
        super(SubjectivityBert, self).__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.classification = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, attention_masks=None, labels=None, n_chunks=None):
        bert_output = self.bert(input, attention_mask=attention_masks)
        sequence_output = bert_output[0]
        # Pooled is what i need
        pooled_output = bert_output[1]
        pooled_output_dropped = self.dropout(pooled_output)

        output = self.classification(pooled_output_dropped)

        return output


class PolarityT5(nn.Module):
    def __init__(self, model_name, dropout_prob=0):
        super(PolarityT5, self).__init__()
        config = T5Config.from_pretrained(model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.classification = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, attention_masks=None, labels=None, n_chunks=None):
        t5_output = self.t5(
            input,
            attention_mask=attention_masks,
            labels=labels,
            output_hidden_states=True,
        )

        indices = []
        for i, index in enumerate(n_chunks):
            indices.append(sum(n_chunks[:i]))
        indices.append(sum(n_chunks))

        decoder_hidden_states = t5_output["decoder_hidden_states"]
        encoder_last_hidden_states = t5_output["encoder_last_hidden_state"]
        pooled_output = encoder_last_hidden_states.mean(dim=1)

        pooled_output = torch.stack(
            [
                torch.tensor(pooled_output[n : indices[i + 1]]).mean(dim=0)
                for i, n in enumerate(indices[:-1])
            ]
        )
        # Pooled is what i need
        pooled_output_dropped = self.dropout(pooled_output)

        output = self.classification(pooled_output_dropped)
        return output


"""
class PolarityLSTM(nn.Module):
  def __init__(self,embedding_dim, hidden_dim, vocab_size, n_layer=1, dropout_prob=0, pad_index = 100):
    super(PolarityLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim,  padding_idx=pad_index)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.classification = nn.Linear(hidden_dim, 1)
    self.emb_dropout = nn.Dropout(dropout_prob)

  def forward(self, inputs, seq_lengths):
      embedded = self.embedding(inputs)
      embedded = self.emb_dropout(embedded)
      embedded = embedded.permute(1,0,2)# we need seq len first -> seq_len X batch_size X emb_size
      # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
      packed_input = pack_padded_sequence(embedded, seq_lengths.cpu().numpy())
      #Encode
      packed_output, _ = self.lstm(packed_input)
      # Unpack the sequence
      lstm_output, input_sizes = pad_packed_sequence(packed_output)

      logits = self.classification(lstm_output[-1, :, :])  # Use the last LSTM output for classification
      return logits
"""
