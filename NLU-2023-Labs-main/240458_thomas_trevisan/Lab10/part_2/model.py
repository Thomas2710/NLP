import transformers
from transformers import BertModel, BertConfig
import torch
from torch import nn as nn

BERT_MODEL = "bert-base-uncased"


class JointIntentAndSlotFillingModel(nn.Module):
    def __init__(self, out_slot, out_int, dropout_prob=0.1, model_name=BERT_MODEL):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, out_int)
        self.slot_classifier = nn.Linear(config.hidden_size, out_slot)

    def forward(self, input, attention_masks):
        bert_output = self.bert(input, attention_mask=attention_masks)
        sequence_output = bert_output[0]
        pooled_output = bert_output[1]

        sequence_output_dropped = self.dropout(sequence_output)
        slots_predicted = self.slot_classifier(sequence_output_dropped)

        pooled_output_dropped = self.dropout(pooled_output)
        intent_predicted = self.intent_classifier(pooled_output_dropped)

        # Slot size: batch size, seq_len, classes
        slots_predicted = slots_predicted.permute(
            0, 2, 1
        )  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len

        return slots_predicted, intent_predicted
