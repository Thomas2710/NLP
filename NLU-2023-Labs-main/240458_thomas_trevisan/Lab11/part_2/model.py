import torch
from torch import optim as optim
from torch import nn as nn
from transformers import BertModel, BertConfig


class SpanLoss(nn.Module):
    def __init__(self):
        super(SpanLoss, self).__init__()

    def forward(self, start_hyp, end_hyp, start_ref, end_ref):
        start_hyp = torch.log(start_hyp)
        end_hyp = torch.log(end_hyp)

        start_prod = torch.mul(start_hyp, start_ref)
        end_prod = torch.mul(end_hyp, end_ref)
        loss = -torch.sum(start_prod) - torch.sum(end_prod)
        return loss


class PolarityLoss(nn.Module):
    def __init__(self):
        super(PolarityLoss, self).__init__()

    # Ref should be a one hot label, like [0,0,0,1] if polarity of span in the last class
    def forward(self, hyp, ref):
        hyp = torch.log(hyp)
        prod = torch.mul(hyp, ref)
        loss = -torch.sum(prod)
        return loss


class SpanExtractor(nn.Module):
    def __init__(self, model_name, number_of_polarities):
        super(SpanExtractor, self).__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.softmax = nn.Softmax(dim=1)

        self.start_weight_vector = nn.Parameter(torch.randn(config.hidden_size, 1))
        self.end_weight_vector = nn.Parameter(torch.randn(config.hidden_size, 1))

        self.polarities = number_of_polarities
        self.tanh = nn.Tanh()
        self.span_reweighting = nn.Parameter(torch.randn(config.hidden_size, 1))
        self.first_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.second_linear = nn.Linear(config.hidden_size, number_of_polarities)

    def forward(self, inputs, masks, mode="span"):
        if mode == "span":
            bert_output = self.bert(inputs, attention_mask=masks)
            # batch_size, str_len, embedding
            sequence_output = bert_output[0]
            pooled_output = bert_output[1]

            start_output = torch.matmul(
                sequence_output, self.start_weight_vector
            ).squeeze()
            end_output = torch.matmul(sequence_output, self.end_weight_vector).squeeze()
            # (32,45) where 45 are the scores per index(also padding, weird)
            # Probabilities beyond mask should be zero
            start_output_soft = self.softmax(start_output)
            end_output_soft = self.softmax(end_output)

            return start_output, end_output, start_output_soft, end_output_soft
        else:
            bert_output = self.bert(inputs)
            sequence_output = bert_output[0]

            weights = torch.matmul(sequence_output, self.span_reweighting)
            weights = self.softmax(weights)
            span_representation = torch.mul(sequence_output, weights)
            span_representation = torch.sum(span_representation, dim=1)

            tmp_output = self.first_linear(span_representation)
            tmp_output = self.tanh(tmp_output)
            output = self.second_linear(tmp_output)
            soft_output = self.softmax(output)
            return output, soft_output
