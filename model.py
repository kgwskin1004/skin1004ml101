import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        last_output = lstm_out[0][-1]
        predictions = self.linear(last_output)
        return predictions


class InfluenceModel(nn.Module):
    def __init__(self):
        super(InfluenceModel, self).__init__()
        # Parameters (a, b, c, d, e)
        self.params = nn.Parameter(torch.randn(5))

    def forward(self, x, cost):
        # x is a tensor of shape [batch_size, 5] containing [view, save, comment, like, share]
        logs = torch.log1p(x / cost.unsqueeze(1))  # Add 1 and take log base e
        # Multiply element-wise by the parameters and sum across columns to get a single score per example

        influence_score = torch.sum(logs * self.params, dim=1)
        return influence_score


class Transformer(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(Transformer, self).__init__()

        raise NotImplementedError("Transformer model not implemented yet")

    def forward(self, input_seq):
        raise NotImplementedError("Transformer model not implemented yet")
