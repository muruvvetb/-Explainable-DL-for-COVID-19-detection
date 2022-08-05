import torch as torch
from collections import OrderedDict


class Classifier(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout_rate=0.5):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size if hidden_size is not None else (in_size + out_size) // 2
        self.dropout_rate = dropout_rate

        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(self.in_size, self.hidden_size)),
            ('dropout', torch.nn.Dropout(self.dropout_rate)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(self.hidden_size, self.out_size)),
            # ('relu2', torch.nn.ReLU())
        ]))

    def forward(self, x):
        return self.classifier(x)
