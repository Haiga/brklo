import torch
from torch import nn


class CrossEntropyLoss(nn.Module):

    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.params = params
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred_score, true_score):
        return self.criterion(pred_score, true_score)
