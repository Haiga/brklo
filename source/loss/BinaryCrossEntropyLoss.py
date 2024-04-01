import torch
from torch import nn


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, params):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.params = params
        self.criterion = nn.BCELoss()()

    def forward(self, pred_score, true_score):
        return self.criterion(pred_score, true_score)
