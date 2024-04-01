from torch import nn


class RiskLoss(nn.Module):

    def __init__(self, params):
        pass

    def forward(self, query_idx, query_rpr, passage_idx, passage_rpr):
        print("---------------------------")
        print(query_idx, query_rpr, passage_idx, passage_rpr)
        print("----------------------flag-------------")
        pass
