import torch
from torch import nn
from pytorch_metric_learning import miners, losses
from source.distance.DotProductDistance import DotProductDistance
from source.miner.RelevanceMiner import RelevanceMiner
from pytorch_metric_learning.utils import common_functions as c_f


class NTXentLoss(nn.Module):

    def __init__(self, params):
        c_f.check_shapes = lambda x, y: None
        super(NTXentLoss, self).__init__()
        self.miner = RelevanceMiner(params.miner)
        self.criterion = losses.NTXentLoss(temperature=params.criterion.temperature, distance=DotProductDistance())

    def forward(self, query_idx, query_rpr, passage_idx, passage_rpr):
        """
        Computes the NTXentLoss.
        """
        miner_outs = self.miner.mine(query_idx=query_idx, passage_idx=passage_idx)
        return self.criterion(query_rpr, None, miner_outs, passage_rpr, None)
