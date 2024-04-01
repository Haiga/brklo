from pytorch_metric_learning.losses import TripletMarginLoss
from torch import nn
from source.miner.RelevanceMiner import RelevanceMiner
from pytorch_metric_learning.utils import common_functions as c_f


class RiskLoss(nn.Module):

    def __init__(self, params):
        c_f.check_shapes = lambda x, y: None
        super(RiskLoss, self).__init__()
        self.miner = RelevanceMiner(params.miner)
        self.criterion = TripletMarginLoss()

    def forward(self, query_idx, query_rpr, passage_idx, passage_rpr):
        print("---------------------------")
        print(query_idx, query_rpr, passage_idx, passage_rpr)
        print("----------------------flag-------------")
        miner_outs = self.miner.mine(query_idx=query_idx, passage_idx=passage_idx)
        return self.criterion(query_rpr, None, miner_outs, passage_rpr, None)