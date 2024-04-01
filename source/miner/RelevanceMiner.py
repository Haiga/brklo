import pickle

import torch
from pytorch_metric_learning.miners import BaseMiner


class RelevanceMiner(BaseMiner):

    def __init__(self, params):
        super().__init__()
        with open(f"{params.relevance_map.dir}relevance_map.pkl", "rb") as relevance_map_file:
            self.relevance_map = pickle.load(relevance_map_file)


    def mine(self, query_idx, passage_idx):
        a1, p, a2, n = [], [], [], []
        for i, text_idx in enumerate(query_idx.tolist()):
            for j, label_idx in enumerate(passage_idx.tolist()):
                if label_idx >= 0:
                    if label_idx in self.relevance_map[text_idx]:
                        a1.append(i)
                        p.append(j)
                    else:
                        a2.append(i)
                        n.append(j)

        return torch.tensor(a1, device=query_idx.device), torch.tensor(p, device=query_idx.device), torch.tensor(a2, device=query_idx.device), torch.tensor(n, device=query_idx.device)

    def output_assertion(self, output):
        pass
