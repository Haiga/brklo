import pickle

import torch
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class RetrieverMetric(Metric):
    def __init__(self, params):
        super(RetrieverMetric, self).__init__(compute_on_cpu=True)
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.ranking = {}

    def _load_relevance_map(self):
        with open(f"{self.params.relevance_map.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"passage_{label_idx}"] = 1.0
            relevance_map[f"query_{text_idx}"] = d
        return relevance_map

    def update(self, query_ids, query_rprs, passage_ids, passage_rprs):
        scores = torch.einsum("ab,cb->ac", query_rprs, passage_rprs) * self.params.scale

        for i, query_idx in enumerate(query_ids.tolist()):
            for j, passage_idx in enumerate(passage_ids.tolist()):
                if f"query_{query_idx}" not in self.ranking:
                    self.ranking[f"query_{query_idx}"] = {}
                self.ranking[f"query_{query_idx}"][f"passage_{passage_idx}"] = scores[i][j].item() + self.ranking[
                    f"query_{query_idx}"].get(
                    f"passage_{passage_idx}", 0)

    def compute(self):
        m = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in self.ranking.keys()}),
            Run(self.ranking),
            ["mrr@1"]
        )
        return m

    def reset(self) -> None:
        self.ranking = {}
