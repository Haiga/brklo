import pickle
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class RerankerMetric(Metric):
    def __init__(self, params):
        super(RerankerMetric, self).__init__(compute_on_cpu=True)
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.ranking = {}

    def _load_relevance_map(self):
        with open(f"{self.params.relevance_map.dir}fit_relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for query_idx, passages_ids in data.items():
            d = {}
            for passage_idx in passages_ids:
                d[f"passage_{passage_idx}"] = 1.0
            relevance_map[f"query_{query_idx}"] = d
        return relevance_map

    def update(self, queries_ids, passages_ids, scores):
        for query_idx, passage_idx, score in zip(queries_ids.tolist(), passages_ids.tolist(), scores.tolist()):
            if f"query_{query_idx}" not in self.ranking:
                self.ranking[f"query_{query_idx}"] = {}
            self.ranking[f"query_{query_idx}"][f"passage_{passage_idx}"] = score + self.ranking[
                f"query_{query_idx}"].get(
                f"passage_{passage_idx}", 0)

    def compute(self):

        # eval
        m = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in self.ranking.keys()}),
            Run(self.ranking),
            ["mrr@1"]
        )
        return m

    def reset(self) -> None:
        self.ranking = {}
