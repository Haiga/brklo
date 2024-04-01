import logging
import pickle
from pathlib import Path

import nmslib
import pandas as pd
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate, fuse
from tqdm import tqdm

from source.helper.Helper import Helper


class RerankerEvalHelper(Helper):
    def __init__(self, params):
        super(RerankerEvalHelper, self).__init__()
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.metrics = self._get_metrics()

    def _load_predictions(self, fold_idx):
        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold_idx}/").glob("*.prd")
        )

        predictions = []

        for path in tqdm(predictions_paths, desc="Loading predictions"):
            for prediction in torch.load(path):
                predictions.append({
                    "query_idx": prediction["query_idx"],
                    "passage_idx": prediction["passage_idx"],
                    "true_score": prediction["true_score"],
                    "pred_score": prediction["pred_score"]
                })

        logging.info(f"Loaded {len(predictions)} predictions.")
        return predictions

    def _get_ranking(self, predictions):
        ranking = {}
        for prediction in tqdm(predictions, desc="Ranking"):
            query_idx = prediction["query_idx"]
            passage_idx = prediction["passage_idx"]
            score = prediction["pred_score"]
            if f"query_{query_idx}" not in ranking:
                ranking[f"query_{query_idx}"] = {}

            ranking[f"query_{query_idx}"][f"passage_{passage_idx}"] = score

        return ranking

    def perform_eval(self):
        results = []
        rankings = {}
        for fold_idx in self.params.data.folds:
            rankings[fold_idx] = {}
            logging.info(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            predictions = self._load_predictions(fold_idx)
            ranking = self._get_ranking(predictions)
            result = evaluate(
                Qrels(
                    {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                ),
                Run(ranking),
                self.metrics
            )
            result = {k: round(v, 3) for k, v in result.items()}
            result["fold_idx"] = fold_idx
            results.append(result)
            rankings[fold_idx] = ranking
            self.checkpoint_ranking(rankings[fold_idx], fold_idx)
            self._checkpoint_result(results, fold_idx)
