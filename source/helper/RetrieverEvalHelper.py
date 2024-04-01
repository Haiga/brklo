import logging
import pickle
from pathlib import Path

import nmslib
import pandas as pd
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

from source.helper.Helper import Helper


class RetrieverEvalHelper(Helper):
    def __init__(self, params):
        super(RetrieverEvalHelper, self).__init__()
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.metrics = self._get_metrics()

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
            metrics.append(f"{metric}@{self.params.eval.num_nearest_neighbors}")

        return metrics

    def _load_predictions(self, fold_idx):
        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold_idx}/").glob("*.prd")
        )
        query_predictions = []
        passage_predictions = []

        for path in tqdm(predictions_paths, desc="Loading predictions"):
            for prediction in torch.load(path):

                if prediction["modality"] == "query":
                    query_predictions.append({
                        "query_idx": prediction["query_idx"],
                        "query_rpr": prediction["query_rpr"]
                    })

                elif prediction["modality"] == "passage":
                    passage_predictions.append({
                        "passage_idx": prediction["passage_idx"],
                        "passage_rpr": prediction["passage_rpr"]
                    })

        logging.info(f"Added {len(query_predictions)} queries")
        logging.info(f"Added {len(passage_predictions)} passages\n")

        return query_predictions, passage_predictions

    def init_index(self, passage_predictions):
        added = 0
        index = nmslib.init(method='hnsw', space='l2')
        for prediction in tqdm(passage_predictions, desc="Adding data to index"):
            index.addDataPoint(id=prediction["passage_idx"], data=prediction["passage_rpr"])
            added += 1

        index.createIndex(
            index_params=OmegaConf.to_container(self.params.eval.index),
            print_progress=False
        )
        logging.info(f"Added {added} passages.")
        return index

    def retrieve(self, index, query_predictions, num_labels):
        # retrieve
        searched = 0
        ranking = {}
        #index.setQueryTimeParams({'efSearch': 2048})

        for prediction in tqdm(query_predictions, desc="Searching"):
            query_idx = prediction["query_idx"]
            retrieved_ids, distances = index.knnQuery(prediction["query_rpr"], k=num_labels)
            for passage_idx, distance in zip(retrieved_ids, distances):
                if f"query_{query_idx}" not in ranking:
                    ranking[f"query_{query_idx}"] = {}
                score = 1.0 / (distance + 1e-9)
                if f"passage_{passage_idx}" in ranking[f"query_{query_idx}"]:
                    if score > ranking[f"query_{query_idx}"][f"passage_{passage_idx}"]:
                        ranking[f"query_{query_idx}"][f"passage_{passage_idx}"] = score
                else:
                    ranking[f"query_{query_idx}"][f"passage_{passage_idx}"] = score
            searched += 1
        logging.info(f"Searched {searched} texts.")
        return ranking

    def _get_ranking(self, text_predictions, label_predictions, num_passages):
        # index data
        index = self.init_index(label_predictions)
        # retrieve
        return self.retrieve(index, text_predictions, num_passages)

    def perform_eval(self):
        results = []
        rankings = {}
        for fold_idx in self.params.data.folds:
            rankings[fold_idx] = {}
            logging.info(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            query_predictions, passage_predictions = self._load_predictions(fold_idx)
            ranking = self._get_ranking(query_predictions, passage_predictions,
                                        num_passages=self.params.eval.num_nearest_neighbors)
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
