import heapq
import logging
import pickle
from pathlib import Path

import pandas as pd
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, EarlyStopping, ModelCheckpoint
from ranx import evaluate, Qrels, Run, fuse
from transformers import AutoTokenizer


class Helper:
    def __int__(self, params):
        self.params = params

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.params.model.tokenizer.architecture
        )

    def get_logger(self, fold_idx):
        return loggers.WandbLogger(
            project=self.params.log.project,
            save_dir=self.params.log.dir,
            name=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}_exp"
        )

    def get_progress_bar_callback(self):
        return TQDMProgressBar(
            refresh_rate=self.params.trainer.progress_bar_refresh_rate,
            process_position=0
        )

    def get_lr_monitor(self):
        return LearningRateMonitor(logging_interval='step')

    def get_early_stopping_callback(self):
        return EarlyStopping(
            monitor='val_MRR',
            patience=self.params.trainer.patience,
            min_delta=self.params.trainer.min_delta,
            mode='max'
        )

    def get_model_checkpoint_callback(self, fold_idx):
        return ModelCheckpoint(
            monitor="val_MRR",
            dirpath=self.params.model_checkpoint.dir,
            filename=f"{self.params.model.name}_{self.params.data.name}_{fold_idx}",
            save_top_k=1,
            save_weights_only=True,
            mode="max"
        )

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for query_idx, passages_ids in data.items():
            d = {}
            for passage_idx in passages_ids:
                d[f"passage_{passage_idx}"] = 1.0
            relevance_map[f"query_{query_idx}"] = d
        return relevance_map

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
        return metrics

    def _load_samples(self):
        with open(f"{self.params.data.dir}samples.pkl", "rb") as samples_file:
            return pickle.load(samples_file)

    def _load_queries(self):
        with open(f"{self.params.data.dir}queries.pkl", "rb") as queries_file:
            return pickle.load(queries_file)

    def _load_passages(self):
        with open(f"{self.params.data.dir}passages.pkl", "rb") as passages_file:
            return pickle.load(passages_file)

    def _get_ids(self, fold_idx, split):
        with open(f"{self.params.data.dir}fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def _load_split_ids(self, fold_idx, split):
        with open(f"{self.params.data.dir}fold_{fold_idx}/{split}.pkl", "rb") as ids_file:
            return pickle.load(ids_file)

    def _min_max_normalize(self, labels_scores):
        values = list(labels_scores.values())
        max_value = max(values)
        min_value = min(values)
        return {key: (value - min_value) / (max_value - min_value) for key, value in labels_scores.items()}

    def _load_ranking(self, fold_idx):
        pass
        logging.info(f"Loading BM25 ranking.")

        with open(f"{self.params.ranking.dir}BM25/BM25_MSMARCO_RERANKING.rnk", "rb") as ranking_file:
            return pickle.load(ranking_file)

    def _slice_ranking(self, ranking, fold_idx, num_labels):
        sliced_ranking = {
            fold_idx: {}
        }
        for split in ["train", "val", "test"]:
            sliced_ranking[fold_idx][split] = {}
            for cls in ["tail", "head"]:
                sliced_ranking[fold_idx][split][cls] = {}
                for text_idx, labels_scores in ranking[fold_idx][split][cls].items():
                    sliced_ranking[fold_idx][split][cls][text_idx] = {k: v for k, v in
                                                                      heapq.nlargest(num_labels, labels_scores.items(),
                                                                                     key=lambda item: item[1])}
        return sliced_ranking

    def _fuse_rankings(self, ranking1, ranking2, fold_idx):
        fused_ranking = {
            fold_idx: {}
        }
        for split in ["train", "val", "test"]:
            fused_ranking[fold_idx][split] = {}
            for cls in ["tail", "head"]:
                r1 = ranking1[fold_idx][split][cls]
                r2 = ranking2[fold_idx][split][cls]
                fused_ranking[fold_idx][split][cls] = fuse(runs=[Run(r1), Run(r2)], norm="zmuv",
                                                           method="mnz").to_dict()

        return fused_ranking

    def _eval_ranking(self, rankings, fold_idx):
        results = []
        relevance_map = self._load_relevance_map()
        metrics = self._get_metrics()
        for split in ["train", "val", "test"]:
            for cls in ["tail", "head"]:
                ranking = rankings[fold_idx][split][cls]
                result = evaluate(
                    Qrels(
                        {key: value for key, value in relevance_map.items() if key in ranking.keys()}
                    ),
                    Run(ranking),
                    metrics
                )
                result = {k: round(v, 3) for k, v in result.items()}
                result["fold_idx"] = fold_idx
                result["split"] = split
                result["cls"] = cls
                results.append(result)
        return pd.DataFrame(results)

    def _checkpoint_results(self, results):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        pd.DataFrame(results).to_csv(
            self.params.result.dir + self.params.model.name + "_" + self.params.data.name + ".rts",
            sep='\t', index=False, header=True)

    def _checkpoint_result(self, result, fold_idx):
        result_dir = f"{self.params.result.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_csv(
            f"{result_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rst",
            sep='\t',
            index=False,
            header=True
        )

    def _checkpoint_rankings(self, rankings):
        with open(
                self.params.ranking.dir + self.params.model.name + "_" + self.params.data.name + ".rnk",
                "wb") as rankings_file:
            pickle.dump(rankings, rankings_file)

    def checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)
