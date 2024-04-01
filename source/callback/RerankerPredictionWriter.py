from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import BasePredictionWriter


class RerankerPredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(RerankerPredictionWriter, self).__init__(params.write_interval)
        self.params = params
        self.checkpoint_dir = f"{self.params.dir}fold_{self.params.fold_idx}/"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass

    def write_on_batch_end(
            self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
            batch_idx: int, dataloader_idx: int
    ):
        predictions = []

        for query_idx, passage_idx, true_score, pred_score in zip(
                prediction["query_idx"].tolist(),
                prediction["passage_idx"].tolist(),
                prediction["true_score"].tolist(),
                prediction["pred_score"].tolist()
        ):
            predictions.append({
                "query_idx": query_idx,
                "passage_idx": passage_idx,
                "true_score": true_score,
                "pred_score": pred_score
            })

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        torch.save(
            predictions,
            f"{self.checkpoint_dir}{dataloader_idx}_{batch_idx}.prd"
        )
