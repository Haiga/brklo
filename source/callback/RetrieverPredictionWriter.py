from pathlib import Path
from typing import Any, List, Sequence, Optional

import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor


class RetrieverPredictionWriter(BasePredictionWriter):

    def __init__(self, params):
        super(RetrieverPredictionWriter, self).__init__(params.write_interval)
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

        if prediction["modality"] == "query":
            for query_idx, query_rpr in zip(
                    prediction["query_idx"].tolist(),
                    prediction["query_rpr"].tolist()):
                predictions.append({

                    "query_idx": query_idx,
                    "query_rpr": query_rpr,
                    "modality": "query"
                })

        elif prediction["modality"] == "passage":
            for passage_idx, passage_rpr in zip(
                    prediction["passage_idx"].tolist(),
                    prediction["passage_rpr"].tolist()):
                predictions.append({
                    "passage_idx": passage_idx,
                    "passage_rpr": passage_rpr,
                    "modality": "passage"
                })

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        torch.save(
            predictions,
            f"{self.checkpoint_dir}{dataloader_idx}_{batch_idx}.prd"
        )
