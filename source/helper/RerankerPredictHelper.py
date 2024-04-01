import logging
import pickle

from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

from source.callback.RerankerPredictionWriter import RerankerPredictionWriter

from source.datamodule.RerankerDataModule import RerankerDataModule
from source.helper.Helper import Helper
from source.model.RerankerModel import RerankerModel


class RerankerPredictHelper(Helper):

    def __init__(self, params):
        super(RerankerPredictHelper, self).__init__()
        self.params = params
        logging.basicConfig(level=logging.INFO)

    def perform_predict(self):
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            # datamodule
            datamodule = RerankerDataModule(
                params=self.params.data,
                samples=None,
                queries=self._load_queries(),
                passages=self._load_passages(),
                ranking=self._load_ranking(fold_idx),
                tokenizer=self.get_tokenizer(),
                fold_idx=fold_idx)

            # model
            model = RerankerModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.ckpt"
            )

            self.params.prediction.fold_idx = fold_idx
            # trainer
            trainer = pl.Trainer(
                accelerator=self.params.trainer.accelerator,
                devices=self.params.trainer.devices,
                precision=self.params.trainer.precision,
                callbacks=[RerankerPredictionWriter(self.params.prediction)]
            )

            # predicting
            datamodule.prepare_data()
            datamodule.setup("predict")

            trainer.predict(
                model=model,
                datamodule=datamodule,

            )
