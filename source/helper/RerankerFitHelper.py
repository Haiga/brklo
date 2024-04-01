import logging

import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything
from omegaconf import OmegaConf

from source.datamodule.RerankerDataModule import RerankerDataModule

from source.helper.Helper import Helper
from source.model.RerankerModel import RerankerModel


class RerankerFitHelper(Helper):

    def __init__(self, params):
        super(RerankerFitHelper, self).__init__()
        self.params = params
        logging.basicConfig(level=logging.INFO)

    def perform_fit(self):
        seed_everything(707, workers=True)
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Fitting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"params\n {OmegaConf.to_yaml(self.params)}\n")

            # Initialize a trainer
            trainer = pl.Trainer(
                accelerator=self.params.trainer.accelerator,
                devices=self.params.trainer.devices,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                logger=self.get_logger(fold_idx),
                callbacks=[
                    self.get_model_checkpoint_callback(fold_idx),  # checkpoint_callback
                    self.get_early_stopping_callback(),  # early_stopping_callback
                    self.get_lr_monitor(),
                    self.get_progress_bar_callback()
                ]
            )

            # datamodule
            # datamodule
            datamodule = RerankerDataModule(
                params=self.params.data,
                samples=self._load_samples(),
                queries=None,
                passages=None,
                ranking=None,
                tokenizer=self.get_tokenizer(),
                fold_idx=fold_idx)

            # model
            model = RerankerModel(self.params.model)

            # Train the ⚡ model
            trainer.fit(
                model=model,
                datamodule=datamodule
            )
