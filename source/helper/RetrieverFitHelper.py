import logging

import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything
from omegaconf import OmegaConf

from source.datamodule.RetrieverDataModule import RetrieverDataModule
from source.helper.Helper import Helper
from source.model.RetrieverModel import RetrieverModel


class RetrieverFitHelper(Helper):

    def __init__(self, params):
        super(RetrieverFitHelper, self).__init__()
        self.params = params
        logging.basicConfig(level=logging.INFO)

    def perform_fit(self, mysamples, mydict={}):
        seed_everything(707, workers=True)
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Fitting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"self.params\n {OmegaConf.to_yaml(self.params)}\n")

            # Initialize a trainer
            trainer = pl.Trainer(
                num_sanity_val_steps=0,
                fast_dev_run=1,
                accelerator=self.params.trainer.accelerator,
                devices=self.params.trainer.devices,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                logger=self.get_logger(fold_idx),
                callbacks=[
                    self.get_model_checkpoint_callback(fold_idx),
                    self.get_early_stopping_callback(),
                    self.get_lr_monitor(),
                    self.get_progress_bar_callback()
                ]
            )

            # datamodule
            datamodule = RetrieverDataModule(
                params=self.params.data,
                tokenizer=self.get_tokenizer(),
                fold_idx=fold_idx,
                mysamples=mysamples)

            # model
            model = RetrieverModel(self.params.model)
            mydict["model"] = model
            mydict["loss"] = model.loss

            # Train the ⚡ model
            trainer.fit(
                model=model,
                datamodule=datamodule
            )
