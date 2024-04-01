import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch import nn
from transformers import get_linear_schedule_with_warmup

from source.metric.RerankerMetric import RerankerMetric
from source.pooling.NoPooling import NoPooling


class RerankerModel(LightningModule):

    def __init__(self, hparams):
        super(RerankerModel, self).__init__()

        self.save_hyperparameters(hparams)

        self.encoder = instantiate(hparams.encoder)

        self.dropout = torch.nn.Dropout(hparams.dropout)

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.hidden_size, self.hparams.num_classes),
            # torch.nn.LogSoftmax(dim=-1)
        )

        # loss
        self.loss = torch.nn.CrossEntropyLoss()

        # metric
        self.mrr = RerankerMetric(hparams.metric)

    def forward(self, features):
        rpr = self.encoder(features)
        return self.cls_head(rpr)[:, -1]

    def training_step(self, features, batch_idx):
        rpr = self.encoder(features)
        pred_score = self.cls_head(
            self.dropout(rpr)
        )
        # print(f"true_score({true_score.shape})\n{true_score}\n")
        # print(f"pred_score({pred_score.shape})\n{pred_score}\n")
        # log training loss
        train_loss = self.loss(pred_score, features["score"])
        self.log('train_Loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, features, batch_idx):
        rpr = self.encoder(features)
        pred_score = self.cls_head(rpr)[:, -1]
        self.mrr.update(features["query_idx"], features["passage_idx"], pred_score)

    def on_validation_epoch_end(self):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, features, batch_idx, dataloader_idx=0):
        true_score = features["score"]
        rpr = self.encoder(features)
        pred_score = self.cls_head(rpr)
        pred_score = nn.functional.softmax(pred_score, dim=-1)[:, -1]

        return {
            "query_idx": features["query_idx"],
            "passage_idx": features["passage_idx"],
            "true_score": true_score,
            "pred_score": pred_score

        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        # schedulers
        # step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
        #                                               base_lr=self.hparams.base_lr,
        #                                               max_lr=self.hparams.max_lr, step_size_up=step_size_up,
        #                                               cycle_momentum=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)

        return (
            {"optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        )

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
    #                                   eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
    #
    #     # schedulers
    #     step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)
    #
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
    #                                                   base_lr=self.hparams.base_lr,
    #                                                   max_lr=self.hparams.max_lr, step_size_up=step_size_up,
    #                                                   cycle_momentum=False)
    #
    #     return (
    #         {"optimizer": optimizer,
    #          "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
    #     )
