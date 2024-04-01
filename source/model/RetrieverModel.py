import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup

from source.metric.RetrieverMetric import RetrieverMetric


class RetrieverModel(LightningModule):
    """Encodes the text and label into a same space of embeddings."""

    def __init__(self, hparams):
        super(RetrieverModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoder
        self.encoder = instantiate(hparams.encoder)
        #print(hparams)
        #print(hparams.dropouts)
        dropoutsarr = [0.1, 0.1, 0.1]
        # dropout layers
        self.dropouts = [
            torch.nn.Dropout(p) for p in dropoutsarr
        ]

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = RetrieverMetric(hparams.metric)

    def forward(self, text, label):
        return self.encoder(text), self.encoder(label)

    def training_step(self, batch, batch_idx):
        query_idx, passage_idx, = batch["query_idx"], batch["passage_idx"]
        query_rpr, passage_rpr = self(batch["query"], batch["passage"])

        if len(self.hparams.dropouts) > 0:
            queries_rprs, passages_rprs = [], []
            for i in range(len(self.hparams.dropouts)):
                queries_rprs.append(self.dropouts[i](query_rpr))
                passages_rprs.append(self.dropouts[i](passage_rpr))

            query_idx = batch["query_idx"].repeat(len(self.hparams.dropouts))
            query_rpr = torch.cat(queries_rprs, dim=0)
            passage_idx = batch["passage_idx"].repeat(len(self.hparams.dropouts))
            passage_rpr = torch.cat(passages_rprs, dim=0)

        print(f"\nquery_rpr({query_rpr.shape}):\n{query_rpr}\n")
        print(f"\npassage_rpr({passage_rpr.shape}):\n{passage_rpr}\n")

        train_loss = self.loss(query_idx, query_rpr, passage_idx, passage_rpr)
        self.log('train_LOSS', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        text_rpr, label_rpr = self(batch["query"], batch["passage"])
        self.mrr.update(batch["query_idx"], text_rpr, batch["passage_idx"], label_rpr)

    def on_validation_epoch_end(self):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx == 0:
            return {
                "query_idx": batch["query_idx"],
                "query_rpr": self.encoder(batch["query"]),
                "modality": "query"
            }
        else:
            return {
                "passage_idx": batch["passage_idx"],
                "passage_rpr": self.encoder(batch["passage"]),
                "modality": "passage"
            }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
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
