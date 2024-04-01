import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from source.dataset.RerankerFitDataset import RerankerFitDataset
from source.dataset.RerankerPredictDataset import RerankerPredictDataset


class RerankerDataModule(pl.LightningDataModule):
    def __init__(self, params, samples, queries, passages, ranking, tokenizer, fold_idx):
        super(RerankerDataModule, self).__init__()
        self.params = params
        self.ranking = ranking
        self.tokenizer = tokenizer
        self.fold_idx = fold_idx
        self.samples = samples
        self.queries = queries
        self.passages = passages

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = RerankerFitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/train.pkl",
                tokenizer=self.tokenizer,
                query_max_length=self.params.query_max_length,
                passage_max_length=self.params.passage_max_length
            )

            self.val_dataset = RerankerFitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/val.pkl",
                tokenizer=self.tokenizer,
                query_max_length=self.params.query_max_length,
                passage_max_length=self.params.passage_max_length
            )

        if stage == "predict":
            self.predict_dataset = RerankerPredictDataset(
                queries=self.queries,
                passages=self.passages,
                ranking=self.ranking,
                tokenizer=self.tokenizer,
                query_max_length=self.params.query_max_length,
                passage_max_length=self.params.passage_max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )
