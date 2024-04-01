import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.PassageDataset import PassageDataset
from source.dataset.QueryDataset import QueryDataset
from source.dataset.RetrieverFitDataset import RetrieverFitDataset


class RetrieverDataModule(pl.LightningDataModule):
    def __init__(self, params, tokenizer, fold_idx, mysamples):
        super(RetrieverDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold_idx = fold_idx
        self.samples = mysamples

    def prepare_data(self):
        x = 1
        #with (open(f"{self.params.dir}samples.pkl", "rb") as dataset_file):
        #    self.samples = []
            # pickle.load(dataset_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = RetrieverFitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/train.pkl",
                tokenizer=self.tokenizer,
                query_max_length=self.params.query_max_length,
                passage_max_length=self.params.passage_max_length
            )

            self.val_dataset = RetrieverFitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/val.pkl",
                tokenizer=self.tokenizer,
                query_max_length=self.params.query_max_length,
                passage_max_length=self.params.passage_max_length
            )

        if stage == "predict":
            self.query_dataset = QueryDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/test.pkl",
                tokenizer=self.tokenizer,
                query_max_length=self.params.query_max_length
            )
            self.passage_dataset = PassageDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/test.pkl",
                tokenizer=self.tokenizer,
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
        return [
            DataLoader(self.query_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
            DataLoader(self.passage_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
        ]
