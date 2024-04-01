import logging
import pickle
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


def _load_ids(ids_path):
    with open(ids_path, "rb") as ids_file:
        return pickle.load(ids_file)


class RetrieverFitDataset(Dataset):
    """Retriever Fit Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, query_max_length, passage_max_length):
        super(RetrieverFitDataset, self).__init__()
        self.samples = samples

        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

        self.ids = _load_ids(ids_path)

    def _encode(self, sample):
        return {
            "query_idx": sample["query_idx"],
            "query": torch.tensor(
                self.tokenizer.encode(
                    text=sample["query"], max_length=self.query_max_length, padding="max_length", truncation=True
                )),
            "passage_idx": sample["passage_idx"],
            "passage": torch.tensor(
                self.tokenizer.encode(
                    text=sample["passage"], max_length=self.passage_max_length, padding="max_length", truncation=True
                ))
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[self.ids[idx]]
        )
