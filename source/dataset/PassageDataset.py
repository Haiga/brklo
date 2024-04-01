import logging
import math
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_ids(ids_path):
    with open(ids_path, "rb") as ids_file:
        return set(pickle.load(ids_file))


class PassageDataset(Dataset):
    """Retriever Predict Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, passage_max_length):
        super(PassageDataset, self).__init__()
        self.passages = []
        self.tokenizer = tokenizer
        self.passage_max_length = passage_max_length
        self.ids = _load_ids(ids_path)

        for sample in tqdm(samples, desc="Reading Queries"):
            if sample["idx"] in self.ids:
                self.passages.append({
                    "passage_idx": sample["passage_idx"],
                    "passage": sample["passage"],
                })

    def _encode(self, sample):
        return {
            "passage_idx": sample["passage_idx"],
            "passage": torch.tensor(
                self.tokenizer.encode(
                    text=sample["passage"], max_length=self.passage_max_length, padding="max_length", truncation=True
                )
            )
        }

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        return self._encode(
            self.passages[idx]
        )
