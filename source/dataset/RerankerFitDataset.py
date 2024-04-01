import math
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RerankerFitDataset(Dataset):

    def __init__(self, samples, ids_path, tokenizer, query_max_length, passage_max_length):
        super(RerankerFitDataset, self).__init__()
        self.samples = samples
        self.max_length = query_max_length + passage_max_length
        self.ids = self._load_ids(ids_path)
        self.tokenizer = tokenizer

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, sample):
        encoding = self.tokenizer(text=sample["query"], text_pair=sample["passage"], max_length=self.max_length,
                                  padding="max_length", truncation=True)
        return {
            "query_idx": sample["query_idx"],
            "passage_idx": sample["passage_idx"],
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "token_type_ids": torch.tensor(encoding["token_type_ids"]),
            "score": torch.tensor(sample["score"], dtype=torch.long)
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[self.ids[idx]]
        )
