import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_ids(ids_path):
    with open(ids_path, "rb") as ids_file:
        return set(pickle.load(ids_file))


class QueryDataset(Dataset):
    """Retriever Predict Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, query_max_length):
        super(QueryDataset, self).__init__()
        self.queries = []
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.ids = _load_ids(ids_path)

        for sample in tqdm(samples, desc="Reading Queries"):
            if sample["idx"] in self.ids:
                self.queries.append({
                    "query_idx": sample["query_idx"],
                    "query": sample["query"],
                })

    def _encode(self, sample):
        return {
            "query_idx": sample["query_idx"],
            "query": torch.tensor(
                self.tokenizer.encode(
                    text=sample["query"], max_length=self.query_max_length, padding="max_length", truncation=True
                )),
        }

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self._encode(
            self.queries[idx]
        )
