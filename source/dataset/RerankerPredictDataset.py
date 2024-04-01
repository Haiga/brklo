import logging

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RerankerPredictDataset(Dataset):
    """Predict Dataset.
    """

    def __init__(self, queries, passages, ranking, tokenizer, query_max_length, passage_max_length):
        super(RerankerPredictDataset, self).__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = query_max_length + passage_max_length

        for query_idx, passages_scores in tqdm(ranking.items(), desc=f"Reading ranking"):
            for passage_idx, score in passages_scores.items():
                self.samples.append({
                    "query_idx": query_idx,
                    "query": queries[query_idx],
                    "passage_idx": passage_idx,
                    "passage": passages[passage_idx],
                    "score": score
                })

        logging.info(f"\nUsing {len(self.samples)} samples.\n")

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
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
