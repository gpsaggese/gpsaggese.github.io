from typing import Iterable, List

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from trlx.pipeline import BasePipeline, register_datapipeline


@register_datapipeline
class DPOPipeline(BasePipeline):
    """
    Minimal DPO pipeline that tokenizes (prompt, chosen, rejected) records.
    """

    def __init__(self, samples: List[dict], max_prompt_length: int, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.samples = samples
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __getitem__(self, index: int):
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def _collate(self, batch: Iterable[dict]):
        prompts = [b["prompt"] for b in batch]
        chosen = [b["chosen"] for b in batch]
        rejected = [b["rejected"] for b in batch]

        prompt_tok = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=True,
            return_tensors="pt",
        )

        def encode_labels(texts):
            out = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_prompt_length,
                padding=True,
                return_tensors="pt",
            )
            labels = out["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100
            return labels

        chosen_labels = encode_labels(chosen)
        rejected_labels = encode_labels(rejected)

        return {
            "prompt_input_ids": prompt_tok["input_ids"],
            "prompt_attention_mask": prompt_tok["attention_mask"],
            "input_ids": prompt_tok["input_ids"],
            "attention_mask": prompt_tok["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
        }

    def create_loader(self, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
        )
