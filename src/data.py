import torch

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class LanguageModelDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoded = self.tokenizer.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length + 1,
            return_tensors="pt",
        )

        tokens = encoded["input_ids"].squeeze(0)

        # Shift by one token for next-token prediction.
        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def load_text_dataset(
    dataset_name,
    config_name=None,
    text_column="text",
    split="train",
    num_samples=None,
):
    dataset = load_dataset(dataset_name, config_name, split=split)

    if num_samples is not None:
        dataset = dataset.select(
            range(min(num_samples, len(dataset)))
        )

    texts = [
        example[text_column]
        for example in dataset
        if example[text_column].strip()
    ]

    return texts


def create_dataloader(
    texts,
    tokenizer,
    batch_size=32,
    max_length=128,
    shuffle=True,
):
    dataset = LanguageModelDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )