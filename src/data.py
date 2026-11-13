import re

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

_FIELD = r"^(?:Features|Words|Summary|Random sentence|Story):"


def _pad(ids, length, value):
    return ids + [value] * (length - len(ids))


# ---------- plain language-model data ----------
class LanguageModelDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts, self.tok, self.max_length = texts, tokenizer.tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = (self.tok.encode(self.texts[idx]) + [self.tok.eos_token_id])[: self.max_length + 1]
        return {
            "input_ids": torch.tensor(_pad(ids[:-1], self.max_length, self.tok.pad_token_id)),
            "labels": torch.tensor(_pad(ids[1:], self.max_length, -100)),
        }


def load_text_dataset(dataset_name, config_name=None, text_column="text",
                      split="train", num_samples=None):
    ds = load_dataset(dataset_name, config_name, split=split)
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))
    return [t for t in ds[text_column] if t.strip()]


def create_dataloader(texts, tokenizer, batch_size=32, max_length=128, shuffle=True):
    return DataLoader(LanguageModelDataset(texts, tokenizer, max_length),
                      batch_size=batch_size, shuffle=shuffle)


# ---------- keyword -> story pairs ----------
def _parse_entry(chunk):
    words = re.search(r"^Words:\s*(.+)$", chunk, re.M)
    story = re.search(r"^Story:\s*(.*?)(?=" + _FIELD + r"|\Z)", chunk, re.M | re.S)
    if not (words and story and story.group(1).strip()):
        return None
    return [w.strip() for w in words.group(1).split(",") if w.strip()], story.group(1).strip()


def load_keyword_story_pairs(split="train", num_samples=None,
                             dataset_name="roneneldan/TinyStoriesInstruct"):
    pairs, buffer = [], ""
    # parquet revision avoids a Windows UnicodeDecodeError in the raw .txt files
    ds = load_dataset(dataset_name, revision="refs/convert/parquet", split=split, streaming=True)
    for row in ds:
        buffer += row["text"] + "\n"
        if "<|endoftext|>" not in buffer:
            continue
        *chunks, buffer = buffer.split("<|endoftext|>")
        for chunk in chunks:
            pair = _parse_entry(chunk)
            if pair:
                pairs.append(pair)
                if num_samples is not None and len(pairs) >= num_samples:
                    return pairs
    return pairs


class KeywordStoryDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src_len=32, max_length=128):
        self.pairs, self.tok = pairs, tokenizer.tokenizer
        self.max_src_len, self.max_length = max_src_len, max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        words, story = self.pairs[idx]
        eos, pad, S, T = self.tok.eos_token_id, self.tok.pad_token_id, self.max_src_len, self.max_length

        src = self.tok.encode(", ".join(words))[:S]
        seq = ([eos] + self.tok.encode(story) + [eos])[: T + 1]
        tgt, labels = seq[:-1], seq[1:]

        return {
            "src_ids": torch.tensor(_pad(src, S, pad)),
            "src_mask": torch.tensor(_pad([1] * len(src), S, 0)),
            "tgt_input": torch.tensor(_pad(tgt, T, pad)),
            "tgt_mask": torch.tensor(_pad([1] * len(tgt), T, 0)),
            "labels": torch.tensor(_pad(labels, T, -100)),
        }


def create_keyword_dataloader(pairs, tokenizer, batch_size=32,
                              max_src_len=32, max_length=128, shuffle=True):
    return DataLoader(KeywordStoryDataset(pairs, tokenizer, max_src_len, max_length),
                      batch_size=batch_size, shuffle=shuffle)
