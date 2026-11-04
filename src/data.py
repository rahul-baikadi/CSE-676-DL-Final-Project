import re

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

def load_keyword_story_pairs(
    split="train",
    num_samples=None,
    dataset_name="roneneldan/TinyStoriesInstruct",
):
    
    ds = load_dataset(dataset_name, split=split, streaming=True)

    pairs, buffer = [], []
    for row in ds:
        buffer.append(row["text"])
        joined = "\n".join(buffer)
        if "<|endoftext|>" not in joined:
            continue

        chunks = joined.split("<|endoftext|>")
        buffer = [chunks[-1]]                 # keep the unfinished remainder
        for chunk in chunks[:-1]:
            pair = _parse_instruct_entry(chunk)
            if pair is not None:
                pairs.append(pair)
                if num_samples is not None and len(pairs) >= num_samples:
                    return pairs
    return pairs


def _parse_instruct_entry(chunk):
    words_match = re.search(r"^Words:\s*(.+)$", chunk, flags=re.MULTILINE)
    story_match = re.search(r"^Story:\s*(.*)", chunk, flags=re.MULTILINE | re.DOTALL)
    if not words_match or not story_match:
        return None   # some entries have no Words field; skip them

    words = [w.strip() for w in words_match.group(1).split(",") if w.strip()]
    story = story_match.group(1).strip()
    if not words or not story:
        return None
    return words, story


class KeywordStoryDataset(Dataset):

    def __init__(self, pairs, tokenizer, max_src_len=32, max_length=128):
        self.pairs = pairs
        self.tok = tokenizer.tokenizer
        self.eos = self.tok.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.max_src_len = max_src_len
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def _pad(self, ids, length, value):
        return ids + [value] * (length - len(ids))

    def __getitem__(self, idx):
        words, story = self.pairs[idx]

        # ---- encoder side ----
        src = self.tok.encode(", ".join(words))[: self.max_src_len]
        src_mask = [1] * len(src)

        # ---- decoder side ----
        # [eos] story [eos], truncated. If the story is too long, the final eos
        # is cut off, so the model is never taught to stop at the truncation point.
        seq = [self.eos] + self.tok.encode(story) + [self.eos]
        seq = seq[: self.max_length + 1]

        tgt_input = seq[:-1]
        labels = seq[1:]
        tgt_mask = [1] * len(tgt_input)

        T, S = self.max_length, self.max_src_len
        return {
            "src_ids":   torch.tensor(self._pad(src, S, self.pad)),
            "src_mask":  torch.tensor(self._pad(src_mask, S, 0)),
            "tgt_input": torch.tensor(self._pad(tgt_input, T, self.pad)),
            "tgt_mask":  torch.tensor(self._pad(tgt_mask, T, 0)),
            "labels":    torch.tensor(self._pad(labels, T, -100)),
        }


def create_keyword_dataloader(pairs, tokenizer, batch_size=32,
                              max_src_len=32, max_length=128, shuffle=True):
    dataset = KeywordStoryDataset(pairs, tokenizer, max_src_len, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
