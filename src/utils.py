import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data import load_keyword_story_pairs
from src.model import Seq2SeqTransformer


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def load_pairs(split, num_samples, cache_dir="data_cache"):
    path = Path(cache_dir) / f"{split}_{num_samples}.json"
    if path.exists():
        return [tuple(p) for p in json.loads(path.read_text(encoding="utf-8"))]
    pairs = load_keyword_story_pairs(split=split, num_samples=num_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs), encoding="utf-8")
    return pairs


def save_checkpoint(path, model, config, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, **extra}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = Seq2SeqTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt
