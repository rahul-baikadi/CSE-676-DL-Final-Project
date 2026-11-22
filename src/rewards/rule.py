import re

import torch
from nltk.stem import WordNetLemmatizer

import nltk

try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    nltk.download("wordnet", quiet=True)


_lemm = WordNetLemmatizer()
_WORD = re.compile(r"[a-z]+")


def lemmas(text):
    return {_lemm.lemmatize(_lemm.lemmatize(w, "v"), "n") for w in _WORD.findall(text.lower())}


def keyword_score(words, story):
    story_lemmas = lemmas(story)
    return sum(1 for w in words if lemmas(w) & story_lemmas) / len(words)


def distinct_2(story):
    toks = _WORD.findall(story.lower())
    bigrams = list(zip(toks, toks[1:]))
    return len(set(bigrams)) / max(len(bigrams), 1)


class RuleReward:
    def __init__(self, all_bonus=0.5, min_words=30, short_pen=0.5,
                 trunc_pen=0.2, rep_weight=0.5):
        self.all_bonus = all_bonus
        self.min_words = min_words
        self.short_pen = short_pen
        self.trunc_pen = trunc_pen
        self.rep_weight = rep_weight

    def _score(self, words, story, finished):
        r = keyword_score(words, story)
        if r == 1.0:
            r += self.all_bonus
        n = len(_WORD.findall(story))
        if n < self.min_words:
            r -= self.short_pen * (1 - n / self.min_words)
        if not finished:
            r -= self.trunc_pen
        r -= self.rep_weight * (1 - distinct_2(story))
        return r

    def __call__(self, words, stories, finished):
        return torch.tensor([self._score(w, s, f) for w, s, f in zip(words, stories, finished)])
