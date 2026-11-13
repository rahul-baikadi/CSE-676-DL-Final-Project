import torch

from src.llm_utils import load_llm, choice_probs

PROMPT = """Rate this short children's story written for the required words: {words}.

Story:
{story}

Give one score from 1 to 5 considering:
- all required words are used, and used naturally in sentences
- grammar and fluency
- coherence of the plot

Answer with a single digit."""


class LLMReward:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", load_4bit=True, batch_size=16):
        self.tok, self.model = load_llm(model_name, load_4bit)
        self.model_name = model_name
        self.batch_size = batch_size
        self.scale = torch.arange(1, 6, dtype=torch.float)

    def __call__(self, words, stories, finished=None):
        prompts = [PROMPT.format(words=", ".join(w), story=s) for w, s in zip(words, stories)]
        probs = choice_probs(self.tok, self.model, prompts, list("12345"), self.batch_size)
        return (probs @ self.scale - 1) / 4
