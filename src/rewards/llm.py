import torch

from src.llm_utils import load_llm, choice_probs

PROMPT = """You are grading a short children's story. It must use these required words: {words}.

Story:
{story}

Grade the writing with this scale:
1 = random words, a list of words, or broken sentences
2 = many grammar errors, or the same text repeats
3 = readable but the plot is confusing
4 = clear story with small problems
5 = fluent, coherent story that uses every required word naturally

Answer with one digit from 1 to 5."""


class LLMReward:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct", load_4bit=True, batch_size=16,
                 score_scale=2.7):
        self.tok, self.model = load_llm(model_name, load_4bit)
        self.model_name = model_name
        self.batch_size = batch_size
        self.score_scale = score_scale      # matches the rule reward's range so beta is comparable
        self.scale = torch.arange(1, 6, dtype=torch.float)

    def __call__(self, words, stories, finished=None):
        prompts = [PROMPT.format(words=", ".join(w), story=s) for w, s in zip(words, stories)]
        probs = choice_probs(self.tok, self.model, prompts, list("12345"), self.batch_size)
        return self.score_scale * (probs @ self.scale - 1) / 4
