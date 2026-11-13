from src.llm_utils import load_llm, choice_probs

PROMPT = """Two short children's stories were written for the required words: {words}.

Story A:
{a}

Story B:
{b}

Which story is better overall, considering grammar, coherence, and whether the
required words are used naturally (not just listed)? Answer with a single letter: A or B."""


class PairwiseJudge:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct",
                 load_4bit=True, batch_size=8, margin=0.1):
        self.tok, self.model = load_llm(model_name, load_4bit)
        self.model_name = model_name
        self.batch_size = batch_size
        self.margin = margin

    def _prompts(self, words, a, b):
        return [PROMPT.format(words=", ".join(w), a=x, b=y) for w, x, y in zip(words, a, b)]

    def prob_b_better(self, words, stories_a, stories_b):
        """P(B better), averaged over both presentation orders to cancel position bias."""
        p1 = choice_probs(self.tok, self.model, self._prompts(words, stories_a, stories_b),
                          ["A", "B"], self.batch_size)[:, 1]
        p2 = choice_probs(self.tok, self.model, self._prompts(words, stories_b, stories_a),
                          ["A", "B"], self.batch_size)[:, 0]
        return (p1 + p2) / 2

    def win_rate(self, words, baseline, candidate):
        p = self.prob_b_better(words, baseline, candidate)
        win = (p > 0.5 + self.margin).float().mean().item()
        loss = (p < 0.5 - self.margin).float().mean().item()
        return {"win": win, "loss": loss, "tie": 1 - win - loss}
