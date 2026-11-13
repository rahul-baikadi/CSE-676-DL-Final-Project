from src.rewards.rule import RuleReward, keyword_score
from src.rewards.llm import LLMReward

REWARDS = {"rule": RuleReward, "llm": LLMReward}


def get_reward(name, **kwargs):
    return REWARDS[name](**kwargs)
