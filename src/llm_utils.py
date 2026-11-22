import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_llm(name, load_4bit=True):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
             if load_4bit else None)
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", quantization_config=quant, torch_dtype=torch.float16
    )
    return tok, model.eval()


@torch.no_grad()
def choice_probs(tok, model, prompts, choices, batch_size=16):
    """Renormalized probability of each single-token answer. Returns (N, len(choices))."""
    # Last token: Llama-style tokenizers put a shared blank token before digits.
    choice_ids = [tok.encode(c, add_special_tokens=False)[-1] for c in choices]
    if len(set(choice_ids)) != len(choices):
        raise ValueError(f"answer choices {choices} do not map to distinct tokens: {choice_ids}")
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts[i:i + batch_size]
        ]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, choice_ids].float()
        out.append(logits.softmax(-1).cpu())
    return torch.cat(out)
