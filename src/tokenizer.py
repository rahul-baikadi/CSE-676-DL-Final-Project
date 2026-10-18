from transformers import AutoTokenizer


class TextTokenizer:
    def __init__(self, tokenizer_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT-style tokenizers may not define a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text, max_length=None):
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
        )

    def decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )

    def batch_encode(self, texts, max_length=128):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )