import tiktoken

class Tokenizer:
    def __init__(self):
        tokenizer_base = tiktoken.get_encoding("r50k_base")
        num_base_tokens = tokenizer_base.n_vocab

        self.special_tokens = {
            "<|begin_of_text|>": num_base_tokens,
            "<|end_of_text|>": num_base_tokens + 1,
            "<|pad|>": num_base_tokens + 2
        }

        self.tiktoken_tokenizer = tiktoken.Encoding(
            name="llm-tokenizer",
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=tokenizer_base._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.pad_token_id = self.special_tokens["<|pad|>"]

    def __len__(self):
        return self.tiktoken_tokenizer.n_vocab

    def __call__(self, text, truncation=False, padding=False, max_length=None):
        if isinstance(text, list):
            return {
                "input_ids": [self._encode_single(t, truncation, padding, max_length) for t in text]
            }
        return self._encode_single(text, truncation, padding, max_length)

    def _encode_single(self, text, truncation, padding, max_length):
        ids = self.tiktoken_tokenizer.encode(
            text,
            allowed_special=set(self.special_tokens.keys())
        )

        if truncation and max_length is not None:
            ids = ids[:max_length]

        if padding == "max_length" and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids
