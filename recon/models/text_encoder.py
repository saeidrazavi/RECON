"""Text encoder of DDT: Qwen3-0.6B hidden states of the captions."""

import torch


class Qwen3TextEncoder:
    """DDT text conditioning: the last hidden state of Qwen3-0.6B over captions padded or truncated
    to max_length tokens, and their attention mask (1 = token, 0 = padding). The model loads in its
    checkpoint dtype (bf16), and the features equal the causal LM's last hidden state that
    DiffusionBench reads."""

    def __init__(self, device, model_name="Qwen/Qwen3-0.6B", max_length=256):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, captions):
        tokens = self.tokenizer(list(captions), padding="max_length", max_length=self.max_length,
                                truncation=True, return_tensors="pt").to(self.device)
        return self.model(**tokens, use_cache=False).last_hidden_state, tokens["attention_mask"]
