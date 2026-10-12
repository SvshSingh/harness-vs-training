"""Hugging Face causal LM behind the harness Model protocol. Greedy decoding, batched."""
from __future__ import annotations


class HFModel:
    def __init__(self, name: str, adapter: str | None = None, batch_size: int = 16, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens

    def generate(self, conversations: list[list[dict]]) -> list[tuple[str, int, int]]:
        out = []
        for i in range(0, len(conversations), self.batch_size):
            batch = conversations[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for row, mask in zip(new, enc["attention_mask"]):
                n_new = int((row != self.tok.pad_token_id).sum())
                out.append((self.tok.decode(row, skip_special_tokens=True).strip(), int(mask.sum()), n_new))
        return out
