"""LoRA supervised fine-tuning on chat jsonl. Loss only on the assistant reply.

A plain loop instead of Trainer so every cost we report (tokens, seconds, peak memory)
is measured in one visible place.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """Token ids and labels, with the prompt masked out. Prompt is tokenized exactly as at inference."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template does not extend the generation prompt")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        raise ValueError(f"example of {len(p_ids) + len(c_ids)} tokens exceeds max_len={max_len}")
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", required=True, help="chat jsonl from hvt.data")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with open(a.train, encoding="utf-8") as f:
        rows = [encode(tok, json.loads(line)["messages"], a.max_len) for line in f if line.strip()]
    if not rows:
        raise SystemExit("training file is empty")

    # fp32 master weights with fp16 autocast: works on T4 and GTX 16xx, which lack bf16.
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    steps = a.epochs * -(-len(rows) // (a.batch_size * a.grad_accum))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, max(1, steps // 20), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    model.train()
    seen_tokens, losses, micro, start = 0, [], 0, time.perf_counter()
    for epoch in range(a.epochs):
        order = list(range(len(rows)))
        random.Random(a.seed * 1000 + epoch).shuffle(order)
        for b in range(0, len(order), a.batch_size):
            batch = [rows[i] for i in order[b:b + a.batch_size]]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([x + [pad_id] * (width - len(x)) for x, _ in batch], device=device)
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch], device=device)
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch], device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / a.grad_accum
            scaler.scale(loss).backward()
            seen_tokens += int(mask.sum())
            micro += 1
            last = b + a.batch_size >= len(order)
            if micro % a.grad_accum == 0 or last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                losses.append(loss.item() * a.grad_accum)
                if len(losses) % 20 == 0:
                    print(f"epoch {epoch} step {len(losses)}/{steps} loss {sum(losses[-20:]) / 20:.4f}", flush=True)
            if last:
                micro = 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    meta = {
        "model": a.model, "train_file": a.train, "examples": len(rows), "epochs": a.epochs, "rank": a.rank,
        "lr": a.lr, "seed": a.seed, "optimizer_steps": len(losses), "trainable_params": trainable,
        "train_tokens": seen_tokens, "seconds": round(time.perf_counter() - start, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_amp else None,
        "final_loss": round(sum(losses[-20:]) / len(losses[-20:]), 4),
        "device": torch.cuda.get_device_name() if use_amp else "cpu",
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
