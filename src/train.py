"""Completion-only LoRA SFT on a small student (MPS / CUDA / CPU).

NIM cannot train. Lightning 30B does not fit this Mac. The studio trains a
small open-weight student with the *same* mask, split, and card contract you
would use on Nemotron Lightning (Colab A100 / NeMo AutoModel).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

Emit = Callable[[dict[str, Any]], None]


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_example(tokenizer, prompt_messages: list[dict[str, str]], completion: str, max_length: int):
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None and (not comp_ids or comp_ids[-1] != eos):
        comp_ids = comp_ids + [eos]
    input_ids = (prompt_ids + comp_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return input_ids, labels


def run_train(
    *,
    model_id: str,
    examples: list[dict[str, Any]],
    output_dir: Path,
    max_steps: int = 24,
    lr: float = 2e-4,
    lora_r: int = 8,
    max_length: int = 384,
    seed: int = 42,
    emit: Emit | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    def log(event: dict[str, Any]) -> None:
        if emit:
            emit(event)

    if not examples:
        raise ValueError("no training examples")

    set_seed(seed)
    device = detect_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log({"type": "status", "message": f"device={device}  model={model_id}  n={len(examples)}  steps={max_steps}"})
    log({"type": "status", "message": "Loading tokenizer + base weights…"})

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device == "mps":
        model = model.to("mps")
    elif device == "cpu":
        model = model.to("cpu")
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, peft_config)
    model.train()
    trainable, total = model.get_nb_trainable_parameters()
    log(
        {
            "type": "status",
            "message": f"LoRA r={lora_r}  trainable={trainable:,} / {total:,} "
            f"({100 * trainable / max(total, 1):.3f}%)",
        }
    )

    encoded = [_encode_example(tokenizer, ex["prompt"], ex["completion"], max_length) for ex in examples]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    losses: list[float] = []
    t0 = time.perf_counter()
    step = 0
    while step < max_steps:
        ids, labels = encoded[step % len(encoded)]
        t = torch.tensor([ids], device=model.device)
        y = torch.tensor([labels], device=model.device)
        attn = torch.ones_like(t)
        out = model(input_ids=t, attention_mask=attn, labels=y)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        val = float(loss.detach().cpu())
        losses.append(val)
        log({"type": "step", "step": step, "max_steps": max_steps, "loss": val})

    elapsed = time.perf_counter() - t0
    log({"type": "status", "message": f"Saving adapter → {output_dir}"})
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    card = {
        "model_id": model_id,
        "device": device,
        "n_examples": len(examples),
        "max_steps": max_steps,
        "lr": lr,
        "lora_r": lora_r,
        "max_length": max_length,
        "trainable": trainable,
        "total": total,
        "elapsed_sec": round(elapsed, 1),
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    log({"type": "done", "run_card": card})
    return card


def generate_with_adapter(
    *,
    model_id: str,
    adapter_dir: Path,
    messages: list[dict[str, str]],
    max_new_tokens: int = 160,
    temperature: float = 0.2,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = detect_device()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(device)
    model.eval()
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()
