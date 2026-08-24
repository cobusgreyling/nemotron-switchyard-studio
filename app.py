#!/usr/bin/env python3
"""Nemotron Switchyard Studio — fine-tune a small specialist, route incoming queries."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import SYSTEM, TEST, TRAIN, as_sft_row, split_rows  # noqa: E402
from src.evaluate import score_generation  # noqa: E402
from src.nim import api_key, base_url, complete as nim_complete, nim_model  # noqa: E402
from src.router import AGENT_SESSION, SAMPLE_QUERIES, STRATEGIES, classify  # noqa: E402
from src.train import detect_device, generate_with_adapter, run_train  # noqa: E402

load_dotenv(ROOT / ".env")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "7878"))
SWITCHYARD_URL = os.getenv("SWITCHYARD_URL", "http://127.0.0.1:4000").rstrip("/")
SFT_MODEL = os.getenv("SFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = ROOT / "outputs" / "lora-sft"
STATIC_DIR = ROOT / "static"
HEADER_SRC = ROOT / "assets" / "header.jpg"
HEADER_DST = STATIC_DIR / "header.jpg"

STUDENTS = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5-0.5B-Instruct",
        "note": "Cached on this Mac · Lightning SFT recipe · ~2–4 min on MPS",
    },
    {
        "id": "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B",
        "label": "Nemotron Research Reasoning 1.5B",
        "note": "Actual NVIDIA Nemotron small model · first run downloads ~3 GB",
    },
]


def _ensure_header() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    if HEADER_SRC.exists() and (
        not HEADER_DST.exists() or HEADER_SRC.stat().st_mtime > HEADER_DST.stat().st_mtime
    ):
        HEADER_DST.write_bytes(HEADER_SRC.read_bytes())


_ensure_header()
app = FastAPI(title="Nemotron Switchyard Studio", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── training job ──────────────────────────────────────────────────────────────

class TrainState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "idle"
        self.message = "Ready"
        self.step = 0
        self.max_steps = 0
        self.losses: list[float] = []
        self.logs: list[str] = []
        self.run_card: dict[str, Any] | None = None
        self.error: str | None = None
        self.started_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "step": self.step,
                "max_steps": self.max_steps,
                "losses": list(self.losses),
                "logs": list(self.logs[-80:]),
                "run_card": self.run_card,
                "error": self.error,
                "adapter_ready": adapter_exists(),
                "elapsed_sec": (
                    round(time.time() - self.started_at, 1) if self.started_at else 0
                ),
            }

    def emit(self, event: dict[str, Any]) -> None:
        with self.lock:
            kind = event.get("type")
            if kind == "status":
                self.message = str(event.get("message") or "")
                self.logs.append(self.message)
            elif kind == "step":
                self.step = int(event.get("step") or 0)
                self.max_steps = int(event.get("max_steps") or self.max_steps)
                loss = event.get("loss")
                if isinstance(loss, (int, float)):
                    self.losses.append(float(loss))
                    self.message = f"step {self.step}/{self.max_steps}  loss={float(loss):.4f}"
                    self.logs.append(self.message)
            elif kind == "done":
                self.status = "done"
                self.run_card = event.get("run_card")
                self.message = "Adapter saved"
                self.logs.append("done — adapter saved")
            elif kind == "error":
                self.status = "error"
                self.error = str(event.get("message") or "train failed")
                self.message = self.error
                self.logs.append(f"ERROR {self.error}")


JOB = TrainState()
def adapter_exists() -> bool:
    return (ADAPTER_DIR / "adapter_config.json").exists()


def _reset_adapter_cache() -> None:
    """Hook for a future in-process adapter cache."""
    return None


def generate_stock(messages: list[dict[str, str]]) -> str:
    """Stock base model, no adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.train import detect_device as _dev

    device = _dev()
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL, trust_remote_code=True, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── routes ────────────────────────────────────────────────────────────────────


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    device = detect_device()
    return {
        "name": "Nemotron Switchyard Studio",
        "version": "0.1.0",
        "device": device,
        "sft_model": SFT_MODEL,
        "nim_model": nim_model(),
        "nim_ready": bool(api_key()),
        "nim_base": base_url() if api_key() else None,
        "adapter_ready": adapter_exists(),
        "students": STUDENTS,
        "contest": {
            "name": "NVIDIA GTC Berlin Golden Ticket",
            "url": "https://developer.nvidia.com/gtc-golden-ticket-contest",
            "hashtag": "#NVIDIAGTC",
        },
        "strategies": STRATEGIES,
        "runtime": _runtime(),
    }


def _switchyard_up() -> bool:
    try:
        r = httpx.get(f"{SWITCHYARD_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _track_for_selected(model_id: str) -> str:
    m = (model_id or "").lower()
    if "lightning" in m:
        return "frontier"
    if "nano" in m or "mini" in m:
        return "lightning"
    return "lightning"


def switchyard_chat(
    messages: list[dict[str, str]],
    *,
    model: str = "switchyard",
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with httpx.Client(timeout=90.0) as client:
        r = client.post(
            f"{SWITCHYARD_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
    ms = int((time.perf_counter() - t0) * 1000)
    if r.status_code >= 400:
        raise RuntimeError(f"Switchyard {r.status_code}: {r.text[:400]}")
    data = r.json()
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    selected = r.headers.get("x-model-router-selected-model") or data.get("model") or ""
    return {
        "content": (msg.get("content") or "").strip(),
        "model": data.get("model") or selected,
        "selected_model": selected,
        "rationale": r.headers.get("x-model-router-rationale") or "",
        "ms": ms,
        "usage": data.get("usage") or {},
        "track": _track_for_selected(selected),
    }


def _runtime() -> dict[str, Any]:
    native = _switchyard_up()
    official_bin = shutil.which("switchyard-server")
    return {
        "mode": "native-switchyard" if native else "educational-in-process",
        "listen": f"http://{HOST}:{PORT}",
        "process": f"Python PID {os.getpid()} · app.py",
        "router_file": "src/router.py",
        "what_runs_here": (
            "Studio UI on :7878. Live replies go through native Switchyard on :4000."
            if native
            else "This FastAPI app. Classify is in-process until native Switchyard is up on :4000."
        ),
        "official_switchyard": {
            "binary": "switchyard-server",
            "installed": bool(official_bin) or native,
            "running": native,
            "url": SWITCHYARD_URL if native else None,
            "path": official_bin,
            "typical_listen": "http://127.0.0.1:4000",
            "how": "python3 run_switchyard.py",
            "upstream": "https://github.com/NVIDIA-NeMo/Switchyard",
        },
        "nim": {
            "ready": bool(api_key()),
            "model": nim_model(),
            "base": base_url() if api_key() else None,
            "role": "upstream behind Switchyard — inference only",
        },
    }


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    def slim(ex: dict[str, Any]) -> dict[str, Any]:
        row = as_sft_row(ex)
        return {
            "id": row["id"],
            "split": row["split"],
            "family": row["family"],
            "user": row["user"],
            "completion": row["completion"],
            "gold": row["gold"],
        }

    return {
        "system": SYSTEM,
        "train": [slim(x) for x in TRAIN],
        "test": [slim(x) for x in TEST],
    }


class RouteBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    live: bool = False
    strategy: str = "content"
    step_type: str | None = None
    signals: list[str] | None = None
    risk: str = "normal"
    failures: int = 0
    restricted: bool | None = None


@app.get("/api/samples")
def samples() -> list[dict[str, str]]:
    return SAMPLE_QUERIES


@app.get("/api/runtime")
def runtime() -> dict[str, Any]:
    return _runtime()


@app.get("/api/session")
def session() -> dict[str, Any]:
    return AGENT_SESSION


class SessionBody(BaseModel):
    strategy: str = "stage"


@app.post("/api/session/run")
def session_run(body: SessionBody) -> dict[str, Any]:
    strategy = body.strategy if body.strategy in STRATEGIES else "stage"
    steps_out = []
    saved = 0.0
    frontier = 0.0
    mix: dict[str, int] = {"frontier": 0, "lightning": 0, "local": 0}
    for step in AGENT_SESSION["steps"]:
        d = classify(
            step["content"],
            adapter_ready=adapter_exists(),
            strategy=strategy,
            step_type=step.get("type"),
            signals=step.get("signals"),
            risk=step.get("risk", "normal"),
            restricted=step.get("restricted"),
        )
        mix[d.track] = mix.get(d.track, 0) + 1
        saved += d.frontier_only_usd - d.cost_usd
        frontier += d.frontier_only_usd
        steps_out.append({"step": step, "decision": d.as_dict()})
    return {
        "session": {"id": AGENT_SESSION["id"], "title": AGENT_SESSION["title"]},
        "strategy": strategy,
        "steps": steps_out,
        "mix": mix,
        "saved_usd": round(saved, 6),
        "frontier_only_usd": round(frontier, 6),
        "runtime": _runtime(),
    }


@app.post("/api/route")
def route_query(body: RouteBody) -> dict[str, Any]:
    strategy = body.strategy if body.strategy in STRATEGIES else "content"
    decision = classify(
        body.query,
        adapter_ready=adapter_exists(),
        strategy=strategy,
        step_type=body.step_type,
        signals=body.signals,
        risk=body.risk,
        failures=body.failures,
        restricted=body.restricted,
    )
    payload: dict[str, Any] = {
        "query": body.query,
        "decision": decision.as_dict(),
        "live": False,
        "reply": None,
        "ms": 0,
    }
    if not body.live:
        return payload

    t0 = time.perf_counter()
    try:
        if decision.track == "local":
            payload["reply"] = (
                "[local track] Restricted data never entered Switchyard or NIM."
            )
            payload["live"] = True
            payload["source"] = "local-stub"
            payload["via"] = "local"
        elif _switchyard_up():
            messages = [
                {
                    "role": "system",
                    "content": "You are a concise Nemotron specialist behind NeMo Switchyard.",
                },
                {"role": "user", "content": body.query},
            ]
            gen = switchyard_chat(
                messages,
                model="switchyard",
                max_tokens=min(max(decision.tokens_out_est, 80), 400),
                temperature=0.3 if decision.thinking else 0.2,
            )
            payload["reply"] = gen["content"]
            payload["live"] = True
            payload["source"] = gen["selected_model"] or gen["model"]
            payload["via"] = "native-switchyard"
            payload["rationale"] = gen["rationale"]
            payload["ms"] = gen["ms"]
            # Official router wins the visual track when it actually selected a model.
            if gen["selected_model"]:
                payload["decision"]["model_id"] = gen["selected_model"]
                payload["decision"]["track"] = gen["track"]
                payload["decision"]["reason"] = gen["rationale"] or payload["decision"]["reason"]
        elif api_key():
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Nemotron Lightning, the execute specialist."
                        if decision.track == "lightning"
                        else "You are a senior planner. Be concise and structured."
                    ),
                },
                {"role": "user", "content": body.query},
            ]
            gen = nim_complete(
                messages,
                model=nim_model(),
                temperature=0.3 if decision.thinking else 0.2,
                max_tokens=min(decision.tokens_out_est, 512),
                enable_thinking=decision.thinking,
            )
            payload["reply"] = gen["content"] or gen.get("reasoning") or ""
            payload["reasoning"] = gen.get("reasoning") or ""
            payload["live"] = True
            payload["source"] = gen["model"]
            payload["via"] = "nim-direct"
            payload["ms"] = gen["ms"]
        else:
            payload["reply"] = (
                f"[offline] Would route to **{decision.label}**. "
                "Start native Switchyard: python3 run_switchyard.py"
            )
            payload["source"] = "offline"
    except Exception as e:  # noqa: BLE001
        payload["error"] = str(e)
        payload["reply"] = f"upstream error: {e}"
    if not payload.get("ms"):
        payload["ms"] = int((time.perf_counter() - t0) * 1000)
    return payload


@app.get("/api/switchyard/stats")
def switchyard_stats() -> dict[str, Any]:
    if not _switchyard_up():
        return {"running": False}
    try:
        r = httpx.get(f"{SWITCHYARD_URL}/v1/stats", timeout=3.0)
        r.raise_for_status()
        return {"running": True, "url": SWITCHYARD_URL, "stats": r.json()}
    except Exception as e:  # noqa: BLE001
        return {"running": True, "error": str(e)}


class TrainBody(BaseModel):
    model_id: str = SFT_MODEL
    max_steps: int = Field(24, ge=4, le=80)
    lr: float = Field(2e-4, gt=0, lt=1)
    lora_r: int = Field(8, ge=4, le=32)


@app.get("/api/train/status")
def train_status() -> dict[str, Any]:
    return JOB.snapshot()


@app.post("/api/train/start")
def train_start(body: TrainBody) -> dict[str, Any]:
    with JOB.lock:
        if JOB.status == "running":
            raise HTTPException(409, "training already running")
        JOB.status = "running"
        JOB.message = "Starting…"
        JOB.step = 0
        JOB.max_steps = body.max_steps
        JOB.losses = []
        JOB.logs = ["queued"]
        JOB.run_card = None
        JOB.error = None
        JOB.started_at = time.time()

    examples = split_rows("train")

    def _worker() -> None:
        try:
            run_train(
                model_id=body.model_id or SFT_MODEL,
                examples=examples,
                output_dir=ADAPTER_DIR,
                max_steps=body.max_steps,
                lr=body.lr,
                lora_r=body.lora_r,
                emit=JOB.emit,
            )
            _reset_adapter_cache()
        except Exception as e:  # noqa: BLE001
            JOB.emit({"type": "error", "message": f"{e}\n{traceback.format_exc()[-600:]}"})

    threading.Thread(target=_worker, daemon=True, name="sft-train").start()
    return JOB.snapshot()


class CompareBody(BaseModel):
    example_id: str | None = None
    query: str | None = None
    include_nim: bool = True
    include_stock: bool = True
    include_adapter: bool = True


@app.post("/api/compare")
def compare(body: CompareBody) -> dict[str, Any]:
    ex = None
    if body.example_id:
        ex = next((as_sft_row(x) for x in TEST if x["id"] == body.example_id), None)
        if ex is None:
            ex = next((as_sft_row(x) for x in TRAIN if x["id"] == body.example_id), None)
        if ex is None:
            raise HTTPException(404, "example not found")
        query = ex["user"]
        gold = ex["gold"]
        messages = ex["prompt"]
    else:
        query = (body.query or "").strip()
        if not query:
            raise HTTPException(400, "query or example_id required")
        gold = None
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]

    out: dict[str, Any] = {"query": query, "gold": gold, "columns": []}

    if body.include_stock:
        try:
            text = generate_stock(messages)
            out["columns"].append(
                {
                    "id": "stock",
                    "label": f"Stock {SFT_MODEL.split('/')[-1]}",
                    "text": text,
                    "scores": score_generation(text, gold),
                    "ms": None,
                }
            )
        except Exception as e:  # noqa: BLE001
            out["columns"].append({"id": "stock", "label": "Stock", "error": str(e), "text": ""})

    if body.include_adapter:
        if adapter_exists():
            t0 = time.perf_counter()
            try:
                text = generate_with_adapter(
                    model_id=SFT_MODEL, adapter_dir=ADAPTER_DIR, messages=messages
                )
                out["columns"].append(
                    {
                        "id": "adapter",
                        "label": "LoRA specialist",
                        "text": text,
                        "scores": score_generation(text, gold),
                        "ms": int((time.perf_counter() - t0) * 1000),
                    }
                )
            except Exception as e:  # noqa: BLE001
                out["columns"].append(
                    {"id": "adapter", "label": "LoRA specialist", "error": str(e), "text": ""}
                )
        else:
            out["columns"].append(
                {
                    "id": "adapter",
                    "label": "LoRA specialist",
                    "text": "No adapter yet. Open Fine-tune and start a run.",
                    "scores": None,
                }
            )

    if body.include_nim and api_key():
        try:
            gen = nim_complete(messages, temperature=0.2, max_tokens=220, enable_thinking=False)
            out["columns"].append(
                {
                    "id": "nim",
                    "label": f"NIM {nim_model().split('/')[-1]}",
                    "text": gen["content"],
                    "scores": score_generation(gen["content"], gold),
                    "ms": gen["ms"],
                }
            )
        except Exception as e:  # noqa: BLE001
            out["columns"].append(
                {"id": "nim", "label": "NIM Lightning", "error": str(e), "text": ""}
            )

    return out


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "nim": bool(api_key()), "adapter": adapter_exists()}


def main() -> None:
    print(f"Nemotron Switchyard Studio → http://{HOST}:{PORT}")
    print(f"  device     : {detect_device()}")
    print(f"  student    : {SFT_MODEL}")
    print(f"  NIM        : {'ready' if api_key() else 'offline'}  ({nim_model()})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
