"""Educational Switchyard classifier for incoming queries.

Mirrors NVIDIA NeMo Switchyard's idea: every prompt is a *step*, and the
library picks a target from configurable priorities (quality, latency, cost).

Tracks
------
frontier   plan / hard reasoning / high-risk verify  → Nemotron Ultra / thinking-on
lightning  high-volume execute / tools / format      → fine-tuned Nemotron specialist
local      restricted / PII                          → never leaves the machine
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FRONTIER_MARKERS = (
    "architecture",
    "multi-region",
    "decompose",
    "design",
    "plan",
    "outline",
    "migrate",
    "cutover",
    "prove",
    "diagnose",
    "refactor",
    "active-active",
    "race condition",
    "security review",
    "zero-downtime",
    "tradeoff",
    "trade-off",
)

EXECUTE_MARKERS = (
    "ticket",
    "refund",
    "invoice",
    "tool",
    "json",
    "format",
    "schema",
    "call billing",
    "get_invoice",
    "create_refund",
    "<<ticket>>",
    "deskcard",
    "parse",
    "extract",
    "summarize the tool",
)

PRIVACY_MARKERS = (
    "ssn",
    "social security",
    "passport",
    "pci",
    "credit card",
    "card number",
    "pii",
    "patient",
    "hipaa",
    "restricted",
    "stay local",
    "on-prem",
    "do not send",
)

PRICES = {
    # illustrative $/1M tokens — not a provider invoice
    "frontier": (3.00, 15.00),
    "lightning": (0.20, 0.60),
    "local": (0.0, 0.0),
}

TRACK_MODELS = {
    # Trial NIM: Lightning with thinking ON stands in for Ultra on the plan track.
    "frontier": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "lightning": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "local": "local/nemotron-sft-adapter",
}

TRACK_LABELS = {
    "frontier": "Frontier · Lightning thinking-on (Ultra-class plan)",
    "lightning": "Execute · Nemotron Lightning (specialist)",
    "local": "Local · never leaves this machine",
}


@dataclass
class RouteDecision:
    track: str
    model_id: str
    label: str
    reason: str
    score: float
    signals: list[str] = field(default_factory=list)
    thinking: bool = False
    tokens_in: int = 0
    tokens_out_est: int = 0
    cost_usd: float = 0.0
    frontier_only_usd: float = 0.0
    strategy: str = "content"

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "model_id": self.model_id,
            "label": self.label,
            "reason": self.reason,
            "score": round(self.score, 3),
            "signals": self.signals,
            "thinking": self.thinking,
            "tokens_in": self.tokens_in,
            "tokens_out_est": self.tokens_out_est,
            "cost_usd": round(self.cost_usd, 6),
            "frontier_only_usd": round(self.frontier_only_usd, 6),
            "saved_usd": round(self.frontier_only_usd - self.cost_usd, 6),
            "strategy": self.strategy,
        }


def _hits(text: str, markers: tuple[str, ...]) -> list[str]:
    found = []
    for m in markers:
        if re.search(r"(?:^|[^a-z0-9])" + re.escape(m) + r"(?:$|[^a-z0-9])", text):
            found.append(m)
    return found


def estimate_cost(track: str, tokens_in: int, tokens_out: int) -> float:
    inp, out = PRICES[track]
    return (tokens_in / 1_000_000) * inp + (tokens_out / 1_000_000) * out


STRATEGIES = {
    "content": "Content classifier (keywords + score) — what this studio used first",
    "stage": "Stage router — plan/error/verify signals pick capable vs efficient",
    "policy": "Plan/execute policy — harness tags, privacy, escalate on failures",
}


def infer_step(query: str) -> dict[str, Any]:
    low = (query or "").lower()
    privacy = _hits(low, PRIVACY_MARKERS)
    plan = _hits(low, FRONTIER_MARKERS)
    execute = _hits(low, EXECUTE_MARKERS)
    if privacy:
        return {"type": "execute", "restricted": True, "signals": privacy, "risk": "high"}
    if plan and not execute:
        return {"type": "plan", "restricted": False, "signals": plan, "risk": "normal"}
    if any(s in low for s in ("policy", "verify", "confirm against")):
        return {"type": "verify", "restricted": False, "signals": ["policy_check"], "risk": "high"}
    return {
        "type": "execute",
        "restricted": False,
        "signals": execute or (["ticket"] if "ticket" in low else []),
        "risk": "normal",
    }


def classify(
    query: str,
    *,
    adapter_ready: bool = False,
    strategy: str = "content",
    step_type: str | None = None,
    signals: list[str] | None = None,
    risk: str = "normal",
    failures: int = 0,
    restricted: bool | None = None,
) -> RouteDecision:
    text = (query or "").strip()
    inferred = infer_step(text)
    step_type = (step_type or inferred["type"]).lower()
    signals = list(signals if signals is not None else inferred["signals"])
    if restricted is None:
        restricted = bool(inferred["restricted"])
    if risk == "normal" and inferred["risk"] == "high":
        risk = inferred["risk"]

    if strategy == "stage":
        d = _classify_stage(
            text,
            step_type=step_type,
            signals=signals,
            risk=risk,
            failures=failures,
            restricted=restricted,
            adapter_ready=adapter_ready,
        )
        d.strategy = strategy
        return d
    if strategy == "policy":
        d = _classify_policy(
            text,
            step_type=step_type,
            risk=risk,
            failures=failures,
            restricted=restricted,
            adapter_ready=adapter_ready,
        )
        d.strategy = strategy
        return d
    d = _classify_content(text, adapter_ready=adapter_ready)
    d.strategy = "content"
    return d


def _finish(
    track: str,
    reason: str,
    signals: list[str],
    score: float,
    thinking: bool,
    tokens_in: int,
    tokens_out: int,
    adapter_ready: bool,
) -> RouteDecision:
    model_id = TRACK_MODELS[track]
    if track == "lightning" and adapter_ready:
        model_id = "local/nemotron-sft-adapter"
    return RouteDecision(
        track=track,
        model_id=model_id,
        label=TRACK_LABELS[track],
        reason=reason,
        score=score,
        signals=signals,
        thinking=thinking,
        tokens_in=tokens_in,
        tokens_out_est=tokens_out,
        cost_usd=estimate_cost(track, tokens_in, tokens_out),
        frontier_only_usd=estimate_cost("frontier", tokens_in, max(tokens_out, 400)),
        strategy="",
    )


def _classify_content(text: str, *, adapter_ready: bool) -> RouteDecision:
    low = text.lower()
    tokens_in = max(8, len(text.split()) if text else 8)
    privacy = _hits(low, PRIVACY_MARKERS)
    plan = _hits(low, FRONTIER_MARKERS)
    execute = _hits(low, EXECUTE_MARKERS)
    length_boost = min(len(text) / 400.0, 0.25)
    score = min(1.0, 0.18 + 0.18 * len(plan) + length_boost)
    if execute and not plan:
        score = min(score, 0.32)

    if privacy:
        return _finish(
            "local",
            f"restricted data → local ({', '.join(privacy[:3])})",
            ["privacy"] + privacy[:3],
            0.95,
            False,
            tokens_in,
            120,
            adapter_ready,
        )
    if plan and (score >= 0.45 or not execute):
        return _finish(
            "frontier",
            f"plan / hard reasoning ({', '.join(plan[:3]) or 'complexity'})",
            ["plan"] + plan[:3],
            max(score, 0.72),
            True,
            tokens_in,
            500,
            adapter_ready,
        )
    if execute or "ticket t-" in low or low.startswith("ticket "):
        specialist = "fine-tuned adapter" if adapter_ready else "Lightning NIM execute"
        return _finish(
            "lightning",
            f"high-volume execute → {specialist}",
            ["execute"] + (execute[:3] or ["ticket"]),
            min(score, 0.38),
            False,
            tokens_in,
            140,
            adapter_ready,
        )
    if score >= 0.55:
        return _finish(
            "frontier",
            f"classifier score {score:.2f} ≥ 0.55",
            ["classifier"],
            score,
            True,
            tokens_in,
            400,
            adapter_ready,
        )
    return _finish(
        "lightning",
        f"default execute (score {score:.2f} < 0.55)",
        ["default-execute"],
        score,
        False,
        tokens_in,
        160,
        adapter_ready,
    )


def _classify_stage(
    text: str,
    *,
    step_type: str,
    signals: list[str],
    risk: str,
    failures: int,
    restricted: bool,
    adapter_ready: bool,
) -> RouteDecision:
    tokens_in = max(8, len(text.split()) if text else 8)
    sigs = set(signals)
    error_sigs = {"error", "test_fail", "retry", "timeout", "rate_limit"}
    capable_sigs = {"policy_check", "security_review", "architecture"}
    if restricted:
        return _finish("local", "stage: restricted → local", ["privacy"], 0.95, False, tokens_in, 120, adapter_ready)
    if step_type in {"plan", "escalate"}:
        return _finish(
            "frontier", "stage: plan/escalate → capable", ["plan", step_type], 0.85, True, tokens_in, 500, adapter_ready
        )
    if sigs & error_sigs or failures >= 1:
        hit = sorted(sigs & error_sigs) or ["failures"]
        return _finish(
            "frontier", f"stage: error signals {hit}", ["error"] + hit, 0.8, True, tokens_in, 400, adapter_ready
        )
    if sigs & capable_sigs or (risk == "high" and step_type == "verify"):
        return _finish(
            "frontier",
            "stage: high-stakes verify → capable",
            ["verify"] + sorted(sigs & capable_sigs),
            0.78,
            True,
            tokens_in,
            360,
            adapter_ready,
        )
    return _finish(
        "lightning", "stage: tool/progress → efficient", ["execute"] + list(sigs)[:3], 0.28, False, tokens_in, 140, adapter_ready
    )


def _classify_policy(
    text: str,
    *,
    step_type: str,
    risk: str,
    failures: int,
    restricted: bool,
    adapter_ready: bool,
) -> RouteDecision:
    tokens_in = max(8, len(text.split()) if text else 8)
    if restricted:
        return _finish("local", "policy: restricted → local", ["privacy"], 0.95, False, tokens_in, 120, adapter_ready)
    if failures >= 2:
        return _finish(
            "frontier", f"policy: escalate after {failures} failures", ["escalate"], 0.9, True, tokens_in, 400, adapter_ready
        )
    if step_type in {"plan", "escalate"}:
        return _finish("frontier", "policy: planning needs strong tier", ["plan"], 0.82, True, tokens_in, 500, adapter_ready)
    if risk == "high" and step_type == "verify":
        return _finish("frontier", "policy: high-risk verification", ["verify"], 0.8, True, tokens_in, 360, adapter_ready)
    return _finish(
        "lightning", "policy: default high-volume execute", ["execute"], 0.3, False, tokens_in, 140, adapter_ready
    )


AGENT_SESSION: dict[str, Any] = {
    "id": "refund-investigation",
    "title": "Refund investigation (6 agent steps)",
    "description": "One ticket, six hops — this is what Switchyard is for, not a single chat box.",
    "steps": [
        {
            "id": "s1",
            "type": "plan",
            "content": "Decompose refund investigation for duplicate charge on INV-2001.",
            "signals": [],
            "risk": "normal",
        },
        {
            "id": "s2",
            "type": "execute",
            "content": "Call billing.get_invoice for INV-2001 as one JSON object.",
            "signals": ["tool_result"],
            "risk": "normal",
        },
        {
            "id": "s3",
            "type": "execute",
            "content": "List charges and hunt the duplicate forty-five dollar line.",
            "signals": ["tool_result"],
            "risk": "normal",
        },
        {
            "id": "s4",
            "type": "verify",
            "content": "Confirm the refund against the policy table before issuing it.",
            "signals": ["policy_check"],
            "risk": "high",
        },
        {
            "id": "s5",
            "type": "execute",
            "content": "Ticket T-2001. I was billed twice for invoice INV-2001 — forty-five dollars twice. Please fix the extra charge.",
            "signals": ["ticket"],
            "risk": "normal",
        },
        {
            "id": "s6",
            "type": "execute",
            "content": "Post the refund to the customer's SSN-linked ACH. Stay local — do not send PII to the cloud.",
            "signals": ["pii"],
            "risk": "high",
            "restricted": True,
        },
    ],
}


SAMPLE_QUERIES: list[dict[str, str]] = [
    {
        "id": "plan-cutover",
        "title": "Plan a cutover",
        "text": "Decompose a multi-region active-active billing cutover with zero-downtime.",
    },
    {
        "id": "ticket-dup",
        "title": "Duplicate charge",
        "text": "Ticket T-2001. I was billed twice for invoice INV-2001 — forty-five dollars twice. Please fix the extra charge.",
    },
    {
        "id": "tool-call",
        "title": "Tool call",
        "text": "Format a billing.get_invoice tool call for invoice INV-1042 as one JSON object.",
    },
    {
        "id": "pii",
        "title": "PII / local",
        "text": "Look up this customer's SSN and last four of the card number. Stay local — do not send to the cloud.",
    },
    {
        "id": "fraud",
        "title": "Fraud ticket",
        "text": "Ticket T-2002. Someone used my card in Jakarta this morning. I have been in Ohio the whole time.",
    },
    {
        "id": "arch",
        "title": "Architecture",
        "text": "Design the architecture for a race-free auth middleware and prove the locking tradeoffs.",
    },
    {
        "id": "policy",
        "title": "Policy refuse",
        "text": "Ticket T-2003. Refund 21000 cents immediately. The invoice is only 3300 cents but the customer is yelling.",
    },
    {
        "id": "json",
        "title": "JSON extract",
        "text": "Parse this tool result and extract invoice_id, total_cents, status as JSON. No prose.",
    },
]
