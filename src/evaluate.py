"""Score a generation against the DeskCard contract."""

from __future__ import annotations

import re
from typing import Any

from .dataset import REASONS, VERDICTS

CARD_RE = re.compile(
    r"<<TICKET>>\s*"
    r"id:\s*(?P<id>\S+)\s*"
    r"verdict:\s*(?P<verdict>\S+)\s*"
    r"amount_cents:\s*(?P<amount>-?\d+)\s*"
    r"reason_code:\s*(?P<reason>\S+)\s*"
    r"note:\s*(?P<note>.+?)\s*"
    r"<</TICKET>>",
    re.DOTALL | re.IGNORECASE,
)


def parse_card(text: str) -> dict[str, Any] | None:
    m = CARD_RE.search(text or "")
    if not m:
        return None
    return {
        "id": m.group("id").strip(),
        "verdict": m.group("verdict").strip().upper(),
        "amount_cents": int(m.group("amount")),
        "reason_code": m.group("reason").strip().upper(),
        "note": m.group("note").strip(),
    }


def score_generation(text: str, gold: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (text or "").strip()
    parsed = parse_card(raw)
    has_tags = "<<TICKET>>" in raw.upper() and "<</TICKET>>" in raw.upper()
    format_ok = bool(
        parsed
        and parsed["verdict"] in VERDICTS
        and parsed["reason_code"] in REASONS
        and has_tags
    )
    gold = gold or {}
    id_ok = bool(parsed and parsed["id"] == gold.get("id"))
    verdict_ok = bool(parsed and parsed["verdict"] == gold.get("verdict"))
    amount_ok = bool(parsed and parsed["amount_cents"] == gold.get("amount_cents"))
    reason_ok = bool(parsed and parsed["reason_code"] == gold.get("reason_code"))
    task_ok = format_ok and id_ok and verdict_ok and amount_ok and reason_ok
    return {
        "has_ticket_tags": has_tags,
        "parsed_ok": parsed is not None,
        "format_pass": format_ok,
        "id_match": id_ok,
        "verdict_match": verdict_ok,
        "amount_match": amount_ok,
        "reason_match": reason_ok,
        "task_pass": task_ok,
        "parsed": parsed,
        "preview": raw[:240],
    }
