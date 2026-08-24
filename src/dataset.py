"""DeskCard SFT corpus — teach a small model a rigid ticket-card contract.

The stock student writes apologies. After LoRA it emits one parseable card.
This is the same specialization story as Nemotron 3.5 Lightning SFT:
high-volume execute steps get a specialist, not a generic chat model.
"""

from __future__ import annotations

from typing import Any

SYSTEM = (
    "You are DeskCard, a first-line support clerk. "
    "Reply with one ticket card and nothing else."
)

SCHEMA_SYSTEM = (
    SYSTEM
    + "\n\nThe card must be exactly this shape:\n"
    "<<TICKET>>\n"
    "id: T-NNNN\n"
    "verdict: REFUND | NO_REFUND | ESCALATE\n"
    "amount_cents: <integer>\n"
    "reason_code: DUPLICATE | FRAUD | POLICY | OTHER\n"
    "note: one short sentence\n"
    "<</TICKET>>\n"
    "Use only those verdict and reason_code values. "
    "Copy the ticket id from the user. "
    "amount_cents is the extra charge to refund, or 0."
)

VERDICTS = ("REFUND", "NO_REFUND", "ESCALATE")
REASONS = ("DUPLICATE", "FRAUD", "POLICY", "OTHER")


def card_text(ex: dict[str, Any]) -> str:
    g = ex["gold"]
    return (
        "<<TICKET>>\n"
        f"id: {g['id']}\n"
        f"verdict: {g['verdict']}\n"
        f"amount_cents: {g['amount_cents']}\n"
        f"reason_code: {g['reason_code']}\n"
        f"note: {ex['note']}\n"
        "<</TICKET>>"
    )


def as_sft_row(ex: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ex["id"],
        "split": ex["split"],
        "family": ex["family"],
        "user": ex["user"],
        "gold": ex["gold"],
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["user"]},
        ],
        "completion": card_text(ex),
    }


TRAIN: list[dict[str, Any]] = [
    {
        "id": "T-1042",
        "split": "train",
        "family": "duplicate",
        "user": (
            "Ticket T-1042. Customer says they were charged twice for the same "
            "Pro seat. Invoice shows two $49.99 lines. Refund the extra charge."
        ),
        "gold": {"id": "T-1042", "verdict": "REFUND", "amount_cents": 4999, "reason_code": "DUPLICATE"},
        "note": "Duplicate Pro seat line; refund the extra 4999 cents.",
    },
    {
        "id": "T-2108",
        "split": "train",
        "family": "duplicate",
        "user": (
            "Ticket T-2108. Billing listed the annual plan twice. Customer "
            "wants the second 7500-cent charge reversed."
        ),
        "gold": {"id": "T-2108", "verdict": "REFUND", "amount_cents": 7500, "reason_code": "DUPLICATE"},
        "note": "Annual plan billed twice; refund the second 7500 cents.",
    },
    {
        "id": "T-3302",
        "split": "train",
        "family": "fraud",
        "user": (
            "Ticket T-3302. Customer was in Ohio all week. Card just paid "
            "for a laptop in Jakarta. They did not authorize it."
        ),
        "gold": {"id": "T-3302", "verdict": "ESCALATE", "amount_cents": 0, "reason_code": "FRAUD"},
        "note": "Unauthorized foreign charge; escalate to fraud review.",
    },
    {
        "id": "T-4410",
        "split": "train",
        "family": "fraud",
        "user": (
            "Ticket T-4410. Forty failed logins overnight, then a password "
            "reset from a new device. Customer says that was not them."
        ),
        "gold": {"id": "T-4410", "verdict": "ESCALATE", "amount_cents": 0, "reason_code": "FRAUD"},
        "note": "Credential-stuffing pattern; escalate, do not refund here.",
    },
    {
        "id": "T-8001",
        "split": "train",
        "family": "policy",
        "user": (
            "Ticket T-8001. Angry customer wants a 9000-cent refund right now. "
            "The invoice total is only 2000 cents."
        ),
        "gold": {"id": "T-8001", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "POLICY"},
        "note": "Asked amount exceeds invoice total; refuse under policy.",
    },
    {
        "id": "T-5519",
        "split": "train",
        "family": "policy",
        "user": (
            "Ticket T-5519. Purchase was 11 months ago. Customer wants a "
            "full refund. Our window is 30 days."
        ),
        "gold": {"id": "T-5519", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "POLICY"},
        "note": "Outside the 30-day refund window; no refund.",
    },
    {
        "id": "T-6622",
        "split": "train",
        "family": "other",
        "user": (
            "Ticket T-6622. Customer wants a refund because they do not like "
            "the new UI color. No billing error, no outage."
        ),
        "gold": {"id": "T-6622", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "OTHER"},
        "note": "Taste complaint is not a billing defect; no refund.",
    },
    {
        "id": "T-7730",
        "split": "train",
        "family": "other",
        "user": (
            "Ticket T-7730. VIP cannot download their invoice PDF. The billing "
            "page returns HTTP 500. They need a human."
        ),
        "gold": {"id": "T-7730", "verdict": "ESCALATE", "amount_cents": 0, "reason_code": "OTHER"},
        "note": "Broken invoice download for a VIP; escalate to engineering.",
    },
]

HOLDOUT: list[dict[str, Any]] = [
    {
        "id": "T-8844",
        "split": "holdout",
        "family": "duplicate",
        "user": (
            "Ticket T-8844. Two identical 3200-cent charges landed on the "
            "same invoice. Please reverse the extra one."
        ),
        "gold": {"id": "T-8844", "verdict": "REFUND", "amount_cents": 3200, "reason_code": "DUPLICATE"},
        "note": "Identical 3200-cent lines; refund the extra charge.",
    },
    {
        "id": "T-9901",
        "split": "holdout",
        "family": "fraud",
        "user": (
            "Ticket T-9901. MFA was bypassed and the account signed in from "
            "two countries in ten minutes. Customer is locked out."
        ),
        "gold": {"id": "T-9901", "verdict": "ESCALATE", "amount_cents": 0, "reason_code": "FRAUD"},
        "note": "Impossible travel plus MFA bypass; escalate to security.",
    },
    {
        "id": "T-1015",
        "split": "holdout",
        "family": "policy",
        "user": (
            "Ticket T-1015. Customer demands a 99999-cent refund. Invoice "
            "total is 1250 cents."
        ),
        "gold": {"id": "T-1015", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "POLICY"},
        "note": "Refund request larger than the invoice; refuse.",
    },
    {
        "id": "T-1120",
        "split": "holdout",
        "family": "other",
        "user": (
            "Ticket T-1120. Package arrived one day late. Customer wants "
            "the full 16600 cents back. Delivery itself succeeded."
        ),
        "gold": {"id": "T-1120", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "OTHER"},
        "note": "One-day late delivery is not a billing defect; no refund.",
    },
]

TEST: list[dict[str, Any]] = [
    {
        "id": "T-2001",
        "split": "test",
        "family": "duplicate",
        "user": (
            "Ticket T-2001. I was billed twice for invoice INV-2001 — "
            "forty-five dollars twice. Please fix the extra charge."
        ),
        "gold": {"id": "T-2001", "verdict": "REFUND", "amount_cents": 4500, "reason_code": "DUPLICATE"},
        "note": "Double 4500-cent charge; refund the extra line.",
    },
    {
        "id": "T-2002",
        "split": "test",
        "family": "fraud",
        "user": (
            "Ticket T-2002. Someone used my card in Jakarta this morning. "
            "I have been in Ohio the whole time and I did not buy anything."
        ),
        "gold": {"id": "T-2002", "verdict": "ESCALATE", "amount_cents": 0, "reason_code": "FRAUD"},
        "note": "Unauthorized overseas charge; escalate to fraud.",
    },
    {
        "id": "T-2003",
        "split": "test",
        "family": "policy",
        "user": (
            "Ticket T-2003. Refund 21000 cents immediately. The invoice "
            "is only 3300 cents but the customer is yelling."
        ),
        "gold": {"id": "T-2003", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "POLICY"},
        "note": "Requested refund exceeds invoice total; refuse.",
    },
    {
        "id": "T-2004",
        "split": "test",
        "family": "other",
        "user": (
            "Ticket T-2004. The new icon is ugly. Cancel my plan and "
            "refund 6700 cents. Nothing is broken."
        ),
        "gold": {"id": "T-2004", "verdict": "NO_REFUND", "amount_cents": 0, "reason_code": "OTHER"},
        "note": "Aesthetic preference is not a defect; no refund.",
    },
]


def split_rows(name: str) -> list[dict[str, Any]]:
    src = {"train": TRAIN, "holdout": HOLDOUT, "test": TEST}[name]
    return [as_sft_row(x) for x in src]


def all_examples() -> list[dict[str, Any]]:
    return [as_sft_row(x) for x in TRAIN + HOLDOUT + TEST]
