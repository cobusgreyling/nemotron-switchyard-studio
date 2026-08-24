from src.router import AGENT_SESSION, classify
from src.evaluate import score_generation
from src.dataset import TEST, as_sft_row


def test_plan_goes_frontier():
    d = classify("Decompose a multi-region active-active billing cutover.")
    assert d.track == "frontier"


def test_ticket_goes_lightning():
    d = classify("Ticket T-2001. I was billed twice for invoice INV-2001.")
    assert d.track == "lightning"


def test_pii_goes_local():
    d = classify("Look up this SSN and stay local — do not send to the cloud.")
    assert d.track == "local"


def test_stage_session_uses_all_three_tracks():
    tracks = [
        classify(
            s["content"],
            strategy="stage",
            step_type=s.get("type"),
            signals=s.get("signals"),
            risk=s.get("risk", "normal"),
            restricted=s.get("restricted"),
        ).track
        for s in AGENT_SESSION["steps"]
    ]
    assert "frontier" in tracks
    assert "lightning" in tracks
    assert "local" in tracks
    assert tracks[0] == "frontier"
    assert tracks[-1] == "local"


def test_card_scorer():
    ex = as_sft_row(TEST[0])
    assert score_generation(ex["completion"], ex["gold"])["task_pass"] is True
    assert score_generation("sorry about the charge", ex["gold"])["task_pass"] is False
