"""End-to-end and unit tests for Ticket Understanding & Routing Agent."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.classify import parse_and_validate_llm_json, truncate_tokens
from app.db import get_ticket, init_db
from app.main import app
from app.rag import get_chroma_collection, seed_chroma_if_empty
from app.router import Team, clean_ticket_text, evaluate_routing


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure database and seed data are initialized for test suite."""
    init_db()
    seed_chroma_if_empty()


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


def test_rag_direct_match_zero_tokens(client):
    """Test 1: Submit a ticket text nearly identical to a seed example.

    Assert resolved_by == 'rag_direct' and tokens_used == 0.
    """
    payload = {
        "text": "I can't log into my account please help",
        "user_id": "user_101",
    }
    response = client.post("/ticket", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["assigned_team"] == Team.CUSTOMER_CARE.value
    assert data["resolved_by"] == "rag_direct"
    assert data["tokens_used"] == 0
    assert data["confidence"] > 0.85
    assert "id" in data


def test_novel_ticket_llm_classified(client):
    """Test 2: Submit an ambiguous / novel ticket text.

    Assert resolved_by == 'llm_classified' when similarity is below threshold.
    """
    payload = {
        "text": "The custom ERP export plugin failed to generate the annual shareholder breakdown report.",
        "user_id": "user_202",
    }
    response = client.post("/ticket", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["resolved_by"] == "llm_classified"
    assert data["tokens_used"] > 0
    assert data["confidence"] >= 0.60
    assert data["assigned_team"] in [Team.REPORTS.value, Team.PRODUCT_UPDATES.value]


def test_low_confidence_escalation_to_general_triage(client):
    """Test 3: Mock a low-confidence LLM response.

    Assert it gets escalated to GENERAL_TRIAGE with resolved_by == 'escalated'.
    """
    mock_llm_output = {
        "team": Team.FINANCE.value,
        "confidence": 0.45,  # below 0.60 threshold
        "reason": "Unsure about intent",
        "tokens_used": 42,
    }

    with patch("app.main.classify_ticket_llm", return_value=mock_llm_output):
        payload = {
            "text": "Super confusing query that matches nothing specifically xzy123.",
            "user_id": "user_303",
        }
        response = client.post("/ticket", json=payload)
        assert response.status_code == 201
        data = response.json()

        assert data["assigned_team"] == Team.GENERAL_TRIAGE.value
        assert data["resolved_by"] == "escalated"
        assert data["tokens_used"] == 42


def test_feedback_loop_updates_vector_store_and_db(client):
    """Test 6: Submit a ticket, confirm corrected team, verify DB & Chroma update."""
    # 1. Create a ticket that gets classified
    initial_payload = {
        "text": "Need clarification on the quarterly tax rebate deduction calculation.",
        "user_id": "user_404",
    }
    create_resp = client.post("/ticket", json=initial_payload)
    assert create_resp.status_code == 201
    ticket_id = create_resp.json()["id"]

    # 2. Confirm feedback to FINANCE
    confirm_resp = client.post(
        f"/ticket/{ticket_id}/confirm",
        json={"correct_team": Team.FINANCE.value},
    )
    assert confirm_resp.status_code == 200
    confirm_data = confirm_resp.json()
    assert confirm_data["assigned_team"] == Team.FINANCE.value

    # 3. Verify SQLite record reflects updated team
    get_resp = client.get(f"/ticket/{ticket_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["assigned_team"] == Team.FINANCE.value

    # 4. Now submit the exact same text again -> it should immediately hit rag_direct with 0 tokens
    repeat_resp = client.post("/ticket", json=initial_payload)
    assert repeat_resp.status_code == 201
    repeat_data = repeat_resp.json()
    assert repeat_data["assigned_team"] == Team.FINANCE.value
    assert repeat_data["resolved_by"] == "rag_direct"
    assert repeat_data["tokens_used"] == 0


def test_clean_ticket_text_sanitization():
    """Test string cleaning guardrail: HTML tag removal and email signature stripping."""
    raw_input = """
    <div><h1>Urgent!</h1><p>My order hasn't arrived yet.</p></div>
    
    Thanks & regards,
    John Doe
    VP of Logistics
    Sent from my iPhone
    """
    cleaned = clean_ticket_text(raw_input)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "John Doe" not in cleaned
    assert "Sent from my iPhone" not in cleaned
    assert "My order hasn't arrived yet." in cleaned


def test_json_parsing_resilience():
    """Test LLM response parser with markdown backticks and malformed inputs."""
    # Test with markdown code fences
    fenced_json = '```json\n{"team": "FINANCE", "confidence": 0.95, "reason": "Invoice issue"}\n```'
    parsed = parse_and_validate_llm_json(fenced_json)
    assert parsed["team"] == Team.FINANCE.value
    assert parsed["confidence"] == 0.95

    # Test with invalid JSON syntax
    broken_json = '{"team": "FINANCE", "confidence": '
    fallback = parse_and_validate_llm_json(broken_json)
    assert fallback["team"] == Team.GENERAL_TRIAGE.value
    assert fallback["confidence"] == 0.0

    # Test with unknown team name
    invalid_team_json = '{"team": "UNKNOWN_DEPARTMENT", "confidence": 0.9, "reason": "test"}'
    parsed_invalid = parse_and_validate_llm_json(invalid_team_json)
    assert parsed_invalid["team"] == Team.GENERAL_TRIAGE.value


def test_input_guardrails_and_boundary_cases(client):
    """Test API input validation guardrails."""
    # 1. Empty text
    empty_resp = client.post("/ticket", json={"text": "   ", "user_id": "u1"})
    assert empty_resp.status_code in [400, 422]

    # 2. Non-existent ticket retrieval
    not_found_resp = client.get("/ticket/999999")
    assert not_found_resp.status_code == 404

    # 3. Invalid team confirmation
    invalid_confirm_resp = client.post(
        "/ticket/1/confirm", json={"correct_team": "NON_EXISTENT_TEAM"}
    )
    assert invalid_confirm_resp.status_code == 400

    # 4. List all tickets
    list_resp = client.get("/tickets")
    assert list_resp.status_code == 200
    assert isinstance(list_resp.json(), list)


def test_token_truncation_limits():
    """Test that context truncation enforces token limits."""
    long_text = "word " * 500
    truncated = truncate_tokens(long_text, max_tokens=150)
    words = truncated.split()
    assert len(words) <= int(150 * 0.75)
