"""Comprehensive live test script testing all API endpoints."""

import json
from fastapi.testclient import TestClient
from app.main import app
from app.db import init_db
from app.rag import seed_chroma_if_empty

client = TestClient(app)


def test_suite():
    init_db()
    seed_chroma_if_empty()

    print("\n" + "=" * 60)
    print("1. Testing POST /ticket (RAG Direct Shortcut)")
    print("=" * 60)
    req1 = {"text": "How do I reset my password", "user_id": "u1"}
    res1 = client.post("/ticket", json=req1)
    print(f"Status Code: {res1.status_code}")
    print("Response:", json.dumps(res1.json(), indent=2))
    assert res1.status_code == 201
    ticket_1_id = res1.json()["id"]

    print("\n" + "=" * 60)
    print("2. Testing GET /ticket/{id} (Fetch Ticket by ID)")
    print("=" * 60)
    res_get = client.get(f"/ticket/{ticket_1_id}")
    print(f"Status Code: {res_get.status_code}")
    print("Fetched Ticket Record:", json.dumps(res_get.json(), indent=2))
    assert res_get.status_code == 200
    assert res_get.json()["id"] == ticket_1_id

    print("\n" + "=" * 60)
    print("3. Testing POST /ticket/{id}/confirm (Feedback Loop)")
    print("=" * 60)
    res_confirm = client.post(
        f"/ticket/{ticket_1_id}/confirm",
        json={"correct_team": "CUSTOMER_CARE"},
    )
    print(f"Status Code: {res_confirm.status_code}")
    print("Confirm Response:", json.dumps(res_confirm.json(), indent=2))
    assert res_confirm.status_code == 200

    print("\n" + "=" * 60)
    print("4. Testing GET /tickets (List All Tickets)")
    print("=" * 60)
    res_all = client.get("/tickets")
    print(f"Status Code: {res_all.status_code}")
    print(f"Total tickets retrieved: {len(res_all.json())}")
    print("First 2 tickets in list:", json.dumps(res_all.json()[:2], indent=2))
    assert res_all.status_code == 200
    assert len(res_all.json()) > 0

    print("\n" + "=" * 60)
    print("5. Testing GET /ticket/999999 (Non-existent Ticket 404)")
    print("=" * 60)
    res_404 = client.get("/ticket/999999")
    print(f"Status Code: {res_404.status_code}")
    print("404 Response:", res_404.json())
    assert res_404.status_code == 404

    print("\n" + "=" * 60)
    print("6. Testing POST /ticket/{id}/confirm with Invalid Team (400 Bad Request)")
    print("=" * 60)
    res_bad_team = client.post(
        f"/ticket/{ticket_1_id}/confirm",
        json={"correct_team": "INVALID_TEAM_NAME"},
    )
    print(f"Status Code: {res_bad_team.status_code}")
    print("400 Response:", res_bad_team.json())
    assert res_bad_team.status_code == 400

    print("\n" + "=" * 60)
    print("ALL API ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_suite()
