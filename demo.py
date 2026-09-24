"""Interactive demonstration script for the Ticket Understanding & Routing Agent.

Runs through each step of the pipeline with live sample inputs and prints
clean, formatted outputs showing tokens used, assigned teams, and resolution origin.
"""

import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


from app.db import init_db
from app.rag import seed_chroma_if_empty

def print_divider(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    # Initialize DB and Chroma collections
    init_db()
    seed_chroma_if_empty()

    print_divider("TICKET UNDERSTANDING & ROUTING AGENT DEMO")

    # Scenario 1: Exact / Near-Identical Seed Ticket (RAG Direct Match -> 0 Tokens)
    print_divider("Scenario 1: High-Similarity Query (RAG Direct Shortcut)")
    ticket_1 = {
        "text": "I can't log into my account, please assist.\n\nBest regards,\nAlice",
        "user_id": "user_alice",
    }
    print(f"Incoming Ticket:\n{ticket_1['text']}\n")
    res1 = client.post("/ticket", json=ticket_1)
    print("API Response:")
    print(json.dumps(res1.json(), indent=2))
    assert res1.json()["resolved_by"] == "rag_direct"
    assert res1.json()["tokens_used"] == 0

    # Scenario 2: Novel Domain Ticket (LLM Classification)
    print_divider("Scenario 2: Novel/Unseen Query (LLM Classification)")
    ticket_2 = {
        "text": "The monthly AWS cloud cost allocation report is missing the regional breakdown table for APAC.",
        "user_id": "user_bob",
    }
    print(f"Incoming Ticket:\n{ticket_2['text']}\n")
    res2 = client.post("/ticket", json=ticket_2)
    print("API Response:")
    print(json.dumps(res2.json(), indent=2))
    ticket_2_id = res2.json()["id"]

    # Scenario 3: Feedback Confirmation & Continuous Learning
    print_divider("Scenario 3: Feedback Loop & Dynamic ChromaDB Indexing")
    print(f"Confirming Ticket #{ticket_2_id} to team 'REPORTS'...")
    confirm_res = client.post(
        f"/ticket/{ticket_2_id}/confirm",
        json={"correct_team": "REPORTS"},
    )
    print("Confirmation Response:")
    print(json.dumps(confirm_res.json(), indent=2))

    print("\nRe-submitting the exact same novel query to test dynamic RAG caching...")
    res2_repeat = client.post("/ticket", json=ticket_2)
    print("Repeat Query API Response (Now 0 tokens via RAG direct match!):")
    print(json.dumps(res2_repeat.json(), indent=2))
    assert res2_repeat.json()["resolved_by"] == "rag_direct"
    assert res2_repeat.json()["tokens_used"] == 0

    # Scenario 4: Ambiguous Query (Escalation to GENERAL_TRIAGE)
    print_divider("Scenario 4: Ambiguous Query (Escalated to GENERAL_TRIAGE)")
    ticket_4 = {
        "text": "Something somewhere isn't right and I need someone to check.",
        "user_id": "user_charlie",
    }
    print(f"Incoming Ticket:\n{ticket_4['text']}\n")
    res4 = client.post("/ticket", json=ticket_4)
    print("API Response:")
    print(json.dumps(res4.json(), indent=2))

    # Scenario 5: Database Inspection
    print_divider("Scenario 5: SQLite Database Inspection (GET /tickets)")
    all_tickets_res = client.get("/tickets")
    print(f"Total tickets stored in SQLite: {len(all_tickets_res.json())}")
    for t in all_tickets_res.json()[:3]:
        print(f"Ticket #{t['id']}: Team={t['assigned_team']} | Method={t['resolved_by']} | Tokens={t['tokens_used']}")

    print_divider("DEMO COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
