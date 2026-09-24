"""FastAPI application with ticket classification, routing, and feedback endpoints."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.classify import classify_ticket_llm
from app.db import get_all_tickets, get_ticket, init_db, save_feedback, save_ticket
from app.rag import (
    add_confirmed_ticket_to_vector_store,
    check_direct_rag_match,
    query_similar_tickets,
    seed_chroma_if_empty,
)
from app.router import (
    ResolutionType,
    Team,
    clean_ticket_text,
    evaluate_routing,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ticket_router")


# Request/Response schemas
class TicketCreateRequest(BaseModel):
    text: str = Field(..., description="Customer ticket content", min_length=1)
    user_id: Optional[str] = Field(default="anonymous", description="User identifier")


class TicketResponse(BaseModel):
    id: int
    assigned_team: str
    confidence: float
    resolved_by: str
    tokens_used: int


class ConfirmFeedbackRequest(BaseModel):
    correct_team: str = Field(..., description="Confirmed correct team name")


class ConfirmFeedbackResponse(BaseModel):
    message: str
    ticket_id: int
    assigned_team: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup initialization: prepare SQLite tables and seed ChromaDB vector store."""
    logger.info("Initializing SQLite database...")
    init_db()
    logger.info("Seeding ChromaDB if empty...")
    seeded_count = seed_chroma_if_empty()
    logger.info(f"ChromaDB ready with {seeded_count} documents.")
    yield


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ticket Understanding & Routing Agent",
    description="Minimal token usage ticket classifier powered by SQLite, RAG, and SentenceTransformers.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for browser access and Swagger UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import FileResponse

# Serve Frontend UI
@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve web frontend."""
    return FileResponse("static/index.html")


@app.post(
    "/ticket",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit and route a support ticket",
)
def create_ticket(payload: TicketCreateRequest) -> Dict[str, Any]:
    """Process ticket through 5-step pipeline:

    1. Understand: Clean text with string/regex (NO LLM).
    2. Retrieve Context (RAG): If top similarity > 0.85 -> rag_direct (0 tokens).
    3. Classify (LLM): If RAG not confident, single LLM call.
    4. Route: Escalate to GENERAL_TRIAGE if confidence < 0.60.
    5. Save: Store to SQLite and return response.
    """
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticket text cannot be empty.",
        )

    # Step 1: Understand (Clean text)
    cleaned = clean_ticket_text(raw_text)
    if not cleaned:
        # If text became empty after stripping signatures/HTML, fallback safely
        cleaned = raw_text

    # Step 2: Retrieve Context (RAG)
    retrieved_examples = query_similar_tickets(cleaned, k=3)
    direct_match = check_direct_rag_match(retrieved_examples)

    if direct_match:
        team_candidate, sim_score = direct_match
        resolved_by = ResolutionType.RAG_DIRECT.value
        confidence = sim_score
        tokens_used = 0
    else:
        # Step 3: Classify (Only runs if step 2 similarity <= 0.85)
        llm_result = classify_ticket_llm(cleaned, retrieved_examples)
        team_candidate = llm_result.get("team", Team.GENERAL_TRIAGE.value)
        confidence = llm_result.get("confidence", 0.0)
        resolved_by = ResolutionType.LLM_CLASSIFIED.value
        tokens_used = llm_result.get("tokens_used", 0)

    # Step 4: Route (Evaluate thresholds & escalate if confidence < 0.6)
    assigned_team, confidence, resolved_by = evaluate_routing(
        assigned_team=team_candidate,
        confidence=confidence,
        resolved_by=resolved_by,
    )

    # Step 5: Save to SQLite
    ticket_id = save_ticket(
        text=raw_text,
        cleaned_text=cleaned,
        assigned_team=assigned_team,
        confidence=confidence,
        resolved_by=resolved_by,
        tokens_used=tokens_used,
    )

    return {
        "id": ticket_id,
        "assigned_team": assigned_team,
        "confidence": round(confidence, 4),
        "resolved_by": resolved_by,
        "tokens_used": tokens_used,
    }


@app.post(
    "/ticket/{ticket_id}/confirm",
    response_model=ConfirmFeedbackResponse,
    summary="Confirm correct team for feedback loop",
)
def confirm_ticket_team(
    ticket_id: int, payload: ConfirmFeedbackRequest
) -> Dict[str, Any]:
    """Feedback Loop: Update ticket in SQLite and add to ChromaDB vector store."""
    correct_team = payload.correct_team.strip().upper()
    if correct_team not in Team._value2member_map_:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid team '{payload.correct_team}'. Allowed teams: {[t.value for t in Team]}",
        )

    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found.",
        )

    # Update SQLite record
    save_feedback(ticket_id, correct_team)

    # Add to ChromaDB vector store for dynamic future direct matches
    ticket_text = ticket.get("cleaned_text") or ticket.get("text")
    add_confirmed_ticket_to_vector_store(
        ticket_id=ticket_id,
        text=ticket_text,
        team=correct_team,
    )

    logger.info(
        f"[FEEDBACK] Ticket #{ticket_id} confirmed as team '{correct_team}' and indexed into vector store."
    )

    return {
        "message": "Feedback recorded and vector store updated successfully.",
        "ticket_id": ticket_id,
        "assigned_team": correct_team,
    }


@app.get(
    "/ticket/{ticket_id}",
    summary="Get a ticket record by ID",
)
def fetch_ticket(ticket_id: int) -> Dict[str, Any]:
    """Retrieve ticket details from SQLite."""
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found.",
        )
    return ticket


@app.get(
    "/tickets",
    summary="List all tickets",
)
def list_tickets() -> List[Dict[str, Any]]:
    """Retrieve all tickets from SQLite."""
    return get_all_tickets()
