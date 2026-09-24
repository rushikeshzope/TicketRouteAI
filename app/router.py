"""Team definitions, text cleaning, and routing logic."""

import logging
import re
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class Team(str, Enum):
    """Fixed team categories for ticket routing."""
    CUSTOMER_CARE = "CUSTOMER_CARE"
    REPORTS = "REPORTS"
    FINANCE = "FINANCE"
    PRODUCT_UPDATES = "PRODUCT_UPDATES"
    GENERAL_TRIAGE = "GENERAL_TRIAGE"


class ResolutionType(str, Enum):
    """Classification resolution origins."""
    RAG_DIRECT = "rag_direct"
    LLM_CLASSIFIED = "llm_classified"
    ESCALATED = "escalated"


# Configurable thresholds
RAG_SIMILARITY_THRESHOLD = 0.85
CONFIDENCE_ROUTING_THRESHOLD = 0.60


def clean_ticket_text(raw_text: str) -> str:
    """Clean ticket text using regex/string operations (NO LLM).

    - Strips HTML tags
    - Strips common email signatures and footers
    - Normalizes multiple whitespaces/newlines
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Remove common email signatures / footers
    # Cut off standard signature markers like '-- \n', 'Best regards,', etc.
    signature_patterns = [
        r"(?i)\n\s*--\s*\n.*$",
        r"(?i)\n\s*--\s*$",
        r"(?i)\n\s*(?:best regards|warm regards|regards|thanks & regards|thanks|sincerely|cheers),?.*$",
        r"(?i)\n\s*sent from my (?:iphone|android|galaxy|ipad|device).*$",
        r"(?i)\n\s*get outlook for (?:ios|android).*$",
    ]
    for pattern in signature_patterns:
        text = re.split(pattern, text, flags=re.DOTALL)[0]

    # 3. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.strip()

    return text


def evaluate_routing(
    assigned_team: str,
    confidence: float,
    resolved_by: str,
) -> Tuple[str, float, str]:
    """Apply routing threshold rules.

    If confidence < 0.6, override assigned_team to GENERAL_TRIAGE and set resolved_by to 'escalated'.
    Logs the team routing simulation.
    """
    final_team = assigned_team
    final_resolved_by = resolved_by
    final_confidence = float(confidence)

    # Validate team validity against enum
    if final_team not in Team._value2member_map_:
        final_team = Team.GENERAL_TRIAGE.value
        final_resolved_by = ResolutionType.ESCALATED.value

    # Confidence check
    if final_confidence < CONFIDENCE_ROUTING_THRESHOLD:
        final_team = Team.GENERAL_TRIAGE.value
        final_resolved_by = ResolutionType.ESCALATED.value

    # Simulate routing queue push via logging
    logger.info(
        f"[QUEUE_ROUTE] Ticket routed to team: '{final_team}' "
        f"(confidence={final_confidence:.2f}, resolved_by='{final_resolved_by}')"
    )
    print(
        f"[QUEUE_ROUTE] Routed ticket to team -> {final_team} "
        f"[Confidence: {final_confidence:.2f} | Method: {final_resolved_by}]"
    )

    return final_team, final_confidence, final_resolved_by
