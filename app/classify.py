"""Single-point LLM classification wrapper with strict JSON output formatting.

Supports both Anthropic Claude (via ANTHROPIC_API_KEY) and OpenAI (via OPENAI_API_KEY),
with an automatic intelligent fallback when no API key is set.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.router import Team

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    'You are a support ticket classifier. Choose exactly one team from this list: '
    'CUSTOMER_CARE, REPORTS, FINANCE, PRODUCT_UPDATES, GENERAL_TRIAGE. '
    'Use the reference examples to help decide. Respond only with valid JSON in this '
    'exact format: {"team": "<TEAM>", "confidence": <0-1 float>, "reason": "<max 15 words>"}. '
    'No other text.'
)


def truncate_tokens(text: str, max_tokens: int = 150) -> str:
    """Truncate text to roughly max_tokens (approx 4 chars per token / word boundary)."""
    if not text:
        return ""
    max_words = int(max_tokens * 0.75)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def estimate_tokens(text: str) -> int:
    """Estimate token count as len(text) // 4 (standard heuristic)."""
    return max(1, len(text) // 4)


def parse_and_validate_llm_json(raw_response: str) -> Dict[str, Any]:
    """Extract and strictly validate JSON payload from LLM output."""
    fallback_result = {
        "team": Team.GENERAL_TRIAGE.value,
        "confidence": 0.0,
        "reason": "Fallback default",
    }

    if not raw_response or not raw_response.strip():
        return fallback_result

    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return fallback_result

        team_val = str(data.get("team", "")).strip().upper()
        if team_val not in Team._value2member_map_:
            team_val = Team.GENERAL_TRIAGE.value

        confidence_val = data.get("confidence", 0.0)
        try:
            confidence_float = float(confidence_val)
            confidence_float = max(0.0, min(1.0, confidence_float))
        except (ValueError, TypeError):
            confidence_float = 0.0

        reason_val = str(data.get("reason", "LLM classified")).strip()
        reason_words = reason_val.split()
        if len(reason_words) > 15:
            reason_val = " ".join(reason_words[:15])

        return {
            "team": team_val,
            "confidence": confidence_float,
            "reason": reason_val,
        }
    except Exception as e:
        logger.warning(f"Failed to parse LLM JSON response: {e}. Raw response: {raw_response}")
        return fallback_result


def _heuristic_fallback_classify(
    cleaned_text: str, retrieved_examples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Heuristic fallback when no external API key is configured."""
    lower = cleaned_text.lower()
    selected_team = Team.GENERAL_TRIAGE.value
    confidence = 0.50
    reason = "Fallback triage"

    if retrieved_examples:
        top = retrieved_examples[0]
        top_sim = top.get("similarity", 0.0)
        if top_sim >= 0.50:
            selected_team = top.get("team", Team.GENERAL_TRIAGE.value)
            confidence = round(top_sim, 2)
            reason = f"Matched {top.get('team')}"

    if any(w in lower for w in ["invoice", "charge", "refund", "billing", "payment", "subscription", "credit card", "price"]):
        selected_team = Team.FINANCE.value
        confidence = 0.88
        reason = "Finance keywords detected"
    elif any(w in lower for w in ["report", "analytics", "dashboard", "export", "metrics", "data", "stats"]):
        selected_team = Team.REPORTS.value
        confidence = 0.85
        reason = "Reporting keywords detected"
    elif any(w in lower for w in ["crash", "bug", "feature", "dark mode", "upload", "update", "error 500", "version"]):
        selected_team = Team.PRODUCT_UPDATES.value
        confidence = 0.86
        reason = "Product keyword detected"
    elif any(w in lower for w in ["login", "password", "order", "account", "delivery", "track"]):
        selected_team = Team.CUSTOMER_CARE.value
        confidence = 0.87
        reason = "Customer care keyword detected"

    prompt_tokens = estimate_tokens(cleaned_text)
    completion_tokens = 25

    return {
        "team": selected_team,
        "confidence": confidence,
        "reason": reason,
        "tokens_used": prompt_tokens + completion_tokens,
    }


def classify_ticket_llm(
    cleaned_text: str,
    retrieved_examples: List[Dict[str, Any]],
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Make single LLM call to classify ticket with strict JSON output format.

    Checks for ANTHROPIC_API_KEY (Claude) first, then OPENAI_API_KEY,
    and falls back to local heuristic classification if no key is configured.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Format user prompt with truncated reference examples
    example_lines = []
    for i, ex in enumerate(retrieved_examples[:3], 1):
        truncated_doc = truncate_tokens(ex.get("text", ""), max_tokens=150)
        team_name = ex.get("team", "GENERAL_TRIAGE")
        example_lines.append(f"Example {i}: \"{truncated_doc}\" -> Team: {team_name}")

    examples_block = "\n".join(example_lines) if example_lines else "No reference examples available."
    user_prompt = f"Ticket:\n\"{cleaned_text}\"\n\nReference Examples:\n{examples_block}"

    # 1. Claude / Anthropic integration
    if anthropic_key:
        model = model_name or os.getenv("LLM_MODEL", "claude-3-5-haiku-latest")
        try:
            # pyrefly: ignore [missing-import]
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model=model,
                max_tokens=60,  # Cap output tokens strictly
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

            raw_content = ""
            if response.content:
                raw_content = "".join(block.text for block in response.content if hasattr(block, "text"))

            parsed = parse_and_validate_llm_json(raw_content)

            # Retrieve exact token metrics from Claude API response
            tokens_used = 0
            if hasattr(response, "usage") and response.usage:
                tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
            else:
                tokens_used = estimate_tokens(user_prompt) + estimate_tokens(raw_content)

            parsed["tokens_used"] = tokens_used
            return parsed

        except Exception as e:
            logger.error(f"Claude API call failed: {e}. Falling back to default triage.")
            fallback = _heuristic_fallback_classify(cleaned_text, retrieved_examples)
            fallback["reason"] = f"Claude fallback: {str(e)[:25]}"
            return fallback

    # 2. OpenAI integration
    if openai_key:
        model = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=60,
            )

            raw_content = response.choices[0].message.content or ""
            parsed = parse_and_validate_llm_json(raw_content)

            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.total_tokens
            else:
                tokens_used = estimate_tokens(user_prompt) + estimate_tokens(raw_content)

            parsed["tokens_used"] = tokens_used
            return parsed

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}. Falling back to default triage.")
            fallback = _heuristic_fallback_classify(cleaned_text, retrieved_examples)
            fallback["reason"] = f"OpenAI fallback: {str(e)[:25]}"
            return fallback

    # 3. Offline heuristic fallback
    logger.info("No LLM API key detected (ANTHROPIC_API_KEY or OPENAI_API_KEY). Using smart fallback.")
    return _heuristic_fallback_classify(cleaned_text, retrieved_examples)
