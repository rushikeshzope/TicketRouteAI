"""SQLite database helper functions and schema setup (pure sqlite3)."""

import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_FILE = os.getenv("SQLITE_DB_PATH", "tickets.db")


def get_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Create and return a database connection with dictionary row access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_FILE) -> None:
    """Initialize SQLite database and create the tickets table if not exists."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                assigned_team TEXT NOT NULL,
                confidence REAL NOT NULL,
                resolved_by TEXT NOT NULL,
                tokens_used INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_ticket(
    text: str,
    cleaned_text: str,
    assigned_team: str,
    confidence: float,
    resolved_by: str,
    tokens_used: int,
    db_path: str = DB_FILE,
) -> int:
    """Insert a new ticket record and return the generated ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tickets (
                text,
                cleaned_text,
                assigned_team,
                confidence,
                resolved_by,
                tokens_used
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                cleaned_text,
                assigned_team,
                float(confidence),
                resolved_by,
                int(tokens_used),
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore


def get_ticket(ticket_id: int, db_path: str = DB_FILE) -> Optional[Dict[str, Any]]:
    """Retrieve a single ticket record by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_all_tickets(db_path: str = DB_FILE) -> List[Dict[str, Any]]:
    """Retrieve all ticket records ordered by creation time descending."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def save_feedback(ticket_id: int, correct_team: str, db_path: str = DB_FILE) -> bool:
    """Update assigned_team for a confirmed ticket."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE tickets
            SET assigned_team = ?
            WHERE id = ?
            """,
            (correct_team, ticket_id),
        )
        conn.commit()
        return cursor.rowcount > 0
