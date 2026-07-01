from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "training_memory.sqlite3"
DB_PATH = Path(os.getenv("TRAINING_DB_PATH", DEFAULT_DB_PATH))


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stage_id TEXT,
                customer_id TEXT,
                difficulty_id TEXT,
                voice_id TEXT,
                avatar_id TEXT,
                turn_count INTEGER NOT NULL DEFAULT 0,
                memory_text TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                stage_id TEXT,
                customer_id TEXT,
                difficulty_id TEXT,
                voice_id TEXT,
                avatar_id TEXT,
                audio_bytes INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
                ON conversation_turns(session_id, turn_index);

            CREATE INDEX IF NOT EXISTS idx_conversation_memory_session
                ON conversation_memory(session_id, turn_index);
            """
        )


def ensure_session(session_id: str, training: dict[str, Any]) -> None:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_sessions (
                session_id, created_at, updated_at,
                stage_id, customer_id, difficulty_id, voice_id, avatar_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                stage_id=excluded.stage_id,
                customer_id=excluded.customer_id,
                difficulty_id=excluded.difficulty_id,
                voice_id=excluded.voice_id,
                avatar_id=excluded.avatar_id
            """,
            (
                session_id,
                now,
                now,
                training.get("stage_id"),
                training.get("customer_id"),
                training.get("difficulty_id"),
                training.get("voice_id"),
                training.get("avatar_id"),
            ),
        )


def get_recent_memory(session_id: str, limit: int = 10) -> str:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT turn_index, user_text, assistant_text
            FROM conversation_turns
            WHERE session_id = ?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    if not rows:
        return ""

    lines = []
    for row in reversed(rows):
        lines.append(f"第{row['turn_index']}轮 员工：{row['user_text']}")
        lines.append(f"第{row['turn_index']}轮 客户：{row['assistant_text']}")
    return "\n".join(lines)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if data.get("metadata_json"):
        try:
            data["metadata"] = json.loads(data["metadata_json"])
        except json.JSONDecodeError:
            data["metadata"] = {}
    data.pop("metadata_json", None)
    return data


def get_session(session_id: str) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM conversation_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    return _row_to_dict(row)


def get_session_turns(session_id: str) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM conversation_turns
            WHERE session_id = ?
            ORDER BY turn_index ASC
            """,
            (session_id,),
        ).fetchall()
    return [_row_to_dict(row) or {} for row in rows]


def save_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    training: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[int, int]:
    init_db()
    ensure_session(session_id, training)
    now = utc_now()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with connect() as conn:
        current = conn.execute(
            "SELECT turn_count FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_index = int(current["turn_count"] if current else 0) + 1
        cursor = conn.execute(
            """
            INSERT INTO conversation_turns (
                session_id, turn_index, created_at,
                user_text, assistant_text,
                stage_id, customer_id, difficulty_id, voice_id, avatar_id,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_index,
                now,
                user_text,
                assistant_text,
                training.get("stage_id"),
                training.get("customer_id"),
                training.get("difficulty_id"),
                training.get("voice_id"),
                training.get("avatar_id"),
                metadata_json,
            ),
        )
        turn_id = int(cursor.lastrowid)
        conn.execute(
            """
            UPDATE conversation_sessions
            SET updated_at = ?, turn_count = ?
            WHERE session_id = ?
            """,
            (now, turn_index, session_id),
        )

    memory_text = get_recent_memory(session_id)
    with connect() as conn:
        conn.execute(
            """
            UPDATE conversation_sessions
            SET updated_at = ?, memory_text = ?
            WHERE session_id = ?
            """,
            (now, memory_text, session_id),
        )
        conn.execute(
            """
            INSERT INTO conversation_memory (
                session_id, turn_index, created_at, memory_text, metadata_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, turn_index, now, memory_text, metadata_json),
        )

    return turn_id, turn_index


def update_turn_audio_bytes(turn_id: int, audio_bytes: int) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE conversation_turns SET audio_bytes = ? WHERE id = ?",
            (audio_bytes, turn_id),
        )


init_db()
