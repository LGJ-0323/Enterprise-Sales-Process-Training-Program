"""
db_sqlite.py — SQLite 数据库后端

实现 conversation_store 所需全部数据库操作，使用 Python 内置 sqlite3 驱动。
此模块与 db_mysql.py 保持完全相同的公开函数签名，以便 conversation_store.py
门面层无差别切换。

公开函数：
    utc_now, connect, init_db, ensure_session, get_recent_memory,
    get_session, get_session_turns, save_turn, update_turn_audio_bytes,
    save_session_evaluation, recent_completed_sessions, end_session
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "training_memory.sqlite3"
DB_PATH = Path(os.getenv("TRAINING_DB_PATH") or DEFAULT_DB_PATH)


# ── 工具函数 ────────────────────────────────────────────

def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串（秒精度，带 Z 后缀）。"""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ── 连接管理 ────────────────────────────────────────────

def connect() -> sqlite3.Connection:
    """创建 SQLite 数据库连接，启用 WAL 模式和外键约束。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── 表结构初始化 ────────────────────────────────────────

def init_db() -> None:
    """初始化数据库表结构（首次运行时自动创建，后续运行幂等）。

    创建三张表：
    - conversation_sessions: 会话主表
    - conversation_turns:    对话轮次表
    - conversation_memory:   会话记忆快照表

    同时通过 _ensure_columns 做增量列迁移（向后兼容旧版本数据库）。
    """
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
                case_id TEXT,
                current_state TEXT,
                final_state TEXT,
                is_complete INTEGER NOT NULL DEFAULT 0,
                is_success INTEGER NOT NULL DEFAULT 0,
                completed_at TEXT,
                evaluation_json TEXT,
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
        _ensure_columns(
            conn,
            "conversation_sessions",
            {
                "case_id": "TEXT",
                "current_state": "TEXT",
                "final_state": "TEXT",
                "is_complete": "INTEGER NOT NULL DEFAULT 0",
                "is_success": "INTEGER NOT NULL DEFAULT 0",
                "completed_at": "TEXT",
                "evaluation_json": "TEXT",
            },
        )


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """增量列迁移：检查表中是否缺少指定列，缺少则自动 ALTER TABLE 添加。

    用于向后兼容旧版本数据库，避免重建表丢失数据。
    """
    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


# ── 行转换工具 ──────────────────────────────────────────

def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """将 sqlite3.Row 转为 dict，并自动解析 metadata_json 和 evaluation_json 字段。"""
    if row is None:
        return None
    data = dict(row)
    if data.get("metadata_json"):
        try:
            data["metadata"] = json.loads(data["metadata_json"])
        except json.JSONDecodeError:
            data["metadata"] = {}
    data.pop("metadata_json", None)
    if data.get("evaluation_json"):
        try:
            data["evaluation"] = json.loads(data["evaluation_json"])
        except json.JSONDecodeError:
            data["evaluation"] = {}
    data.pop("evaluation_json", None)
    return data


# ── CRUD 操作 ───────────────────────────────────────────

def ensure_session(session_id: str, training: dict[str, Any]) -> None:
    """创建或更新会话记录（UPSERT）。

    首次对话时 INSERT 新会话，后续对话 UPDATE 状态字段。
    当 training 中 training_complete=True 时标记会话完成。
    """
    now = utc_now()
    is_complete = int(bool(training.get("training_complete")))
    completed_at = now if is_complete else training.get("completed_at")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_sessions (
                session_id, created_at, updated_at,
                stage_id, customer_id, difficulty_id, voice_id, avatar_id,
                case_id, current_state, final_state, is_complete, is_success, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                stage_id=excluded.stage_id,
                customer_id=excluded.customer_id,
                difficulty_id=excluded.difficulty_id,
                voice_id=excluded.voice_id,
                avatar_id=excluded.avatar_id,
                case_id=excluded.case_id,
                current_state=excluded.current_state,
                final_state=excluded.final_state,
                is_complete=excluded.is_complete,
                is_success=excluded.is_success,
                completed_at=excluded.completed_at
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
                training.get("case_id"),
                training.get("current_state"),
                training.get("final_state"),
                is_complete,
                int(bool(training.get("is_success"))),
                completed_at,
            ),
        )


def get_recent_memory(session_id: str, limit: int = 10) -> str:
    """获取指定会话最近 N 轮的对话文本，用于注入下一轮 prompt 作为上下文记忆。

    返回格式：
        第1轮 员工：xxx
        第1轮 客户：xxx
        第2轮 员工：xxx
        第2轮 客户：xxx
    """
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


def get_session(session_id: str) -> dict[str, Any] | None:
    """根据 session_id 查询完整会话信息（含评分结果）。"""
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
    """获取指定会话的所有对话轮次（按 turn_index 升序排列）。"""
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
    """保存一轮对话记录，并更新会话记忆。

    执行流程：
    1. ensure_session() 确保会话记录存在
    2. 插入 conversation_turns 记录（含用户输入、客户回复、元数据）
    3. 更新会话的 turn_count 和 memory_text
    4. 插入 conversation_memory 快照记录

    Returns:
        (turn_id, turn_index): 新轮次的数据库 ID 和轮次序号
    """
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

    # 更新会话记忆快照
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
    """更新轮次记录的音频字节数（TTS 合成完成后调用）。"""
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE conversation_turns SET audio_bytes = ? WHERE id = ?",
            (audio_bytes, turn_id),
        )


def save_session_evaluation(session_id: str, evaluation: dict[str, Any]) -> None:
    """保存训练会话的评分结果（训练完成后由 evaluator 调用）。"""
    init_db()
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE conversation_sessions
            SET updated_at = ?, evaluation_json = ?
            WHERE session_id = ?
            """,
            (now, json.dumps(evaluation, ensure_ascii=False), session_id),
        )


def recent_completed_sessions(limit: int = 3) -> list[dict[str, Any]]:
    """获取最近完成的 N 个训练会话（用于首页展示历史训练记录）。"""
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM conversation_sessions
            WHERE is_complete = 1
            ORDER BY COALESCE(completed_at, updated_at) DESC
            LIMIT ?
            """,
            (max(limit, 0),),
        ).fetchall()
    return [_row_to_dict(row) or {} for row in rows]


def end_session(session_id: str, end_reason: str = "manual") -> bool:
    """标记已有轮次的会话为已完成。"""
    init_db()
    now = utc_now()
    with connect() as conn:
        row = conn.execute(
            "SELECT turn_count, final_state FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row or int(row["turn_count"] or 0) == 0:
            return False
        conn.execute(
            """
            UPDATE conversation_sessions
            SET is_complete = 1,
                completed_at = ?,
                updated_at = ?,
                final_state = COALESCE(final_state, ?)
            WHERE session_id = ?
            """,
            (now, now, end_reason, session_id),
        )
        return True
