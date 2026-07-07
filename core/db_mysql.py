"""
db_mysql.py — MySQL 数据库后端

实现与 db_sqlite.py 完全相同的公开函数签名，使用 pymysql 驱动。
通过 DB_ENGINE=mysql 环境变量切换到此后端。

需要的环境变量：
    MYSQL_HOST      — MySQL 主机地址（默认 127.0.0.1）
    MYSQL_PORT      — MySQL 端口（默认 3306）
    MYSQL_USER      — MySQL 用户名（默认 root）
    MYSQL_PASSWORD  — MySQL 密码
    MYSQL_DATABASE  — MySQL 数据库名（默认 training_memory）

公开函数：
    utc_now, connect, init_db, ensure_session, get_recent_memory,
    get_session, get_session_turns, save_turn, update_turn_audio_bytes,
    save_session_evaluation, recent_completed_sessions, end_session
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pymysql
import pymysql.cursors


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


MYSQL_HOST = _env("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(_env("MYSQL_PORT", "3306"))
MYSQL_USER = _env("MYSQL_USER", "root")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "")
MYSQL_DATABASE = _env("MYSQL_DATABASE", "training_memory")


# ── 工具函数 ────────────────────────────────────────────

def utc_now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串（秒精度，带 Z 后缀）。"""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ── 连接管理 ────────────────────────────────────────────

def connect() -> pymysql.Connection:
    """创建 MySQL 数据库连接。

    使用 autocommit=True 匹配 SQLite 的自动提交行为，
    使用 DictCursor 使查询结果直接返回 dict。
    """
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    return conn


# ── 表结构初始化 ────────────────────────────────────────

def init_db() -> None:
    """初始化数据库表结构（首次运行时自动创建，后续运行幂等）。

    使用 ENGINE=InnoDB + utf8mb4 字符集，
    同时通过 _ensure_columns 做增量列迁移（向后兼容）。
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id VARCHAR(64) PRIMARY KEY,
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
                    memory_text TEXT NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    session_id VARCHAR(64) NOT NULL,
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
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    session_id VARCHAR(64) NOT NULL,
                    turn_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    memory_text TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    CREATE INDEX idx_conversation_turns_session
                        ON conversation_turns(session_id, turn_index)
                    """
                )
            except Exception:
                pass  # 索引已存在则跳过
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    CREATE INDEX idx_conversation_memory_session
                        ON conversation_memory(session_id, turn_index)
                    """
                )
            except Exception:
                pass  # 索引已存在则跳过

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


def _ensure_columns(conn: pymysql.Connection, table: str, columns: dict[str, str]) -> None:
    """增量列迁移：通过 INFORMATION_SCHEMA 检查表中是否缺少指定列。

    MySQL 版本等效于 SQLite 的 PRAGMA table_info。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
            """,
            (table,),
        )
        existing = {row["COLUMN_NAME"] for row in cur.fetchall()}
    for name, ddl in columns.items():
        if name not in existing:
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{name}` {ddl}")


# ── 行转换工具 ──────────────────────────────────────────

def _row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """将 MySQL DictCursor 返回的 dict 中的 JSON 字段解析为 Python 对象。"""
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

    使用 MySQL 的 ON DUPLICATE KEY UPDATE 语法替代 SQLite 的 ON CONFLICT。
    """
    now = utc_now()
    is_complete = int(bool(training.get("training_complete")))
    completed_at = now if is_complete else training.get("completed_at")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_sessions (
                    session_id, created_at, updated_at,
                    stage_id, customer_id, difficulty_id, voice_id, avatar_id,
                    case_id, current_state, final_state, is_complete, is_success, completed_at,
                    memory_text
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    updated_at = VALUES(updated_at),
                    stage_id = VALUES(stage_id),
                    customer_id = VALUES(customer_id),
                    difficulty_id = VALUES(difficulty_id),
                    voice_id = VALUES(voice_id),
                    avatar_id = VALUES(avatar_id),
                    case_id = VALUES(case_id),
                    current_state = VALUES(current_state),
                    final_state = VALUES(final_state),
                    is_complete = VALUES(is_complete),
                    is_success = VALUES(is_success),
                    completed_at = VALUES(completed_at)
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
                    "",
                ),
            )


def get_recent_memory(session_id: str, limit: int = 10) -> str:
    """获取指定会话最近 N 轮的对话文本，用于注入下一轮 prompt 作为上下文记忆。"""
    init_db()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT turn_index, user_text, assistant_text
                FROM conversation_turns
                WHERE session_id = %s
                ORDER BY turn_index DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()

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
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM conversation_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
    return _row_to_dict(row)


def get_session_turns(session_id: str) -> list[dict[str, Any]]:
    """获取指定会话的所有对话轮次（按 turn_index 升序排列）。"""
    init_db()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM conversation_turns
                WHERE session_id = %s
                ORDER BY turn_index ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
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
    2. 插入 conversation_turns 记录
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
        with conn.cursor() as cur:
            cur.execute(
                "SELECT turn_count FROM conversation_sessions WHERE session_id = %s",
                (session_id,),
            )
            current = cur.fetchone()
        turn_index = int(current["turn_count"] if current else 0) + 1

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_turns (
                    session_id, turn_index, created_at,
                    user_text, assistant_text,
                    stage_id, customer_id, difficulty_id, voice_id, avatar_id,
                    metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            turn_id = int(cur.lastrowid)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = %s, turn_count = %s
                WHERE session_id = %s
                """,
                (now, turn_index, session_id),
            )

    # 更新会话记忆快照
    memory_text = get_recent_memory(session_id)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = %s, memory_text = %s
                WHERE session_id = %s
                """,
                (now, memory_text, session_id),
            )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_memory (
                    session_id, turn_index, created_at, memory_text, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, turn_index, now, memory_text, metadata_json),
            )

    return turn_id, turn_index


def update_turn_audio_bytes(turn_id: int, audio_bytes: int) -> None:
    """更新轮次记录的音频字节数（TTS 合成完成后调用）。"""
    init_db()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversation_turns SET audio_bytes = %s WHERE id = %s",
                (audio_bytes, turn_id),
            )


def save_session_evaluation(session_id: str, evaluation: dict[str, Any]) -> None:
    """保存训练会话的评分结果（训练完成后由 evaluator 调用）。"""
    init_db()
    now = utc_now()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = %s, evaluation_json = %s
                WHERE session_id = %s
                """,
                (now, json.dumps(evaluation, ensure_ascii=False), session_id),
            )


def recent_completed_sessions(limit: int = 3) -> list[dict[str, Any]]:
    """获取最近完成的 N 个训练会话（用于首页展示历史训练记录）。"""
    init_db()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM conversation_sessions
                WHERE is_complete = 1
                ORDER BY COALESCE(completed_at, updated_at) DESC
                LIMIT %s
                """,
                (max(limit, 0),),
            )
            rows = cur.fetchall()
    return [_row_to_dict(row) or {} for row in rows]


def end_session(session_id: str, end_reason: str = "manual") -> bool:
    """标记已有轮次的会话为已完成。"""
    init_db()
    now = utc_now()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT turn_count, final_state FROM conversation_sessions WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row or int(row.get("turn_count") or 0) == 0:
                return False
            cur.execute(
                """
                UPDATE conversation_sessions
                SET is_complete = 1,
                    completed_at = %s,
                    updated_at = %s,
                    final_state = COALESCE(final_state, %s)
                WHERE session_id = %s
                """,
                (now, now, end_reason, session_id),
            )
            return True
