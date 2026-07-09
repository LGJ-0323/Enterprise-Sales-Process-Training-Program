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
                trainer_user_id TEXT,
                trainer_external_user_id TEXT,
                trainer_name TEXT,
                trainer_department TEXT,
                trainer_source TEXT,
                trainer_avatar_url TEXT,
                case_id TEXT,
                current_state TEXT,
                final_state TEXT,
                is_complete INTEGER NOT NULL DEFAULT 0,
                is_success INTEGER NOT NULL DEFAULT 0,
                completed_at TEXT,
                evaluation_json TEXT,
                case_snapshot_json TEXT,
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

            CREATE TABLE IF NOT EXISTS training_users (
                user_id TEXT PRIMARY KEY,
                source TEXT NOT NULL DEFAULT 'local',
                external_user_id TEXT,
                display_name TEXT,
                department TEXT,
                avatar_url TEXT,
                mobile TEXT,
                email TEXT,
                role TEXT,
                raw_profile_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_seen_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_conversation_turns_session
                ON conversation_turns(session_id, turn_index);

            CREATE INDEX IF NOT EXISTS idx_conversation_memory_session
                ON conversation_memory(session_id, turn_index);

            CREATE INDEX IF NOT EXISTS idx_training_users_source_external
                ON training_users(source, external_user_id);
            """
        )
        _ensure_columns(
            conn,
            "conversation_sessions",
            {
                "trainer_user_id": "TEXT",
                "trainer_external_user_id": "TEXT",
                "trainer_name": "TEXT",
                "trainer_department": "TEXT",
                "trainer_source": "TEXT",
                "trainer_avatar_url": "TEXT",
                "case_id": "TEXT",
                "current_state": "TEXT",
                "final_state": "TEXT",
                "is_complete": "INTEGER NOT NULL DEFAULT 0",
                "is_success": "INTEGER NOT NULL DEFAULT 0",
                "completed_at": "TEXT",
                "evaluation_json": "TEXT",
                "case_snapshot_json": "TEXT",
            },
        )
        _ensure_columns(
            conn,
            "training_users",
            {
                "source": "TEXT NOT NULL DEFAULT 'local'",
                "external_user_id": "TEXT",
                "display_name": "TEXT",
                "department": "TEXT",
                "avatar_url": "TEXT",
                "mobile": "TEXT",
                "email": "TEXT",
                "role": "TEXT",
                "raw_profile_json": "TEXT",
                "created_at": "TEXT",
                "updated_at": "TEXT",
                "last_seen_at": "TEXT",
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
    if data.get("case_snapshot_json"):
        try:
            data["case_snapshot"] = json.loads(data["case_snapshot_json"])
        except json.JSONDecodeError:
            data["case_snapshot"] = {}
    data.pop("case_snapshot_json", None)
    return data


# ── CRUD 操作 ───────────────────────────────────────────

def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        value = " / ".join(str(item) for item in value if item is not None)
    value = str(value).strip()
    return value or None


def _first(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _text(source.get(key))
        if value:
            return value
    return None


def _profile_json(user: dict[str, Any]) -> str | None:
    raw = user.get("raw_profile")
    if raw is None:
        raw = user.get("profile")
    if raw is None:
        return _text(user.get("raw_profile_json"))
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(raw), ensure_ascii=False)


def _trainer_snapshot(training: dict[str, Any]) -> dict[str, str | None]:
    trainer = training.get("trainer")
    trainer_data = trainer if isinstance(trainer, dict) else {}
    merged = {**trainer_data, **training}
    return {
        "trainer_user_id": _first(
            merged, "trainer_user_id", "training_user_id", "user_id", "userid", "employee_id"
        ),
        "trainer_external_user_id": _first(
            merged, "trainer_external_user_id", "external_user_id", "wecom_userid", "wecom_user_id", "userid"
        ),
        "trainer_name": _first(
            merged, "trainer_name", "display_name", "user_name", "name", "sales_name", "employee_name"
        ),
        "trainer_department": _first(
            merged, "trainer_department", "department", "department_name", "dept", "dept_name"
        ),
        "trainer_source": _first(merged, "trainer_source", "source", "identity_source"),
        "trainer_avatar_url": _first(merged, "trainer_avatar_url", "avatar_url", "avatar", "headimgurl"),
    }


def upsert_training_user(user: dict[str, Any]) -> dict[str, Any]:
    """Create or update a training user identity record."""
    init_db()
    source = _first(user, "source", "identity_source") or "local"
    external_user_id = _first(user, "external_user_id",
                              "wecom_userid", "wecom_user_id", "userid")
    user_id = _first(user, "user_id", "training_user_id", "employee_id")
    if not user_id and external_user_id:
        user_id = f"{source}:{external_user_id}"
    if not user_id:
        raise ValueError("training user requires user_id or external_user_id")

    now = utc_now()
    payload = {
        "user_id": user_id,
        "source": source,
        "external_user_id": external_user_id,
        "display_name": _first(user, "display_name", "trainer_name", "user_name", "name", "sales_name"),
        "department": _first(user, "department", "department_name", "dept", "dept_name"),
        "avatar_url": _first(user, "avatar_url", "trainer_avatar_url", "avatar", "headimgurl"),
        "mobile": _first(user, "mobile", "phone", "telephone"),
        "email": _first(user, "email", "mail"),
        "role": _first(user, "role", "position", "title"),
        "raw_profile_json": _profile_json(user),
        "last_seen_at": _first(user, "last_seen_at") or now,
    }
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO training_users (
                user_id, source, external_user_id, display_name, department,
                avatar_url, mobile, email, role, raw_profile_json,
                created_at, updated_at, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                source=COALESCE(excluded.source, training_users.source),
                external_user_id=COALESCE(excluded.external_user_id, training_users.external_user_id),
                display_name=COALESCE(excluded.display_name, training_users.display_name),
                department=COALESCE(excluded.department, training_users.department),
                avatar_url=COALESCE(excluded.avatar_url, training_users.avatar_url),
                mobile=COALESCE(excluded.mobile, training_users.mobile),
                email=COALESCE(excluded.email, training_users.email),
                role=COALESCE(excluded.role, training_users.role),
                raw_profile_json=COALESCE(excluded.raw_profile_json, training_users.raw_profile_json),
                updated_at=excluded.updated_at,
                last_seen_at=COALESCE(excluded.last_seen_at, training_users.last_seen_at)
            """,
            (
                payload["user_id"],
                payload["source"],
                payload["external_user_id"],
                payload["display_name"],
                payload["department"],
                payload["avatar_url"],
                payload["mobile"],
                payload["email"],
                payload["role"],
                payload["raw_profile_json"],
                now,
                now,
                payload["last_seen_at"],
            ),
        )
    return get_training_user(user_id) or {"user_id": user_id}


def get_training_user(user_id: str) -> dict[str, Any] | None:
    """Fetch a training user identity record by internal user_id."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM training_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    data = _row_to_dict(row)
    if not data:
        return None
    if data.get("raw_profile_json"):
        try:
            data["raw_profile"] = json.loads(data["raw_profile_json"])
        except json.JSONDecodeError:
            data["raw_profile"] = data["raw_profile_json"]
    data.pop("raw_profile_json", None)
    return data


def ensure_session(session_id: str, training: dict[str, Any]) -> None:
    """创建或更新会话记录（UPSERT）。

    首次对话时 INSERT 新会话，后续对话 UPDATE 状态字段。
    当 training 中 training_complete=True 时标记会话完成。
    """
    now = utc_now()
    is_complete = int(bool(training.get("training_complete")))
    completed_at = now if is_complete else training.get("completed_at")
    trainer = _trainer_snapshot(training)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_sessions (
                session_id, created_at, updated_at,
                stage_id, customer_id, difficulty_id, voice_id, avatar_id,
                trainer_user_id, trainer_external_user_id, trainer_name,
                trainer_department, trainer_source, trainer_avatar_url,
                case_id, current_state, final_state, is_complete, is_success, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                stage_id=excluded.stage_id,
                customer_id=excluded.customer_id,
                difficulty_id=excluded.difficulty_id,
                voice_id=excluded.voice_id,
                avatar_id=excluded.avatar_id,
                trainer_user_id=COALESCE(excluded.trainer_user_id, conversation_sessions.trainer_user_id),
                trainer_external_user_id=COALESCE(excluded.trainer_external_user_id, conversation_sessions.trainer_external_user_id),
                trainer_name=COALESCE(excluded.trainer_name, conversation_sessions.trainer_name),
                trainer_department=COALESCE(excluded.trainer_department, conversation_sessions.trainer_department),
                trainer_source=COALESCE(excluded.trainer_source, conversation_sessions.trainer_source),
                trainer_avatar_url=COALESCE(excluded.trainer_avatar_url, conversation_sessions.trainer_avatar_url),
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
                trainer["trainer_user_id"],
                trainer["trainer_external_user_id"],
                trainer["trainer_name"],
                trainer["trainer_department"],
                trainer["trainer_source"],
                trainer["trainer_avatar_url"],
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


def save_session_snapshot(session_id: str, snapshot: dict[str, Any]) -> None:
    """保存会话训练快照（在训练开始时调用，冻结选中的 case、标签、配置）。"""
    init_db()
    now = utc_now()
    selected_config = snapshot.get("selected_config") if isinstance(snapshot.get("selected_config"), dict) else {}
    training_summary = snapshot.get("training_summary") if isinstance(snapshot.get("training_summary"), dict) else {}
    merged = {**selected_config, **training_summary, **snapshot}
    trainer = _trainer_snapshot(merged)
    snapshot_json = json.dumps(snapshot, ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_sessions (
                session_id, created_at, updated_at,
                stage_id, difficulty_id, voice_id, avatar_id,
                trainer_user_id, trainer_external_user_id, trainer_name,
                trainer_department, trainer_source, trainer_avatar_url,
                case_id, case_snapshot_json, memory_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                stage_id=COALESCE(excluded.stage_id, conversation_sessions.stage_id),
                difficulty_id=COALESCE(excluded.difficulty_id, conversation_sessions.difficulty_id),
                voice_id=COALESCE(excluded.voice_id, conversation_sessions.voice_id),
                avatar_id=COALESCE(excluded.avatar_id, conversation_sessions.avatar_id),
                trainer_user_id=COALESCE(excluded.trainer_user_id, conversation_sessions.trainer_user_id),
                trainer_external_user_id=COALESCE(excluded.trainer_external_user_id, conversation_sessions.trainer_external_user_id),
                trainer_name=COALESCE(excluded.trainer_name, conversation_sessions.trainer_name),
                trainer_department=COALESCE(excluded.trainer_department, conversation_sessions.trainer_department),
                trainer_source=COALESCE(excluded.trainer_source, conversation_sessions.trainer_source),
                trainer_avatar_url=COALESCE(excluded.trainer_avatar_url, conversation_sessions.trainer_avatar_url),
                case_id=COALESCE(excluded.case_id, conversation_sessions.case_id),
                case_snapshot_json=excluded.case_snapshot_json
            """,
            (
                session_id,
                now,
                now,
                _first(merged, "stage_id"),
                _first(merged, "difficulty_id", "diff_id"),
                _first(merged, "voice_id"),
                _first(merged, "avatar_id"),
                trainer["trainer_user_id"],
                trainer["trainer_external_user_id"],
                trainer["trainer_name"],
                trainer["trainer_department"],
                trainer["trainer_source"],
                trainer["trainer_avatar_url"],
                _first(merged, "case_id", "base_case_id"),
                snapshot_json,
                "",
            ),
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
