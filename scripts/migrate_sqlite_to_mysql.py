"""
migrate_sqlite_to_mysql.py — SQLite → MySQL 数据迁移脚本

用途：
    将 data/training_memory.sqlite3 中的全部数据迁移到 MySQL 数据库。

用法：
    # 1. 确保 MySQL 服务已启动，目标数据库已创建：
    #    mysql -u root -e "CREATE DATABASE IF NOT EXISTS training_memory CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"

    # 2. 设置环境变量后运行（或直接修改下方的 MYSQL_* 常量）：
    set MYSQL_HOST=127.0.0.1
    set MYSQL_PORT=3306
    set MYSQL_USER=root
    set MYSQL_PASSWORD=your_password
    set MYSQL_DATABASE=training_memory
    python scripts/migrate_sqlite_to_mysql.py

    # 3. 迁移完成后，将 .env 中 DB_ENGINE 改为 mysql 即可切换后端。

安全特性：
    - 使用 ON DUPLICATE KEY UPDATE，多次运行幂等（不重复插入）
    - 按 FK 依赖顺序迁移：sessions → turns → memory
    - 保留原始 id 值，确保数据一致性
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path，以便在任意位置运行
PROJECT_DIR = Path(__file__).resolve().parents[1]

# 加载 .env 文件中的配置（MYSQL_PASSWORD 等）
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass
sys.path.insert(0, str(PROJECT_DIR))


# ── 配置 ────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


SQLITE_PATH = Path(_env("TRAINING_DB_PATH") or PROJECT_DIR /
                   "data" / "training_memory.sqlite3")

MYSQL_HOST = _env("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(_env("MYSQL_PORT", "3306"))
MYSQL_USER = _env("MYSQL_USER", "root")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "")
MYSQL_DATABASE = _env("MYSQL_DATABASE", "training_memory")


# ── 连接 ────────────────────────────────────────────────

def connect_sqlite() -> sqlite3.Connection:
    if not SQLITE_PATH.exists():
        print(f"[错误] SQLite 数据库文件不存在: {SQLITE_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def connect_mysql():
    try:
        import pymysql
    except ImportError:
        print("[错误] 需要安装 pymysql: pip install pymysql")
        sys.exit(1)

    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        autocommit=True,
    )
    return conn


def _row_to_dict(row) -> dict:
    """将 sqlite3.Row 转为纯 dict。"""
    return dict(row)


# ── MySQL 建表 ──────────────────────────────────────────

CREATE_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
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
    memory_text TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

CREATE_TRAINING_USERS_SQL = """
CREATE TABLE IF NOT EXISTS training_users (
    user_id VARCHAR(128) PRIMARY KEY,
    source VARCHAR(32) NOT NULL DEFAULT 'local',
    external_user_id VARCHAR(128),
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

CREATE_TURNS_SQL = """
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

CREATE_MEMORY_SQL = """
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

CREATE_INDEXES_SQL = [
    "CREATE INDEX idx_conversation_turns_session ON conversation_turns(session_id, turn_index)",
    "CREATE INDEX idx_conversation_memory_session ON conversation_memory(session_id, turn_index)",
    "CREATE INDEX idx_training_users_source_external ON training_users(source, external_user_id)",
]


def create_mysql_tables(conn) -> None:
    """在 MySQL 中创建所有表（幂等）。"""
    with conn.cursor() as cur:
        cur.execute(CREATE_SESSIONS_SQL)
        cur.execute(CREATE_TRAINING_USERS_SQL)
        cur.execute(CREATE_TURNS_SQL)
        cur.execute(CREATE_MEMORY_SQL)
        for sql in CREATE_INDEXES_SQL:
            try:
                cur.execute(sql)
            except Exception:
                pass  # 索引已存在则跳过
    print("[建表] MySQL 表结构已就绪")


# ── 数据迁移 ────────────────────────────────────────────

INSERT_SESSION_SQL = """
INSERT INTO conversation_sessions (
    session_id, created_at, updated_at,
    stage_id, customer_id, difficulty_id, voice_id, avatar_id,
    trainer_user_id, trainer_external_user_id, trainer_name,
    trainer_department, trainer_source, trainer_avatar_url,
    case_id, current_state, final_state, is_complete, is_success, completed_at,
    evaluation_json, case_snapshot_json, turn_count, memory_text
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    created_at = VALUES(created_at),
    updated_at = VALUES(updated_at),
    stage_id = VALUES(stage_id),
    customer_id = VALUES(customer_id),
    difficulty_id = VALUES(difficulty_id),
    voice_id = VALUES(voice_id),
    avatar_id = VALUES(avatar_id),
    trainer_user_id = VALUES(trainer_user_id),
    trainer_external_user_id = VALUES(trainer_external_user_id),
    trainer_name = VALUES(trainer_name),
    trainer_department = VALUES(trainer_department),
    trainer_source = VALUES(trainer_source),
    trainer_avatar_url = VALUES(trainer_avatar_url),
    case_id = VALUES(case_id),
    current_state = VALUES(current_state),
    final_state = VALUES(final_state),
    is_complete = VALUES(is_complete),
    is_success = VALUES(is_success),
    completed_at = VALUES(completed_at),
    evaluation_json = VALUES(evaluation_json),
    case_snapshot_json = VALUES(case_snapshot_json),
    turn_count = VALUES(turn_count),
    memory_text = VALUES(memory_text)
"""

INSERT_TRAINING_USER_SQL = """
INSERT INTO training_users (
    user_id, source, external_user_id, display_name, department,
    avatar_url, mobile, email, role, raw_profile_json,
    created_at, updated_at, last_seen_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    source = VALUES(source),
    external_user_id = VALUES(external_user_id),
    display_name = VALUES(display_name),
    department = VALUES(department),
    avatar_url = VALUES(avatar_url),
    mobile = VALUES(mobile),
    email = VALUES(email),
    role = VALUES(role),
    raw_profile_json = VALUES(raw_profile_json),
    updated_at = VALUES(updated_at),
    last_seen_at = VALUES(last_seen_at)
"""

INSERT_TURN_SQL = """
INSERT INTO conversation_turns (
    id, session_id, turn_index, created_at,
    user_text, assistant_text,
    stage_id, customer_id, difficulty_id, voice_id, avatar_id,
    audio_bytes, metadata_json
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    session_id = VALUES(session_id),
    turn_index = VALUES(turn_index),
    created_at = VALUES(created_at),
    user_text = VALUES(user_text),
    assistant_text = VALUES(assistant_text),
    stage_id = VALUES(stage_id),
    customer_id = VALUES(customer_id),
    difficulty_id = VALUES(difficulty_id),
    voice_id = VALUES(voice_id),
    avatar_id = VALUES(avatar_id),
    audio_bytes = VALUES(audio_bytes),
    metadata_json = VALUES(metadata_json)
"""

INSERT_MEMORY_SQL = """
INSERT INTO conversation_memory (
    id, session_id, turn_index, created_at,
    memory_text, metadata_json
)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    session_id = VALUES(session_id),
    turn_index = VALUES(turn_index),
    created_at = VALUES(created_at),
    memory_text = VALUES(memory_text),
    metadata_json = VALUES(metadata_json)
"""


def migrate_sessions(sqlite_conn, mysql_conn) -> int:
    rows = sqlite_conn.execute(
        "SELECT * FROM conversation_sessions").fetchall()
    count = 0
    with mysql_conn.cursor() as cur:
        for row in rows:
            d = _row_to_dict(row)
            cur.execute(
                INSERT_SESSION_SQL,
                (
                    d["session_id"],
                    d["created_at"],
                    d["updated_at"],
                    d.get("stage_id"),
                    d.get("customer_id"),
                    d.get("difficulty_id"),
                    d.get("voice_id"),
                    d.get("avatar_id"),
                    d.get("trainer_user_id"),
                    d.get("trainer_external_user_id"),
                    d.get("trainer_name"),
                    d.get("trainer_department"),
                    d.get("trainer_source"),
                    d.get("trainer_avatar_url"),
                    d.get("case_id"),
                    d.get("current_state"),
                    d.get("final_state"),
                    d.get("is_complete", 0),
                    d.get("is_success", 0),
                    d.get("completed_at"),
                    d.get("evaluation_json"),
                    d.get("case_snapshot_json"),
                    d.get("turn_count", 0),
                    d.get("memory_text", ""),
                ),
            )
            count += 1
    return count


def sqlite_table_exists(sqlite_conn, table_name: str) -> bool:
    row = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def migrate_training_users(sqlite_conn, mysql_conn) -> int:
    if not sqlite_table_exists(sqlite_conn, "training_users"):
        return 0
    rows = sqlite_conn.execute("SELECT * FROM training_users").fetchall()
    count = 0
    with mysql_conn.cursor() as cur:
        for row in rows:
            d = _row_to_dict(row)
            cur.execute(
                INSERT_TRAINING_USER_SQL,
                (
                    d["user_id"],
                    d.get("source") or "local",
                    d.get("external_user_id"),
                    d.get("display_name"),
                    d.get("department"),
                    d.get("avatar_url"),
                    d.get("mobile"),
                    d.get("email"),
                    d.get("role"),
                    d.get("raw_profile_json"),
                    d.get("created_at") or datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    d.get("updated_at") or datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    d.get("last_seen_at"),
                ),
            )
            count += 1
    return count


def migrate_turns(sqlite_conn, mysql_conn) -> int:
    rows = sqlite_conn.execute(
        "SELECT * FROM conversation_turns ORDER BY id").fetchall()
    count = 0
    with mysql_conn.cursor() as cur:
        for row in rows:
            d = _row_to_dict(row)
            cur.execute(
                INSERT_TURN_SQL,
                (
                    d["id"],
                    d["session_id"],
                    d["turn_index"],
                    d["created_at"],
                    d["user_text"],
                    d["assistant_text"],
                    d.get("stage_id"),
                    d.get("customer_id"),
                    d.get("difficulty_id"),
                    d.get("voice_id"),
                    d.get("avatar_id"),
                    d.get("audio_bytes", 0),
                    d.get("metadata_json"),
                ),
            )
            count += 1
    return count


def migrate_memory(sqlite_conn, mysql_conn) -> int:
    rows = sqlite_conn.execute(
        "SELECT * FROM conversation_memory ORDER BY id").fetchall()
    count = 0
    with mysql_conn.cursor() as cur:
        for row in rows:
            d = _row_to_dict(row)
            cur.execute(
                INSERT_MEMORY_SQL,
                (
                    d["id"],
                    d["session_id"],
                    d["turn_index"],
                    d["created_at"],
                    d["memory_text"],
                    d.get("metadata_json"),
                ),
            )
            count += 1
    return count


# ── 主流程 ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SQLite → MySQL 数据迁移")
    print("=" * 60)
    print(f"  源 (SQLite):  {SQLITE_PATH}")
    print(
        f"  目标 (MySQL): {MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
    print()

    # 连接
    print("[连接] 正在连接 SQLite ...")
    sqlite_conn = connect_sqlite()
    print(
        f"[连接] SQLite 已连接 (WAL: {sqlite_conn.execute('PRAGMA journal_mode').fetchone()[0]})")

    print("[连接] 正在连接 MySQL ...")
    try:
        mysql_conn = connect_mysql()
        print(f"[连接] MySQL 已连接 (server: {mysql_conn.get_server_info()})")
    except Exception as exc:
        print(f"[错误] 无法连接 MySQL: {exc}")
        print("  请确保:")
        print("    1. MySQL 服务已启动")
        print("    2. 数据库 training_memory 已创建")
        print("    3. 环境变量 MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD 正确")
        sys.exit(1)

    try:
        # 建表
        print()
        create_mysql_tables(mysql_conn)

        # 迁移数据（按 FK 依赖顺序）
        print()
        print("[迁移] 1/4 conversation_sessions ...")
        n_sessions = migrate_sessions(sqlite_conn, mysql_conn)
        print(f"[迁移]    → 已迁移 {n_sessions} 条记录")

        print("[迁移] 2/4 training_users ...")
        n_users = migrate_training_users(sqlite_conn, mysql_conn)
        print(f"[迁移]    → 已迁移 {n_users} 条记录")

        print("[迁移] 3/4 conversation_turns ...")
        n_turns = migrate_turns(sqlite_conn, mysql_conn)
        print(f"[迁移]    → 已迁移 {n_turns} 条记录")

        print("[迁移] 4/4 conversation_memory ...")
        n_memory = migrate_memory(sqlite_conn, mysql_conn)
        print(f"[迁移]    → 已迁移 {n_memory} 条记录")

        print()
        print("=" * 60)
        print("  迁移完成！")
        print(f"  sessions: {n_sessions}  条")
        print(f"  users:    {n_users}  条")
        print(f"  turns:    {n_turns}  条")
        print(f"  memory:   {n_memory}  条")
        print()
        print("  下一步：将 .env 中 DB_ENGINE 改为 mysql 并重启服务")
        print("=" * 60)

    finally:
        sqlite_conn.close()
        mysql_conn.close()


if __name__ == "__main__":
    main()
