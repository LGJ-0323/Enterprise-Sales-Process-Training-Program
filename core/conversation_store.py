"""
conversation_store.py — 会话持久化门面层

根据 DB_ENGINE 环境变量自动选择数据库后端：
    - sqlite（默认）：使用 Python 内置 sqlite3 驱动，数据文件存于 data/training_memory.sqlite3
    - mysql：使用 pymysql 驱动，连接信息通过 MYSQL_HOST / MYSQL_PORT / MYSQL_USER /
             MYSQL_PASSWORD / MYSQL_DATABASE 环境变量配置

所有上层调用方直接 import 此模块即可，无需感知底层数据库类型。
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_project_env(path: Path) -> None:
    """Load project .env before selecting the persistence backend."""
    if not path.exists():
        return
    override = _truthy_env(os.getenv("DOTENV_OVERRIDE"))
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "DOTENV_OVERRIDE" and _truthy_env(value):
                override = True
                break
    except OSError:
        pass
    if load_dotenv:
        load_dotenv(path, override=override)
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


PROJECT_DIR = Path(__file__).resolve().parents[1]
_load_project_env(PROJECT_DIR / ".env")

_backend_name = os.getenv("DB_ENGINE", "sqlite").strip().lower()

if _backend_name == "mysql":
    try:
        from .db_mysql import (
            connect,
            end_session,
            ensure_session,
            get_recent_memory,
            get_session,
            get_session_turns,
            get_training_user,
            init_db,
            recent_completed_sessions,
            save_session_evaluation,
            save_session_snapshot,
            save_turn,
            update_turn_audio_bytes,
            upsert_training_user,
            utc_now,
        )
    except ImportError as exc:
        raise ImportError(
            "DB_ENGINE=mysql 需要安装 pymysql 驱动，请执行: pip install pymysql"
        ) from exc
else:
    from .db_sqlite import (
        connect,
        end_session,
        ensure_session,
        get_recent_memory,
        get_session,
        get_session_turns,
        get_training_user,
        init_db,
        recent_completed_sessions,
        save_session_evaluation,
        save_session_snapshot,
        save_turn,
        update_turn_audio_bytes,
        upsert_training_user,
        utc_now,
    )


# SQLite 文件库可在导入时安全初始化；MySQL 仍由后端函数内部懒初始化。
if _backend_name == "sqlite":
    init_db()
