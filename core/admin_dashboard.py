from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from starlette.responses import HTMLResponse, RedirectResponse

try:
    from .case_loader import get_case, get_case_summary
    from .conversation_store import connect, get_session, get_session_turns, get_training_user, init_db
except ImportError:
    from case_loader import get_case, get_case_summary
    from conversation_store import connect, get_session, get_session_turns, get_training_user, init_db


app = FastAPI(title="Training Admin Dashboard")

DB_ENGINE = os.getenv("DB_ENGINE", "sqlite").strip().lower()
PARAM = "%s" if DB_ENGINE == "mysql" else "?"

TRAINER_KEYS = (
    "trainer_name",
    "trainer_user_id",
    "trainer_external_user_id",
    "trainer",
    "training_user",
    "user_name",
    "user_id",
    "external_user_id",
    "wecom_userid",
    "wecom_user_id",
    "display_name",
    "sales_name",
    "employee_name",
    "operator",
)
UNKNOWN_TRAINER = "\u672a\u8bb0\u5f55"


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _execute_fetchall(sql: str, params: list[Any] | tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        if DB_ENGINE == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        else:
            rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def _metadata(turn: dict[str, Any]) -> dict[str, Any]:
    data = turn.get("metadata")
    if isinstance(data, dict):
        return data
    return _json_obj(turn.get("metadata_json"))


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = " / ".join(str(item) for item in value if item is not None)
    return str(value).strip()


def _first_value(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _text(source.get(key))
        if value:
            return value
    return ""


def _trainer_meta(turns: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    for turn in turns or []:
        meta = _metadata(turn)
        for key in TRAINER_KEYS:
            if meta.get(key):
                return meta
    return {}


def _lookup_training_user(user_id: str) -> dict[str, Any]:
    if not user_id:
        return {}
    try:
        return get_training_user(user_id) or {}
    except Exception:
        return {}


def _extract_trainer_detail(session: dict[str, Any], turns: list[dict[str, Any]] | None = None) -> dict[str, str]:
    meta = _trainer_meta(turns)
    trainer_user_id = _first_value(session, "trainer_user_id", "user_id", "training_user_id")
    if not trainer_user_id:
        trainer_user_id = _first_value(meta, "trainer_user_id", "user_id", "training_user_id")
    user = _lookup_training_user(trainer_user_id)
    external_user_id = (
        _first_value(session, "trainer_external_user_id", "external_user_id", "wecom_userid", "wecom_user_id")
        or _first_value(user, "external_user_id", "wecom_userid", "wecom_user_id")
        or _first_value(meta, "trainer_external_user_id", "external_user_id", "wecom_userid", "wecom_user_id")
    )
    name = (
        _first_value(session, "trainer_name", "display_name", "user_name", "name", "sales_name", "employee_name")
        or _first_value(user, "display_name", "trainer_name", "user_name", "name")
        or _first_value(meta, "trainer_name", "display_name", "user_name", "name", "sales_name", "employee_name")
        or trainer_user_id
        or external_user_id
        or UNKNOWN_TRAINER
    )
    return {
        "user_id": trainer_user_id,
        "external_user_id": external_user_id,
        "name": name,
        "department": (
            _first_value(session, "trainer_department", "department", "department_name")
            or _first_value(user, "department", "department_name")
            or _first_value(meta, "trainer_department", "department", "department_name")
        ),
        "source": (
            _first_value(session, "trainer_source", "source")
            or _first_value(user, "source")
            or _first_value(meta, "trainer_source", "source")
        ),
        "avatar_url": (
            _first_value(session, "trainer_avatar_url", "avatar_url")
            or _first_value(user, "avatar_url")
            or _first_value(meta, "trainer_avatar_url", "avatar_url")
        ),
    }


def _extract_trainer(session: dict[str, Any], turns: list[dict[str, Any]] | None = None) -> str:
    return _extract_trainer_detail(session, turns).get("name") or UNKNOWN_TRAINER


def _score(evaluation: dict[str, Any]) -> int | float | None:
    value = evaluation.get("total_score")
    if value is None:
        return None
    try:
        score = float(value)
        return int(score) if score.is_integer() else round(score, 1)
    except (TypeError, ValueError):
        return None


def _session_payload(row: dict[str, Any], include_trainer: bool = True) -> dict[str, Any]:
    evaluation = _json_obj(row.get("evaluation_json")) or _json_obj(row.get("evaluation"))
    turns = get_session_turns(row.get("session_id", "")) if include_trainer else []
    trainer_detail = _extract_trainer_detail(row, turns) if include_trainer else {"name": UNKNOWN_TRAINER}
    case = get_case(row.get("case_id"))
    case_summary = get_case_summary(case) if case else {}
    return {
        "session_id": row.get("session_id", ""),
        "trainer": trainer_detail.get("name") or UNKNOWN_TRAINER,
        "trainer_detail": trainer_detail,
        "stage_id": row.get("stage_id") or "",
        "difficulty_id": row.get("difficulty_id") or "",
        "voice_id": row.get("voice_id") or "",
        "case_id": row.get("case_id") or "",
        "case_scene": case_summary.get("scene") or (case or {}).get("scene", ""),
        "business_line": case_summary.get("business_line") or "",
        "customer_name": case_summary.get("customer_name") or "",
        "turn_count": row.get("turn_count") or 0,
        "is_complete": bool(row.get("is_complete")),
        "is_success": bool(row.get("is_success")),
        "final_state": row.get("final_state") or "",
        "created_at": row.get("created_at") or "",
        "updated_at": row.get("updated_at") or "",
        "completed_at": row.get("completed_at") or "",
        "score": _score(evaluation),
        "evaluation_source": evaluation.get("source") or "",
        "evaluation": evaluation,
    }


def _fetch_sessions(limit: int, status: str, query: str) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if status == "completed":
        clauses.append("is_complete = 1")
    elif status == "active":
        clauses.append("is_complete = 0")
    if query:
        like = f"%{query}%"
        clauses.append(
            "("
            f"session_id LIKE {PARAM} OR stage_id LIKE {PARAM} OR difficulty_id LIKE {PARAM} "
            f"OR customer_id LIKE {PARAM} OR case_id LIKE {PARAM} "
            f"OR trainer_name LIKE {PARAM} OR trainer_user_id LIKE {PARAM} OR trainer_external_user_id LIKE {PARAM}"
            ")"
        )
        params.extend([like, like, like, like, like, like, like, like])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    safe_limit = max(1, min(int(limit), 200))
    sql = f"""
        SELECT *
        FROM conversation_sessions
        {where}
        ORDER BY COALESCE(completed_at, updated_at, created_at) DESC
        LIMIT {safe_limit}
    """
    rows = _execute_fetchall(sql, params)
    return [_session_payload(row) for row in rows]


def _stats(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [s["score"] for s in sessions if s.get("score") is not None]
    return {
        "listed_count": len(sessions),
        "completed_count": sum(1 for s in sessions if s.get("is_complete")),
        "active_count": sum(1 for s in sessions if not s.get("is_complete")),
        "avg_score": round(sum(scored) / len(scored), 1) if scored else None,
    }


@app.get("/")
async def index():
    return RedirectResponse(url="/admin")


@app.get("/admin")
async def admin_page():
    return HTMLResponse(ADMIN_HTML)


@app.get("/api/admin/sessions")
async def api_sessions(
    limit: int = Query(30, ge=1, le=200),
    status: str = Query("all", pattern="^(all|completed|active)$"),
    q: str = "",
):
    sessions = _fetch_sessions(limit=limit, status=status, query=q.strip())
    return {"stats": _stats(sessions), "sessions": sessions}


@app.get("/api/admin/sessions/{session_id}")
async def api_session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    turns = get_session_turns(session_id)
    case = get_case(session.get("case_id"))
    case_summary = get_case_summary(case) if case else {}
    payload = _session_payload(session)
    trainer_detail = _extract_trainer_detail(session, turns)
    payload["trainer"] = trainer_detail.get("name") or UNKNOWN_TRAINER
    payload["trainer_detail"] = trainer_detail
    payload["case"] = case_summary
    payload["turns"] = [
        {
            "turn_index": turn.get("turn_index"),
            "created_at": turn.get("created_at"),
            "user_text": turn.get("user_text"),
            "assistant_text": turn.get("assistant_text"),
            "metadata": _metadata(turn),
        }
        for turn in turns
    ]
    return payload


ADMIN_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
<title>训练管理员后台</title>
<style>
  :root {
    --brand: #0f8f83;
    --brand-deep: #0a6158;
    --ink: #192330;
    --muted: #66778d;
    --line: rgba(15, 23, 42, 0.08);
    --page-a: #f4f8f4;
    --page-b: #ecf3f0;
    --shadow-lg: 0 18px 42px rgba(15, 23, 42, 0.08);
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    min-height: 100%;
    background:
      radial-gradient(circle at top left, rgba(15, 143, 131, 0.15), transparent 30%),
      radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 26%),
      linear-gradient(180deg, var(--page-a), var(--page-b));
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  .page {
    width: min(calc(100vw - 28px), 1180px);
    margin: 0 auto;
    padding: 18px 0 40px;
  }
  .hero {
    margin: 0 2px 14px;
    padding: 20px 20px 18px;
    border-radius: 24px;
    color: #f8fffd;
    background: linear-gradient(145deg, rgba(8, 29, 32, 0.92), rgba(12, 76, 70, 0.9));
    box-shadow: 0 22px 44px rgba(8, 29, 32, 0.2);
  }
  .hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 900;
    color: rgba(232, 246, 243, 0.9);
  }
  .hero-kicker::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #3dd3bf;
    box-shadow: 0 0 0 6px rgba(61, 211, 191, 0.14);
  }
  .hero-title {
    margin-top: 12px;
    font-size: 30px;
    line-height: 1.1;
    font-weight: 950;
  }
  .hero-sub {
    margin-top: 8px;
    font-size: 13px;
    color: rgba(235, 247, 244, 0.78);
  }
  .module-stack { display: grid; gap: 14px; }
  .stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  .module-bubble, .stat-card {
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,250,249,0.88));
    box-shadow: var(--shadow-lg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
  }
  .stat-card { padding: 16px; }
  .stat-label { color: var(--muted); font-size: 12px; font-weight: 800; }
  .stat-value { margin-top: 6px; color: var(--brand-deep); font-size: 28px; font-weight: 950; }
  .module-bubble { padding: 16px; }
  .module-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
  }
  .module-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(15, 143, 131, 0.08);
    border: 1px solid rgba(15, 143, 131, 0.12);
    font-size: 12px;
    font-weight: 850;
    color: var(--brand-deep);
  }
  .module-pill::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--brand);
    box-shadow: 0 0 0 5px rgba(15, 143, 131, 0.14);
  }
  .filters {
    display: grid;
    grid-template-columns: 1fr 150px 110px;
    gap: 10px;
  }
  input, select, button {
    min-height: 42px;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0 12px;
    background: rgba(255,255,255,0.78);
    color: var(--ink);
    font: inherit;
    outline: none;
  }
  button {
    border: 0;
    background: linear-gradient(135deg, #0f8f83, #0a6158);
    color: white;
    font-weight: 900;
    cursor: pointer;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(360px, .85fr);
    gap: 14px;
  }
  .session-list {
    display: grid;
    gap: 10px;
    max-height: 66vh;
    overflow: auto;
    padding-right: 2px;
  }
  .session-item {
    display: grid;
    grid-template-columns: 58px 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 12px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255,255,255,0.72);
    cursor: pointer;
  }
  .session-item.active {
    border-color: rgba(15,143,131,.48);
    box-shadow: inset 0 0 0 1px rgba(15,143,131,.12);
  }
  .score {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    background: rgba(15, 143, 131, 0.1);
    color: var(--brand-deep);
    font-size: 18px;
    font-weight: 950;
  }
  .score.empty { color: var(--muted); background: #f1f5f9; }
  .title-line {
    font-size: 14px;
    font-weight: 900;
    color: var(--ink);
    word-break: break-all;
  }
  .meta-line {
    margin-top: 5px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 0 9px;
    border-radius: 999px;
    background: rgba(15,143,131,.08);
    color: var(--brand-deep);
    font-size: 12px;
    font-weight: 850;
    white-space: nowrap;
  }
  .badge.gray { background: #f1f5f9; color: var(--muted); }
  .detail-empty {
    padding: 42px 12px;
    color: var(--muted);
    text-align: center;
    font-size: 13px;
  }
  .detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 12px;
  }
  .kv {
    padding: 10px 12px;
    border-radius: 14px;
    background: rgba(255,255,255,0.7);
    border: 1px solid var(--line);
  }
  .kv-label { color: var(--muted); font-size: 11px; font-weight: 850; }
  .kv-value { margin-top: 5px; font-size: 13px; font-weight: 850; word-break: break-all; }
  .muted { color: var(--muted); font-size: 12px; font-weight: 750; }
  .dim-list, .turn-list { display: grid; gap: 8px; }
  .dim-row {
    display: grid;
    grid-template-columns: 98px 1fr 42px;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  }
  .dim-name { font-weight: 850; text-align: right; color: var(--ink); }
  .bar { height: 8px; border-radius: 999px; overflow: hidden; background: #e2e8f0; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #0f8f83, #3b82f6); }
  .turn {
    padding: 10px 0;
    border-bottom: 1px solid rgba(15, 23, 42, 0.07);
  }
  .turn:last-child { border-bottom: 0; }
  .turn-title { color: var(--brand-deep); font-size: 12px; font-weight: 950; }
  .turn-text { margin-top: 5px; font-size: 13px; line-height: 1.55; }
  .turn-text span { color: var(--muted); font-weight: 900; }
  .section-title {
    margin: 16px 0 8px;
    color: var(--brand-deep);
    font-size: 13px;
    font-weight: 950;
  }
  @media (max-width: 860px) {
    .stats, .layout, .filters { grid-template-columns: 1fr; }
    .session-list { max-height: none; }
  }
</style>
</head>
<body>
<main class="page">
  <section class="hero">
    <div class="hero-kicker">管理员后台</div>
    <div class="hero-title">训练记录与评分总览</div>
    <div class="hero-sub">查看最近会话、评分结果、会话 ID、训练人和完整对话记录。</div>
  </section>

  <div class="module-stack">
    <section class="stats">
      <div class="stat-card"><div class="stat-label">列表会话</div><div class="stat-value" id="statListed">--</div></div>
      <div class="stat-card"><div class="stat-label">已完成</div><div class="stat-value" id="statCompleted">--</div></div>
      <div class="stat-card"><div class="stat-label">进行中</div><div class="stat-value" id="statActive">--</div></div>
      <div class="stat-card"><div class="stat-label">平均分</div><div class="stat-value" id="statAvg">--</div></div>
    </section>

    <section class="module-bubble">
      <div class="module-head">
        <span class="module-pill">筛选</span>
        <span class="badge gray" id="refreshTime">--</span>
      </div>
      <div class="filters">
        <input id="queryInput" placeholder="搜索 session_id / 阶段 / 难度 / case_id" />
        <select id="statusSelect">
          <option value="all">全部会话</option>
          <option value="completed">只看完成</option>
          <option value="active">只看进行中</option>
        </select>
        <button id="refreshBtn" type="button">刷新</button>
      </div>
    </section>

    <section class="layout">
      <div class="module-bubble">
        <div class="module-head">
          <span class="module-pill">最近会话</span>
          <span class="badge gray" id="listCount">0 条</span>
        </div>
        <div class="session-list" id="sessionList"></div>
      </div>
      <div class="module-bubble">
        <div class="module-head">
          <span class="module-pill">会话详情</span>
          <span class="badge gray" id="detailStatus">未选择</span>
        </div>
        <div id="detailPane" class="detail-empty">选择左侧一条会话查看评分和对话记录。</div>
      </div>
    </section>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
let selectedSessionId = "";

function escapeHtml(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function fmt(value) {
  if (!value) return "";
  return String(value).replace("T", " ").slice(0, 19);
}

function scoreClass(score) {
  return score == null ? "score empty" : "score";
}

function sessionStatus(item) {
  if (!item.is_complete) return '<span class="badge gray">进行中</span>';
  if (item.is_success) return '<span class="badge">成功</span>';
  return '<span class="badge gray">已完成</span>';
}

async function loadSessions() {
  const q = $("queryInput").value.trim();
  const status = $("statusSelect").value;
  const url = `/api/admin/sessions?limit=50&status=${encodeURIComponent(status)}&q=${encodeURIComponent(q)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("记录加载失败");
  const data = await res.json();
  renderStats(data.stats || {});
  renderSessions(data.sessions || []);
  $("refreshTime").textContent = new Date().toLocaleTimeString();
}

function renderStats(stats) {
  $("statListed").textContent = stats.listed_count ?? "--";
  $("statCompleted").textContent = stats.completed_count ?? "--";
  $("statActive").textContent = stats.active_count ?? "--";
  $("statAvg").textContent = stats.avg_score ?? "--";
}

function renderSessions(items) {
  $("listCount").textContent = `${items.length} 条`;
  if (!items.length) {
    $("sessionList").innerHTML = '<div class="detail-empty">暂无记录</div>';
    return;
  }
  $("sessionList").innerHTML = items.map((item) => `
    <div class="session-item ${item.session_id === selectedSessionId ? "active" : ""}" data-session-id="${escapeHtml(item.session_id)}">
      <div class="${scoreClass(item.score)}">${item.score ?? "--"}</div>
      <div>
        <div class="title-line">${escapeHtml(item.session_id)}</div>
        <div class="meta-line">
          训练人：${escapeHtml(item.trainer)} · ${escapeHtml(item.stage_id || "训练")} · ${escapeHtml(item.difficulty_id || "")}
        </div>
        <div class="meta-line">
          ${escapeHtml(item.customer_name || "客户")} · ${escapeHtml(item.business_line || "未记录业务线")} · ${escapeHtml(fmt(item.completed_at || item.updated_at))}
        </div>
      </div>
      <div>${sessionStatus(item)}</div>
    </div>
  `).join("");
  $("sessionList").querySelectorAll(".session-item").forEach((node) => {
    node.addEventListener("click", () => loadDetail(node.dataset.sessionId));
  });
}

async function loadDetail(sessionId) {
  selectedSessionId = sessionId;
  $("detailStatus").textContent = "加载中";
  const res = await fetch(`/api/admin/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error("详情加载失败");
  const data = await res.json();
  $("detailStatus").textContent = data.is_complete ? "已完成" : "进行中";
  renderDetail(data);
  loadSessions().catch(() => {});
}

function renderDimensions(evaluation) {
  const dims = evaluation?.dimension_scores || [];
  if (!dims.length) return '<div class="detail-empty">暂无评分维度</div>';
  return `<div class="dim-list">${dims.map((d) => {
    const max = Number(d.max_score || d.max || 0);
    const score = Number(d.score || 0);
    const pct = max ? Math.round(score * 100 / max) : Math.min(100, score);
    return `<div class="dim-row">
      <div class="dim-name">${escapeHtml(d.dimension || d.name || "")}</div>
      <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div>${pct}%</div>
    </div>`;
  }).join("")}</div>`;
}

function renderTurns(turns) {
  if (!turns?.length) return '<div class="detail-empty">暂无对话轮次</div>';
  return `<div class="turn-list">${turns.map((turn) => `
    <div class="turn">
      <div class="turn-title">第 ${escapeHtml(turn.turn_index || "?")} 轮 · ${escapeHtml(fmt(turn.created_at))}</div>
      <div class="turn-text"><span>训练人：</span>${escapeHtml(turn.user_text || "")}</div>
      <div class="turn-text"><span>客户：</span>${escapeHtml(turn.assistant_text || "")}</div>
    </div>
  `).join("")}</div>`;
}

function renderDetail(data) {
  const evaluation = data.evaluation || {};
  const trainer = data.trainer_detail || {};
  const trainerMeta = [trainer.department, trainer.source, trainer.external_user_id].filter(Boolean).join(" / ");
  $("detailPane").className = "";
  $("detailPane").innerHTML = `
    <div class="detail-grid">
      <div class="kv"><div class="kv-label">会话 ID</div><div class="kv-value">${escapeHtml(data.session_id)}</div></div>
      <div class="kv"><div class="kv-label">训练人</div><div class="kv-value">${escapeHtml(data.trainer)}${trainerMeta ? `<br><span class="muted">${escapeHtml(trainerMeta)}</span>` : ""}</div></div>
      <div class="kv"><div class="kv-label">评分</div><div class="kv-value">${data.score ?? "--"}</div></div>
      <div class="kv"><div class="kv-label">轮次</div><div class="kv-value">${escapeHtml(data.turn_count)}</div></div>
      <div class="kv"><div class="kv-label">阶段 / 难度</div><div class="kv-value">${escapeHtml(data.stage_id)} / ${escapeHtml(data.difficulty_id)}</div></div>
      <div class="kv"><div class="kv-label">完成时间</div><div class="kv-value">${escapeHtml(fmt(data.completed_at || data.updated_at))}</div></div>
      <div class="kv"><div class="kv-label">客户</div><div class="kv-value">${escapeHtml(data.customer_name || "--")}</div></div>
      <div class="kv"><div class="kv-label">业务线</div><div class="kv-value">${escapeHtml(data.business_line || "--")}</div></div>
    </div>
    <div class="section-title">评分记录</div>
    ${renderDimensions(evaluation)}
    <div class="section-title">对话记录</div>
    ${renderTurns(data.turns || [])}
  `;
}

$("refreshBtn").addEventListener("click", () => loadSessions().catch((err) => alert(err.message)));
$("statusSelect").addEventListener("change", () => loadSessions().catch((err) => alert(err.message)));
$("queryInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadSessions().catch((err) => alert(err.message));
});
loadSessions().catch((err) => {
  $("sessionList").innerHTML = `<div class="detail-empty">${escapeHtml(err.message)}</div>`;
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ADMIN_APP_HOST", "127.0.0.1")
    port = int(os.getenv("ADMIN_APP_PORT", "8521"))
    print(f"\n  http://{host}:{port}/admin  |  训练管理员后台\n")
    uvicorn.run(app, host=host, port=port, reload=False)
