"""
training_data_context.py - Connect docs JSONL assets to runtime prompts and UI data.

The three docs files serve different jobs:
- roleplay_cases.jsonl drives the simulated customer identity and state machine.
- raw_calls.jsonl provides real call summaries and dialogue examples.
- evaluation_rubrics.jsonl provides scoring dimensions and coach feedback anchors.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_DIR / "docs"
RAW_CALLS_PATH = DOCS_DIR / "raw_calls.jsonl"
RUBRICS_PATH = DOCS_DIR / "evaluation_rubrics.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


@lru_cache(maxsize=1)
def load_raw_calls() -> list[dict[str, Any]]:
    return _read_jsonl(RAW_CALLS_PATH)


@lru_cache(maxsize=1)
def load_rubrics() -> list[dict[str, Any]]:
    return _read_jsonl(RUBRICS_PATH)


def find_raw_call(
    source_call_id: str | None = None,
    call_id: str | None = None,
    training_type: str | None = None,
) -> dict[str, Any] | None:
    calls = load_raw_calls()
    if not calls:
        return None

    lookup_ids = {v for v in (source_call_id, call_id) if v}
    if lookup_ids:
        for call in calls:
            if call.get("call_id") in lookup_ids:
                return call

    if training_type:
        for call in calls:
            metadata = call.get("call_metadata", {})
            if metadata.get("call_type") == training_type:
                return call

    return None


def find_rubric(
    case_id: str | None = None,
    source_call_id: str | None = None,
    training_type: str | None = None,
) -> dict[str, Any] | None:
    rubrics = load_rubrics()
    if not rubrics:
        return None

    if case_id:
        for rubric in rubrics:
            if rubric.get("case_id") == case_id:
                return rubric

    if source_call_id:
        for rubric in rubrics:
            if rubric.get("source_call_id") == source_call_id:
                return rubric

    if training_type:
        for rubric in rubrics:
            if rubric.get("training_type") == training_type:
                return rubric

    return None


def find_assets_for_case(case: dict[str, Any] | None) -> dict[str, Any]:
    case = case or {}
    raw_call = find_raw_call(
        source_call_id=case.get("source_call_id"),
        training_type=case.get("training_type"),
    )
    rubric = find_rubric(
        case_id=case.get("case_id"),
        source_call_id=case.get("source_call_id"),
        training_type=case.get("training_type"),
    )
    return {"case": case, "raw_call": raw_call, "rubric": rubric}


def _clip(text: Any, limit: int = 140) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _raw_call_turn_samples(raw_call: dict[str, Any], max_turns: int = 8) -> list[str]:
    samples: list[str] = []
    for turn in raw_call.get("transcript_turns", [])[:max_turns]:
        speaker = "销售" if turn.get("speaker") == "sales" else "客户"
        text = _clip(turn.get("text"), 120)
        if text:
            samples.append(f"{speaker}：{text}")
    return samples


def build_raw_call_context(raw_call: dict[str, Any] | None) -> str:
    if not raw_call:
        return ""

    metadata = raw_call.get("call_metadata", {})
    summary = raw_call.get("summary", {})
    key_points = [p for p in summary.get("key_points", []) if p][:5]
    turns = _raw_call_turn_samples(raw_call)

    lines = [
        "## 真实通话参考（来自 raw_calls.jsonl）",
        f"场景：{_clip(metadata.get('scenario') or summary.get('one_sentence'), 220)}",
        f"结果：{_clip(summary.get('outcome'), 180)}",
    ]
    if key_points:
        lines.append("有效销售动作：")
        lines.extend(f"- {_clip(point, 120)}" for point in key_points)
    if turns:
        lines.append("对话节奏参考：")
        lines.extend(f"- {turn}" for turn in turns)
    return "\n".join(line for line in lines if line)


def build_rubric_context(rubric: dict[str, Any] | None) -> str:
    if not rubric:
        return ""

    dimensions = rubric.get("scoring_dimensions", [])[:8]
    must_do = rubric.get("must_do", [])[:6]
    critical = rubric.get("critical_mistakes", [])[:6]
    ideal_flow = rubric.get("ideal_sales_flow", [])[:5]

    lines = [
        "## 本轮评分标准（来自 evaluation_rubrics.jsonl，仅用于内部判断）",
        "评分维度：",
    ]
    for dim in dimensions:
        lines.append(
            f"- {dim.get('dimension', '')}（{dim.get('score', 0)}分）："
            f"优秀={_clip(dim.get('excellent'), 90)}；"
            f"不合格={_clip(dim.get('fail'), 90)}"
        )
    if must_do:
        lines.append("必须覆盖：")
        lines.extend(f"- {_clip(item, 80)}" for item in must_do)
    if critical:
        lines.append("扣分红线：")
        lines.extend(f"- {_clip(item, 90)}" for item in critical)
    if ideal_flow:
        lines.append("理想推进路径：")
        lines.extend(f"- {_clip(item, 100)}" for item in ideal_flow)
    return "\n".join(line for line in lines if line)


def build_model_context_for_case(case: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    assets = find_assets_for_case(case)
    raw_context = build_raw_call_context(assets.get("raw_call"))
    rubric_context = build_rubric_context(assets.get("rubric"))
    context = "\n\n".join(part for part in (raw_context, rubric_context) if part)

    raw_call = assets.get("raw_call") or {}
    rubric = assets.get("rubric") or {}
    metadata = {
        "raw_call_id": raw_call.get("call_id"),
        "rubric_id": rubric.get("rubric_id"),
        "rubric_dimension_count": len(rubric.get("scoring_dimensions", [])) if rubric else 0,
        "raw_call_turn_count": len(raw_call.get("transcript_turns", [])) if raw_call else 0,
    }
    return context, metadata


def summarize_case_assets(case: dict[str, Any] | None) -> dict[str, Any]:
    assets = find_assets_for_case(case)
    case = assets.get("case") or {}
    raw_call = assets.get("raw_call") or {}
    rubric = assets.get("rubric") or {}
    role = case.get("customer_role_card", {})
    hidden = case.get("hidden_customer_state", {})

    return {
        "case_id": case.get("case_id"),
        "source_call_id": case.get("source_call_id"),
        "scene": case.get("scene"),
        "business_line": case.get("business_line"),
        "customer_name": role.get("name"),
        "customer_role": role.get("role"),
        "customer_location": role.get("company_location"),
        "current_status": role.get("current_status"),
        "personality": role.get("personality"),
        "communication_style": role.get("communication_style"),
        "main_concerns": hidden.get("main_concerns", []),
        "price_sensitivity": hidden.get("price_sensitivity"),
        "trust_start": hidden.get("trust_level_at_start"),
        "few_shot_count": len(case.get("few_shot_examples", [])),
        "raw_call_id": raw_call.get("call_id"),
        "raw_call_summary": (raw_call.get("summary") or {}).get("one_sentence"),
        "rubric_id": rubric.get("rubric_id"),
        "rubric_dimensions": [
            {"dimension": d.get("dimension"), "score": d.get("score")}
            for d in rubric.get("scoring_dimensions", [])[:8]
        ],
        "must_do": rubric.get("must_do", [])[:6] if rubric else [],
        "critical_mistakes": rubric.get("critical_mistakes", [])[:6] if rubric else [],
    }


def recent_training_records(limit: int = 3) -> list[dict[str, Any]]:
    try:
        from .conversation_store import recent_completed_sessions
    except ImportError:
        try:
            from conversation_store import recent_completed_sessions
        except ImportError:
            recent_completed_sessions = None

    if recent_completed_sessions:
        completed = recent_completed_sessions(limit=limit)
        records = []
        for session in completed:
            evaluation = session.get("evaluation") or {}
            records.append(
                {
                    "time": "最近",
                    "title": f"{session.get('stage_id') or '训练'} · {session.get('final_state') or '已结束'}",
                    "desc": _clip(evaluation.get("summary") or session.get("memory_text"), 38),
                    "score": str(evaluation.get("total_score") or ""),
                }
            )
        if records:
            return records

    records: list[dict[str, Any]] = []
    for raw_call in load_raw_calls()[: max(limit, 0)]:
        metadata = raw_call.get("call_metadata", {})
        summary = raw_call.get("summary", {})
        records.append(
            {
                "time": "历史",
                "title": f"{metadata.get('call_type', '训练')} · {metadata.get('sales_stage', '')}",
                "desc": _clip(summary.get("outcome") or summary.get("one_sentence"), 38),
                "score": "",
            }
        )
    return records
