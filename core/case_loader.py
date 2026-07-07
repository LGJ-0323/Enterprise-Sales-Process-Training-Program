"""
case_loader.py — 从 roleplay_cases.jsonl 加载和匹配案例

用于在训练开始时根据 training_type + difficulty 匹配角色扮演案例。
如果 JSONL 中没有匹配的案例，返回 None（调用方 fallback 到 YAML 配置）。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_DIR / "docs"
DEFAULT_CASES_PATH = DOCS_DIR / "roleplay_cases.jsonl"

# ── Schema 校验 ──────────────────────────────────────────────

# v2.0 必填顶层字段
_REQUIRED_TOP_FIELDS = [
    "case_id", "source_call_id", "schema_version", "training_type",
    "difficulty", "business_line", "scene",
    "customer_role_card", "hidden_customer_state", "state_machine", "few_shot_examples",
]

# customer_role_card 必填子字段
_REQUIRED_ROLE_FIELDS = [
    "name", "role", "company_location", "business_context",
    "personality", "communication_style", "decision_style", "current_status",
]

# hidden_customer_state 必填子字段
_REQUIRED_HIDDEN_FIELDS = [
    "main_concerns", "price_sensitivity", "trust_level_at_start",
]

_VALID_SCHEMA_VERSIONS = frozenset({"2.0"})


def _fmt_loc(filepath: Path, line_no: int, case_id: str | None = None) -> str:
    """格式化错误位置信息。"""
    base = f"{filepath}:{line_no}"
    return f"{base} (case_id={case_id})" if case_id else base


def validate_case(case: dict[str, Any], filepath: Path, line_no: int) -> None:
    """按 SPEC.md §2.1 校验单条 case 的结构完整性。

    校验项：
    - 必填顶层字段
    - schema_version 合法性
    - customer_role_card / hidden_customer_state 子字段
    - state_machine 结构：states 非空、initial_state 存在、transitions 闭合
    - few_shot_examples 基本结构

    校验失败直接抛 ValueError，附带文件位置和 case_id。
    """
    case_id = str(case.get("case_id") or "?")

    # ── 顶层必填字段 ──
    missing_top = [f for f in _REQUIRED_TOP_FIELDS if f not in case]
    if missing_top:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"缺少必填字段: {', '.join(missing_top)}"
        )

    # ── schema_version ──
    sv = str(case.get("schema_version", ""))
    if sv not in _VALID_SCHEMA_VERSIONS:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"schema_version 必须为 2.0，当前为 {sv!r}"
        )

    # ── customer_role_card 子字段 ──
    role = case.get("customer_role_card", {})
    if not isinstance(role, dict):
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"customer_role_card 必须是对象，当前类型为 {type(role).__name__}"
        )
    missing_role = [f for f in _REQUIRED_ROLE_FIELDS if f not in role]
    if missing_role:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"customer_role_card 缺少必填字段: {', '.join(missing_role)}"
        )

    # ── hidden_customer_state 子字段 ──
    hidden = case.get("hidden_customer_state", {})
    if not isinstance(hidden, dict):
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"hidden_customer_state 必须是对象，当前类型为 {type(hidden).__name__}"
        )
    missing_hidden = [f for f in _REQUIRED_HIDDEN_FIELDS if f not in hidden]
    if missing_hidden:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"hidden_customer_state 缺少必填字段: {', '.join(missing_hidden)}"
        )

    # ── state_machine ──
    sm = case.get("state_machine", {})
    if not isinstance(sm, dict):
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"state_machine 必须是对象，当前类型为 {type(sm).__name__}"
        )

    states = sm.get("states", {})
    if not isinstance(states, dict) or len(states) == 0:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"state_machine.states 必须是非空对象"
        )

    initial = sm.get("initial_state")
    if initial not in states:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"state_machine.initial_state={initial!r} 不在 states 中（有效值: {sorted(states.keys())}）"
        )

    transitions = sm.get("transitions", [])
    if not isinstance(transitions, list):
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"state_machine.transitions 必须是数组"
        )

    for ti, t in enumerate(transitions):
        if not isinstance(t, dict):
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"state_machine.transitions[{ti}] 不是对象"
            )
        fr = t.get("from")
        to = t.get("to")
        if fr is None or to is None:
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"state_machine.transitions[{ti}] 缺少 from/to 字段"
            )
        if fr not in states:
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"state_machine.transitions[{ti}].from={fr!r} 不在 states 中（有效值: {sorted(states.keys())}）"
            )
        if to not in states:
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"state_machine.transitions[{ti}].to={to!r} 不在 states 中（有效值: {sorted(states.keys())}）"
            )

    # ── few_shot_examples ──
    shots = case.get("few_shot_examples", [])
    if not isinstance(shots, list) or len(shots) == 0:
        raise ValueError(
            f"{_fmt_loc(filepath, line_no, case_id)}: "
            f"few_shot_examples 必须是非空数组"
        )

    for si, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"few_shot_examples[{si}] 不是对象"
            )
        shot_state = shot.get("state")
        if not shot_state:
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"few_shot_examples[{si}] 缺少 state 字段"
            )
        if shot_state not in states:
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"few_shot_examples[{si}].state={shot_state!r} 不在 states 中（有效值: {sorted(states.keys())}）"
            )
        if "sales_input" not in shot or "customer_reply" not in shot:
            raise ValueError(
                f"{_fmt_loc(filepath, line_no, case_id)}: "
                f"few_shot_examples[{si}] 缺少 sales_input 或 customer_reply 字段"
            )


# ── 加载 ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_all_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载全部 roleplay cases，逐行校验（缓存结果）。

    校验失败时抛 ValueError，不再静默跳过坏数据。
    """
    filepath = Path(path) if path else DEFAULT_CASES_PATH
    if not filepath.exists():
        return []

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with open(filepath, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{_fmt_loc(filepath, line_no)}: JSON 解析失败 — {e}"
                ) from e

            if not isinstance(case, dict):
                raise ValueError(
                    f"{_fmt_loc(filepath, line_no)}: "
                    f"每行必须是 JSON 对象，当前类型为 {type(case).__name__}"
                )

            # 逐行校验结构
            validate_case(case, filepath, line_no)

            # case_id 唯一性
            cid = str(case["case_id"])
            if cid in seen_ids:
                raise ValueError(
                    f"{_fmt_loc(filepath, line_no, cid)}: case_id 重复，此前已出现"
                )
            seen_ids.add(cid)

            cases.append(case)

    return cases


def reload_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """强制重新加载（绕过 lru_cache），复用同一套校验。"""
    _load_all_cases.cache_clear()
    return _load_all_cases(path)


def _type_matches(case_type: str, query_type: str) -> bool:
    """模糊匹配 training_type，支持：
    - 精确匹配："回访" == "回访"
    - 子串匹配："报价后回访" 包含 "回访"
    - 业务别名："陌call" 可匹配 "新客户开发"
    """
    aliases = {
        "陌call": "新客户开发",
        "陌 Call": "新客户开发",
        "陌拜": "新客户开发",
        "首次触达": "新客户开发",
        "cold_call": "新客户开发",
        "回访": "报价后回访",
        "follow_up": "报价后回访",
        "深入回访": "报价后回访",
        "deep_follow_up": "报价后回访",
        "逼单": "报价后回访",
        "closing": "报价后回访",
    }
    case_candidates = {case_type, aliases.get(case_type, "")}
    query_candidates = {query_type, aliases.get(query_type, "")}
    case_candidates = {item for item in case_candidates if item}
    query_candidates = {item for item in query_candidates if item}
    for left in case_candidates:
        for right in query_candidates:
            if left == right or left in right or right in left:
                return True
    return False


def _diff_matches(case_diff: str, query_diff: str) -> bool:
    """模糊匹配 difficulty，支持中英文："中级" == "medium", "medium" == "中级" """
    if case_diff == query_diff:
        return True
    # 中英文映射
    mapping = {
        "初级": "easy", "easy": "初级", "简单": "easy",
        "中级": "medium", "medium": "中级", "普通": "medium", "normal": "medium",
        "高级": "hard", "hard": "高级", "困难": "hard", "专家": "hard", "expert": "hard",
        "normal": "medium", "expert": "hard",
    }
    return mapping.get(case_diff) == query_diff or mapping.get(query_diff) == case_diff


def find_case(
    training_type: str,
    difficulty: str,
    business_line: str | None = None,
) -> dict[str, Any] | None:
    """根据 training_type + difficulty 匹配案例。

    匹配优先级：
    1. training_type + difficulty + business_line 三者精确/模糊匹配
    2. training_type + difficulty 匹配（忽略 business_line）
    3. training_type 匹配（忽略 difficulty 和 business_line）

    training_type 支持子串匹配（"回访" 可匹配 "报价后回访"）。
    difficulty 支持中英文互转（"中级" ⇔ "medium"）。

    Returns:
        匹配到的 case dict，如果没有任何匹配返回 None。
    """
    matches = find_cases(training_type, difficulty, business_line)
    return matches[0] if matches else None


def find_cases(
    training_type: str,
    difficulty: str,
    business_line: str | None = None,
) -> list[dict[str, Any]]:
    """返回所有匹配 training_type + difficulty 的候选案例。"""
    cases = _load_all_cases()
    if not cases:
        return []

    matches: list[dict[str, Any]] = []
    # 优先级 1: 三者匹配
    if business_line:
        for case in cases:
            if (
                _type_matches(case.get("training_type", ""), training_type)
                and _diff_matches(case.get("difficulty", ""), difficulty)
                and case.get("business_line") == business_line
            ):
                matches.append(case)
        if matches:
            return matches

    # 优先级 2: training_type + difficulty
    for case in cases:
        if (
            _type_matches(case.get("training_type", ""), training_type)
            and _diff_matches(case.get("difficulty", ""), difficulty)
        ):
            matches.append(case)
    if matches:
        return matches

    # 优先级 3: 只匹配 training_type
    for case in cases:
        if _type_matches(case.get("training_type", ""), training_type):
            matches.append(case)

    return matches


def get_case(case_id: str | None) -> dict[str, Any] | None:
    """按 case_id 返回案例。"""
    if not case_id:
        return None
    for case in _load_all_cases():
        if case.get("case_id") == case_id:
            return case
    return None


def get_case_summary(case: dict[str, Any]) -> dict[str, Any]:
    """提取案例的摘要信息，用于 Gradio 界面展示。"""
    role = case.get("customer_role_card", {})
    hidden = case.get("hidden_customer_state", {})
    sm = case.get("state_machine", {})

    return {
        "case_id": case.get("case_id"),
        "training_type": case.get("training_type"),
        "difficulty": case.get("difficulty"),
        "business_line": case.get("business_line"),
        "customer_name": role.get("name"),
        "customer_role": role.get("role"),
        "customer_location": role.get("company_location"),
        "trust_start": hidden.get("trust_level_at_start"),
        "price_sensitivity": hidden.get("price_sensitivity"),
        "initial_state": sm.get("initial_state"),
        "state_count": len(sm.get("states", {})),
        "transition_count": len(sm.get("transitions", [])),
        "few_shot_count": len(case.get("few_shot_examples", [])),
        "objection_count": len(case.get("likely_objections", [])),
        "has_failure_conditions": bool(case.get("failure_conditions")),
        "has_difficulty_variants": bool(case.get("difficulty_variants")),
        "training_goal_count": len(case.get("training_goals", [])),
    }


def get_available_training_types() -> list[str]:
    """返回所有可用的 training_type 列表。"""
    cases = _load_all_cases()
    return sorted(set(c.get("training_type", "") for c in cases if c.get("training_type")))


def case_count() -> int:
    """返回已加载的案例总数。"""
    return len(_load_all_cases())
