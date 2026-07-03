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


@lru_cache(maxsize=1)
def _load_all_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载全部 roleplay cases（缓存结果）。"""
    filepath = Path(path) if path else DEFAULT_CASES_PATH
    if not filepath.exists():
        return []

    cases: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                cases.append(case)
            except json.JSONDecodeError:
                continue
    return cases


def reload_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """强制重新加载（绕过 lru_cache）。"""
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
