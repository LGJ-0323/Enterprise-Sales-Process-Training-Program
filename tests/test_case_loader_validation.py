"""test_case_loader_validation.py — validate_case() 最小测试

运行: cd D:\workspace\personal_project && python -m pytest tests/test_case_loader_validation.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 确保 core/ 在 sys.path 中
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "core"))

from case_loader import validate_case  # noqa: E402

# ── helpers ──────────────────────────────────────────────────

def _valid_case(**overrides) -> dict:
    """构造一条完全合法的 v2.0 case 作为基准。"""
    case = {
        "case_id": "case_test_001",
        "source_call_id": "call_test_001",
        "schema_version": "2.0",
        "training_type": "报价后回访",
        "difficulty": "medium",
        "business_line": "国际物流/美线",
        "scene": "测试场景描述",
        "customer_role_card": {
            "name": "测试客户",
            "role": "物流经理",
            "company_location": "深圳",
            "business_context": "测试背景",
            "personality": "理性",
            "communication_style": "直接",
            "decision_style": "比价",
            "current_status": "已报价-等待确认",
        },
        "hidden_customer_state": {
            "main_concerns": ["价格", "时效"],
            "price_sensitivity": "medium",
            "trust_level_at_start": 50,
        },
        "state_machine": {
            "initial_state": "guarded",
            "states": {
                "guarded": {"description": "防备", "tone": "neutral", "reply_length": "1句"},
                "warming_up": {"description": "放松", "tone": "cooperative", "reply_length": "2句"},
                "shut_down": {
                    "description": "结束", "tone": "dismissive", "reply_length": "1句",
                    "is_terminal": True, "is_failure": True,
                },
            },
            "transitions": [
                {"from": "guarded", "to": "warming_up", "trigger": "销售展现专业度"},
                {"from": "warming_up", "to": "shut_down", "trigger": "销售冒犯客户"},
            ],
        },
        "few_shot_examples": [
            {"state": "guarded", "sales_input": "您好", "customer_reply": "什么事？", "why": "防备"},
            {"state": "warming_up", "sales_input": "最近怎么样？", "customer_reply": "还行", "why": "放松"},
        ],
    }
    case.update(overrides)
    return case


# ── 通过测试 ──────────────────────────────────────────────────

def test_valid_case_passes():
    """合法 case 应该静默通过。"""
    validate_case(_valid_case(), Path("test.jsonl"), 1)  # 不抛异常即通过


# ── 顶层字段校验 ──────────────────────────────────────────────

def test_missing_top_field_raises():
    """缺少必填顶层字段时抛 ValueError。"""
    case = _valid_case()
    del case["scene"]
    with pytest.raises(ValueError, match="缺少必填字段.*scene"):
        validate_case(case, Path("test.jsonl"), 1)


def test_bad_schema_version_raises():
    """非 2.0 的 schema_version 抛错。"""
    case = _valid_case(schema_version="1.0")
    with pytest.raises(ValueError, match="schema_version 必须为 2.0"):
        validate_case(case, Path("test.jsonl"), 1)


# ── customer_role_card 子字段 ─────────────────────────────────

def test_missing_role_field_raises():
    """customer_role_card 缺必填子字段抛错。"""
    case = _valid_case()
    del case["customer_role_card"]["current_status"]
    with pytest.raises(ValueError, match="customer_role_card 缺少必填字段.*current_status"):
        validate_case(case, Path("test.jsonl"), 1)


# ── hidden_customer_state ─────────────────────────────────────

def test_missing_hidden_field_raises():
    """hidden_customer_state 缺必填子字段抛错。"""
    case = _valid_case()
    del case["hidden_customer_state"]["trust_level_at_start"]
    with pytest.raises(ValueError, match="hidden_customer_state 缺少必填字段.*trust_level_at_start"):
        validate_case(case, Path("test.jsonl"), 1)


# ── state_machine ─────────────────────────────────────────────

def test_empty_states_raises():
    """states 为空对象抛错。"""
    case = _valid_case()
    case["state_machine"]["states"] = {}
    with pytest.raises(ValueError, match="states 必须是非空对象"):
        validate_case(case, Path("test.jsonl"), 1)


def test_initial_state_not_in_states_raises():
    """initial_state 不在 states 中抛错。"""
    case = _valid_case()
    case["state_machine"]["initial_state"] = "ghost_state"
    with pytest.raises(ValueError, match="initial_state='ghost_state' 不在 states"):
        validate_case(case, Path("test.jsonl"), 1)


def test_transition_from_not_in_states_raises():
    """transition.from 引用不存在状态抛错。"""
    case = _valid_case()
    case["state_machine"]["transitions"].append(
        {"from": "ghost", "to": "guarded", "trigger": "不可能"}
    )
    with pytest.raises(ValueError, match="transitions.*from='ghost' 不在 states"):
        validate_case(case, Path("test.jsonl"), 1)


def test_transition_to_not_in_states_raises():
    """transition.to 引用不存在状态抛错。"""
    case = _valid_case()
    case["state_machine"]["transitions"].append(
        {"from": "guarded", "to": "ghost", "trigger": "不可能"}
    )
    with pytest.raises(ValueError, match="transitions.*to='ghost' 不在 states"):
        validate_case(case, Path("test.jsonl"), 1)


# ── few_shot_examples ─────────────────────────────────────────

def test_empty_few_shot_raises():
    """few_shot_examples 为空数组抛错。"""
    case = _valid_case(few_shot_examples=[])
    with pytest.raises(ValueError, match="few_shot_examples 必须是非空数组"):
        validate_case(case, Path("test.jsonl"), 1)


def test_few_shot_state_not_in_states_raises():
    """few_shot 引用了不存在的 state 抛错。"""
    case = _valid_case()
    case["few_shot_examples"][0]["state"] = "ghost_state"
    with pytest.raises(ValueError, match=r"few_shot_examples\[0\].state='ghost_state' 不在 states"):
        validate_case(case, Path("test.jsonl"), 1)


def test_few_shot_missing_fields_raises():
    """few_shot 缺 sales_input 或 customer_reply 抛错。"""
    case = _valid_case()
    del case["few_shot_examples"][0]["sales_input"]
    with pytest.raises(ValueError, match="缺少 sales_input 或 customer_reply"):
        validate_case(case, Path("test.jsonl"), 1)


# ── 错误信息包含行号和 case_id ─────────────────────────────────

def test_error_message_includes_location():
    """错误信息应包含文件路径、行号和 case_id。"""
    case = _valid_case()
    del case["scene"]
    with pytest.raises(ValueError, match=r"test\.jsonl:42.*case_id=case_test_001"):
        validate_case(case, Path("test.jsonl"), 42)
