"""
training_evaluator.py — 训练评分引擎

提供两种评分模式：
1. LLM 评分（_llm_evaluation）：调用千问模型，基于评分维度（rubric）对对话进行智能打分
2. 启发式评分（_heuristic_evaluation）：基于关键词匹配的规则打分，无需调用 LLM

评分维度来自 evaluation_rubrics.jsonl，包含：
- scoring_dimensions: 各评分维度及分值
- must_do: 必须覆盖的销售动作
- critical_mistakes: 扣分红线
- ideal_sales_flow: 理想销售推进路径

评分结果包含：
- total_score: 综合得分（0-100）
- dimension_scores: 各维度得分明细
- strengths: 本轮亮点
- improvements: 改进建议
- summary: 一句话总结
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

try:
    from .case_loader import get_case
    from .conversation_store import get_session_turns
    from .training_data_context import find_rubric
except ImportError:
    from case_loader import get_case
    from conversation_store import get_session_turns
    from training_data_context import find_rubric


def _clip(text: Any, limit: int = 180) -> str:
    """文本截断工具，超过 limit 字符时截断并加省略号。"""
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "..."


def _conversation_text(turns: list[dict[str, Any]], limit: int = 24) -> str:
    """将对话轮次格式化为「销售：xxx / 客户：xxx」的纯文本，用于评分 prompt。"""
    lines: list[str] = []
    for turn in turns[-limit:]:
        user = str(turn.get("user_text") or "").strip()
        assistant = str(turn.get("assistant_text") or "").strip()
        if user:
            lines.append(f"销售：{user}")
        if assistant:
            lines.append(f"客户：{assistant}")
    return "\n".join(lines)


def _all_sales_text(turns: list[dict[str, Any]]) -> str:
    """提取所有销售（员工）发言的纯文本拼接。"""
    return "\n".join(str(t.get("user_text") or "") for t in turns)


def _all_customer_text(turns: list[dict[str, Any]]) -> str:
    """提取所有客户发言的纯文本拼接。"""
    return "\n".join(str(t.get("assistant_text") or "") for t in turns)


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    """检查文本中是否包含任意一个关键词。"""
    return any(word in text for word in words)


def _heuristic_ratio(dimension: str, sales_text: str, customer_text: str, full_text: str) -> tuple[float, str]:
    """基于关键词匹配的启发式评分。

    根据评分维度名称中的关键词，判断销售发言中是否包含对应的有效动作，
    返回 (得分比例 0-1, 评分依据)。

    覆盖的维度类型：
    - 开场/破冰：检查是否有自我介绍和来意说明
    - 需求追问：检查是否追问客户现状和业务信息
    - 异议处理：检查是否识别并回应客户顾虑
    - 专业价值：检查是否体现物流专业知识
    - 推进闭环：检查是否有明确的下一步动作
    - 信息补全：检查是否补充了客户或货物信息
    """
    dim = dimension
    evidence: list[str] = []
    ratio = 0.58

    if any(key in dim for key in ("开场", "来意", "破冰", "沟通")):
        if _has_any(sales_text[:180], ("您好", "你好", "我是", "今天", "联系", "回访")):
            ratio = 0.86
            evidence.append("开场说明了身份或来意")
        else:
            evidence.append("开场身份和来意不够清晰")

    elif any(key in dim for key in ("需求", "进展", "反馈", "问题理解", "服务反馈")):
        if _has_any(sales_text, ("需求", "出货", "货量", "报价", "进展", "反馈", "满意", "问题", "产品", "目的地", "时效")):
            ratio = 0.82
            evidence.append("有追问客户现状或关键业务信息")
        else:
            evidence.append("需求追问和客户信息补全偏少")

    elif any(key in dim for key in ("价格", "异议", "情绪", "解决方案", "异常")):
        concern = _has_any(customer_text, ("贵", "价格", "比",
                           "固定", "担心", "问题", "延误", "清关", "排舱", "涨价"))
        response = _has_any(sales_text, ("理解", "费用", "方案",
                            "对比", "清关", "派送", "时效", "锁舱", "处理", "跟进"))
        if concern and response:
            ratio = 0.8
            evidence.append("识别并回应了客户顾虑")
        elif response:
            ratio = 0.7
            evidence.append("有解决方案表达，但和客户异议匹配度一般")
        else:
            evidence.append("异议或问题处理证据不足")

    elif any(key in dim for key in ("价值", "专业", "增值", "形象")):
        if _has_any(sales_text, ("美线", "FBA", "清关", "派送", "舱位", "旺季", "报关", "费用", "渠道", "时效", "船公司")):
            ratio = 0.83
            evidence.append("体现了物流专业信息或差异化价值")
        else:
            evidence.append("专业价值呈现不够具体")

    elif any(key in dim for key in ("推进", "下一步", "结束", "闭环", "转介绍", "业务拓展")):
        if _has_any(full_text, ("加微信", "发您", "发我", "资料", "报价", "明天", "下午", "试单", "托书", "锁定", "下一步", "再联系")):
            ratio = 0.84
            evidence.append("有明确后续动作或时间节点")
        else:
            evidence.append("收尾缺少明确下一步")

    elif any(key in dim for key in ("信息", "补全")):
        if _has_any(full_text, ("地址", "哪里", "东莞", "深圳", "广州", "上海", "公斤", "票", "B2C", "B2B", "FBA")):
            ratio = 0.78
            evidence.append("补充了部分客户或货物信息")
        else:
            evidence.append("客户信息补全较少")

    else:
        if len(turns := full_text.splitlines()) >= 6:
            ratio = 0.72
            evidence.append("对话有基本往返，按通用维度给出保守评分")
        else:
            evidence.append("对话轮次不足，按通用维度保守评分")

    return ratio, "；".join(evidence)


def _heuristic_evaluation(
    session_id: str,
    case: dict[str, Any] | None,
    rubric: dict[str, Any] | None,
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """启发式评分：基于关键词规则对每个评分维度进行打分。

    无需调用 LLM，速度快，适合快速反馈场景。
    评分精度低于 LLM 评分，但可作为 LLM 评分的 fallback。
    """
    dimensions = (rubric or {}).get("scoring_dimensions") or []
    sales_text = _all_sales_text(turns)
    customer_text = _all_customer_text(turns)
    full_text = _conversation_text(turns)

    dimension_scores: list[dict[str, Any]] = []
    for dim in dimensions:
        max_score = int(dim.get("score") or 0)
        ratio, reason = _heuristic_ratio(
            str(dim.get("dimension") or ""), sales_text, customer_text, full_text)
        score = max(0, min(max_score, round(max_score * ratio)))
        dimension_scores.append(
            {
                "dimension": dim.get("dimension") or "评分维度",
                "score": score,
                "max_score": max_score,
                "reason": reason,
            }
        )

    total_score = sum(item["score"] for item in dimension_scores)
    max_total = sum(item["max_score"] for item in dimension_scores) or int(
        (rubric or {}).get("total_score") or 100)
    normalized_total = round(total_score * 100 / max_total) if max_total else 0

    # 生成固定格式的改进建议和亮点总结
    improvements = [
        "下一轮优先把客户当前进展、货量、目的地和核心顾虑问具体。",
        "遇到价格或时效异议时，先复述客户担心，再用费用结构或案例对比回应。",
        "收尾时给出一个低门槛下一步，并带具体时间点。",
    ]
    strengths = []
    if _has_any(sales_text[:220], ("您好", "你好", "我是", "回访", "联系")):
        strengths.append("开场能说明身份或联系原因。")
    if _has_any(full_text, ("发您", "微信", "试单", "报价", "资料")):
        strengths.append("对话中出现了下一步推进信号。")
    if not strengths:
        strengths.append("完成了基本对话往返，可作为复盘素材。")

    return {
        "session_id": session_id,
        "case_id": (case or {}).get("case_id"),
        "rubric_id": (rubric or {}).get("rubric_id"),
        "source": "heuristic",
        "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_score": normalized_total,
        "dimension_scores": dimension_scores,
        "strengths": strengths[:3],
        "improvements": improvements[:3],
        "summary": f"本轮综合得分 {normalized_total}/100。评分已按当前 case 的 rubric 维度折算。",
    }


def _extract_json_payload(text: str) -> dict[str, Any]:
    """从 LLM 回复文本中提取 JSON 对象（支持纯 JSON 和 Markdown 包裹的 JSON）。"""
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    # 尝试从文本中提取第一个 {...} 块
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_evaluation(
    payload: dict[str, Any],
    session_id: str,
    case: dict[str, Any] | None,
    rubric: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """将 LLM 返回的评分 JSON 标准化为统一格式。

    校验维度名称和分值是否与 rubric 匹配，对分数做边界约束。
    """
    dimensions = (rubric or {}).get("scoring_dimensions") or []
    if not dimensions:
        return None

    by_name = {str(d.get("dimension") or ""): int(
        d.get("score") or 0) for d in dimensions}
    raw_scores = payload.get("dimension_scores")
    if not isinstance(raw_scores, list):
        return None

    normalized: list[dict[str, Any]] = []
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        name = str(item.get("dimension") or "").strip()
        max_score = int(item.get("max_score") or by_name.get(name) or 0)
        if not name or max_score <= 0:
            continue
        try:
            score = round(float(item.get("score") or 0))
        except (TypeError, ValueError):
            score = 0
        normalized.append(
            {
                "dimension": name,
                "score": max(0, min(max_score, score)),
                "max_score": max_score,
                "reason": _clip(item.get("reason"), 160),
            }
        )

    if not normalized:
        return None

    total = sum(item["score"] for item in normalized)
    max_total = sum(item["max_score"] for item in normalized) or 100
    normalized_total = round(total * 100 / max_total)

    strengths = payload.get("strengths") if isinstance(
        payload.get("strengths"), list) else []
    improvements = payload.get("improvements") if isinstance(
        payload.get("improvements"), list) else []

    return {
        "session_id": session_id,
        "case_id": (case or {}).get("case_id"),
        "rubric_id": (rubric or {}).get("rubric_id"),
        "source": "llm",
        "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_score": int(payload.get("total_score") or normalized_total),
        "dimension_scores": normalized,
        "strengths": [_clip(item, 120) for item in strengths[:3]] or ["对话已完成，可进入复盘。"],
        "improvements": [_clip(item, 140) for item in improvements[:3]] or ["下一轮补充更明确的客户需求和下一步动作。"],
        "summary": _clip(payload.get("summary") or f"本轮综合得分 {normalized_total}/100。", 220),
    }


def _llm_evaluation(
    session_id: str,
    case: dict[str, Any] | None,
    rubric: dict[str, Any] | None,
    turns: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """LLM 智能评分：调用千问模型基于 rubric 维度对对话进行打分。

    优先使用此方法，评分更精准。失败时 fallback 到启发式评分。
    """
    if not rubric or not os.getenv("DASHSCOPE_API_KEY"):
        return None
    try:
        from dashscope import Generation
    except ImportError:
        return None

    dimensions = [
        {
            "dimension": dim.get("dimension"),
            "max_score": dim.get("score"),
            "excellent": dim.get("excellent"),
            "pass": dim.get("pass"),
            "fail": dim.get("fail"),
        }
        for dim in rubric.get("scoring_dimensions", [])
    ]
    prompt = (
        "你是国际物流销售训练评分教练。请严格基于评分维度和真实对话打分，不要凭空编造。\n"
        "输出必须是 JSON，不要 Markdown，不要额外解释。\n\n"
        f"场景：{(case or {}).get('scene', '')}\n\n"
        f"评分维度：{json.dumps(dimensions, ensure_ascii=False)}\n\n"
        f"关键红线：{json.dumps((rubric or {}).get('critical_mistakes', []), ensure_ascii=False)}\n\n"
        f"对话：\n{_conversation_text(turns)}\n\n"
        "JSON 格式："
        "{\"total_score\": 0, \"dimension_scores\": "
        "[{\"dimension\": \"\", \"score\": 0, \"max_score\": 0, \"reason\": \"\"}], "
        "\"strengths\": [\"\"], \"improvements\": [\"\"], \"summary\": \"\"}"
    )
    try:
        resp = Generation.call(
            model=os.getenv("DASHSCOPE_LLM_MODEL", "qwen3.6-plus"),
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
        )
    except Exception:
        return None

    if getattr(resp, "status_code", None) != 200:
        return None
    raw = resp.output.choices[0].message.content
    return _normalize_evaluation(_extract_json_payload(raw), session_id, case, rubric)


def evaluate_training_session(
    session_id: str,
    case_id: str | None = None,
    source_call_id: str | None = None,
    training_type: str | None = None,
    turns: list[dict[str, Any]] | None = None,
    prefer_llm: bool = True,
) -> dict[str, Any]:
    """训练会话评分入口函数。

    优先使用 LLM 评分（更精准），LLM 评分失败时自动 fallback 到启发式评分（更稳定）。

    Args:
        session_id: 会话 ID
        case_id: 案例 ID（用于查找对应的 rubric）
        source_call_id: 原始通话 ID
        training_type: 训练类型
        turns: 对话轮次列表（不传则从数据库读取）
        prefer_llm: 是否优先使用 LLM 评分

    Returns:
        评分结果 dict，包含 total_score、dimension_scores、strengths、improvements 等
    """
    case = get_case(case_id) if case_id else None
    if case:
        source_call_id = source_call_id or case.get("source_call_id")
        training_type = training_type or case.get("training_type")
    rubric = find_rubric(
        case_id=case_id, source_call_id=source_call_id, training_type=training_type)
    session_turns = turns if turns is not None else get_session_turns(
        session_id)

    if prefer_llm:
        evaluation = _llm_evaluation(session_id, case, rubric, session_turns)
        if evaluation:
            return evaluation
    return _heuristic_evaluation(session_id, case, rubric, session_turns)
