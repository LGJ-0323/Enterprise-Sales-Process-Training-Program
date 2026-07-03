"""
prompt_assembler.py — 将 case 数据动态组装成发给千问的完整 prompt

核心职责：
1. 从 case 中提取 state_machine、behavior_rules、few_shot_examples
2. 根据当前状态选取对应的行为规则和 few-shot 示例
3. 组装成结构化 prompt，注入 LLM 调用
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
BASE_SYSTEM_PATH = BASE_DIR / "prompts" / "customer_profile.md"


# ── 内部工具函数 ──────────────────────────────────────────

def _safe_dump(data: Any) -> str:
    """将 Python 对象转为可嵌入 prompt 的文本。"""
    if isinstance(data, (list, dict)):
        return json.dumps(data, ensure_ascii=False, indent=2)
    return str(data)


def _format_history(turns: list[dict[str, Any]], max_turns: int = 6) -> str:
    """将对话历史格式化为 prompt 可用的文本。"""
    if not turns:
        return "（尚无对话记录）"

    recent = turns[-max_turns:]
    lines = []
    for t in recent:
        user = t.get("user_text", "")
        assistant = t.get("assistant_text", t.get("customer_reply", ""))
        lines.append(f"销售：{user}")
        lines.append(f"客户（你）：{assistant}")
    return "\n".join(lines)


# ── 核心组装函数 ──────────────────────────────────────────

def assemble_prompt(
    case: dict[str, Any],
    current_state: str,
    history: list[dict[str, Any]],
    difficulty: str | None = None,
) -> str:
    """将 case 数据和运行时状态组装成完整 system prompt。

    Args:
        case: 从 roleplay_cases.jsonl 加载的案例 dict（v2.0 schema）
        current_state: 当前客户状态（如 "guarded", "warming_up"）
        history: 对话历史列表 [{"user_text": "...", "assistant_text": "..."}, ...]
        difficulty: 当前难度等级（可选，用于加载 difficulty_variants）

    Returns:
        完整的 system prompt 字符串，可直接发给千问。
    """
    parts: list[str] = []

    # ── Part 1: 基础角色定位 ──
    if BASE_SYSTEM_PATH.exists():
        parts.append(BASE_SYSTEM_PATH.read_text(encoding="utf-8").strip())
    else:
        parts.append(
            "你正在扮演公司内部销售训练系统里的「企业客户」。\n"
            "你只扮演客户，不扮演教练，不跳出角色。每次回复 1-3 句话。"
        )

    # ── Part 2: 客户身份 ──
    role = case.get("customer_role_card", {})
    parts.append(
        "## 你扮演的客户\n"
        f"姓名：{role.get('name', '未知')}\n"
        f"职位：{role.get('role', '')}\n"
        f"公司所在地：{role.get('company_location', '未知')}\n"
        f"当前业务状态：{role.get('current_status', '')}\n"
        f"业务背景：{role.get('business_context', '')}\n"
        f"性格：{role.get('personality', '')}\n"
        f"沟通风格：{role.get('communication_style', '')}\n"
        f"决策风格：{role.get('decision_style', '')}"
    )

    # ── Part 3: 隐藏状态 ──
    hidden = case.get("hidden_customer_state", {})
    parts.append(
        "## 你内心知道但不要主动说的信息\n"
        f"初始信任度：{hidden.get('trust_level_at_start', '?')}/100\n"
        f"核心关切：{_safe_dump(hidden.get('main_concerns', []))}\n"
        f"价格敏感度：{hidden.get('price_sensitivity', '?')}\n"
        f"被问到时可以透露：\n{_safe_dump(hidden.get('known_facts_can_reveal_if_asked', []))}\n"
        f"绝对不主动说的：\n{_safe_dump(hidden.get('do_not_reveal_unless_deep_trust', []))}"
    )

    # ── Part 4: 当前状态 + 行为规则 ──
    rules = case.get("customer_behavior_rules", {})
    global_rules = rules.get("global", [])
    state_rules = rules.get("by_state", {}).get(current_state, [])

    sm = case.get("state_machine", {})
    state_desc = sm.get("states", {}).get(current_state, {}).get("description", "")

    parts.append(
        "## 当前状态与行为规则\n"
        f"当前状态：{current_state}（{state_desc}）\n\n"
        "在此状态下你必须遵守：\n"
        + "\n".join(f"  - {r}" for r in global_rules)
        + "\n"
        + "\n".join(f"  - {r}" for r in state_rules)
    )

    # ── Part 5: 状态转移规则 ──
    transitions = sm.get("transitions", [])
    if transitions:
        trans_lines = ["## 状态转移规则（何时改变状态）"]
        for t in transitions:
            fr = t.get("from", "?")
            to = t.get("to", "?")
            trigger = t.get("trigger", "")
            trans_lines.append(f"  - {fr} → {to}: {trigger}")
        parts.append("\n".join(trans_lines))

    # ── Part 6: Few-shot 示例 ★ 最关键 ★ ──
    few_shots = case.get("few_shot_examples", [])
    if few_shots:
        # 优先选取匹配当前状态的示例，再补其他状态
        state_shots = [s for s in few_shots if s.get("state") == current_state]
        other_shots = [s for s in few_shots if s.get("state") != current_state]
        selected = (state_shots + other_shots)[:4]

        shot_lines = ["## 真实对话风格参考（请模仿这种语气和回复方式）"]
        for i, ex in enumerate(selected, 1):
            why = ex.get('why', '')
            # 跳过乱码 why 字段（全是全角问号或占位符的无效内容）
            is_garbled = (
                not why
                or why.count('?') > len(why) * 0.6
                or why.count('？') > len(why) * 0.6
                or why.strip() in ('?', '？', '...')
            )
            example_block = (
                f"示例{i}（{ex.get('state', '?')}状态）：\n"
                f"  销售说：「{ex.get('sales_input', '')}」\n"
                f"  客户回：「{ex.get('customer_reply', '')}」"
            )
            if not is_garbled:
                example_block += f"\n  要点：{why}"
            shot_lines.append(example_block)
        parts.append("\n".join(shot_lines))

    # ── Part 7: 可能异议 ──
    objections = case.get("likely_objections", [])
    if objections:
        obj_lines = ["## 你可以使用的异议话术"]
        for o in objections:
            obj_lines.append(
                f"  - 当销售「{o.get('trigger', '')}」时 → "
                f"回复：「{o.get('objection', '')}」"
            )
        parts.append("\n".join(obj_lines))

    # ── Part 8: 失败红线 ──
    failures = case.get("failure_conditions", [])
    if failures:
        fail_lines = ["## ⚠️ 绝对红线（一旦触发立即进入 shut_down）"]
        for f in failures:
            fail_lines.append(
                f"  - 如果销售「{f.get('condition', '')}」→ "
                f"你的反应：{f.get('customer_reaction', '')}"
            )
        parts.append("\n".join(fail_lines))

    # ── Part 9: 难度参数覆盖 ──
    if difficulty:
        variants = case.get("difficulty_variants", {})
        variant = variants.get(difficulty)
        if variant:
            parts.append(
                f"## 难度参数（当前: {difficulty}）\n"
                f"信任起点：{variant.get('trust_start', '?')}/100\n"
                f"异议数量：{variant.get('objection_count', '?')}个\n"
                f"异议强度：{variant.get('objection_intensity', '?')}\n"
                f"信息主动透露量：{variant.get('info_volunteered', '?')}条"
            )

    # ── Part 10: 对话历史 ──
    parts.append(f"## 当前对话\n{_format_history(history)}")

    # ── Part 11: 输出格式 ──
    valid_states = list(sm.get("states", {}).keys()) or [current_state]
    state_options = "/".join(str(state) for state in valid_states)
    parts.append(
        "## 输出要求\n"
        "只输出以下 JSON，不要任何其他文字：\n"
        '{"customer_reply": "你的回复（1-3句话，纯文本不要前缀）", '
        f'"next_state": "新状态（{state_options}之一；如果状态不变则输出当前状态）", '
        '"triggered_events": ["触发的事件"], '
        '"score_notes": {"相关维度": "评分依据简要说明"}}'
    )

    return "\n\n".join(parts)


# ── 快速调用入口 ──

def assemble_from_case_id(
    case_id: str,
    current_state: str = "guarded",
    history: list[dict[str, Any]] | None = None,
) -> str | None:
    """通过 case_id 快速组装 prompt（需要先调用 case_loader 加载 case）。

    Args:
        case_id: 案例 ID（如 "case_20260630102634"）
        current_state: 当前状态
        history: 对话历史

    Returns:
        组装好的 prompt，如果没有找到 case 返回 None。
    """
    from .case_loader import _load_all_cases

    cases = _load_all_cases()
    for case in cases:
        if case.get("case_id") == case_id:
            return assemble_prompt(case, current_state, history or [])
    return None
