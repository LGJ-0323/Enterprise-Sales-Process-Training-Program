"""
app_new_web.py - 国际物流模拟客户陪练控制台

Gradio-first shell:
- /app: Gradio Blocks dashboard styled like docs/ui-mockups/logistics-training-dashboard.png
- /stream: existing FastRTC/Gradio voice training UI
- /api/*: light JSON endpoints for the dashboard and future report export
"""

from __future__ import annotations

import html
import os
import random
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse, RedirectResponse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

try:
    from .case_loader import case_count as get_case_count, find_case, find_cases, get_case
    from .fastrtc_new_web import LAST_STATUS, set_active_stream_selection, stream
    from .training_session import get_or_create_session_context
    from .training_config import (
        _label,
        difficulty_choices,
        resolve_training,
        resolve_voice,
        stage_choices,
        voice_choices,
    )
    from .training_data_context import recent_training_records, summarize_case_assets
except ImportError:
    from case_loader import case_count as get_case_count, find_case, find_cases, get_case
    from fastrtc_new_web import LAST_STATUS, set_active_stream_selection, stream
    from training_session import get_or_create_session_context
    from training_config import (
        _label,
        difficulty_choices,
        resolve_training,
        resolve_voice,
        stage_choices,
        voice_choices,
    )
    from training_data_context import recent_training_records, summarize_case_assets


DEFAULT_STAGE_ID = os.getenv("TRAINING_STAGE_ID", "cold_call")
DEFAULT_DIFFICULTY_ID = os.getenv("TRAINING_DIFFICULTY_ID", "easy")
DEFAULT_VOICE_ID = os.getenv("TRAINING_VOICE_ID", "longsanshu_v3")


def _choice(items: list[tuple[str, str]]) -> list[dict[str, str]]:
    """将 (label, value) 列表转为 Gradio Radio/Select 组件所需的 dict 格式。"""
    return [{"label": label, "value": value} for label, value in items]


def _escape(value: Any) -> str:
    """HTML 转义，防止 XSS 注入。"""
    return html.escape(str(value or ""), quote=True)


def _clip(value: Any, limit: int = 90) -> str:
    """文本截断，超过 limit 字符时加省略号。"""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _label_for(choices: list[tuple[str, str]], value: str) -> str:
    """根据 value 在 choices 中查找对应的显示标签。"""
    for label, item_value in choices:
        if item_value == value:
            return label
    return value


def _resolve_case(
    stage_id: str,
    difficulty_id: str,
    case_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """解析训练配置并匹配案例，返回 (stage, customer, difficulty, case)。"""
    stage, customer, difficulty = resolve_training(stage_id, None, difficulty_id)
    case = get_case(case_id) if case_id else None
    if not case:
        candidates = find_cases(_label(stage), _label(difficulty))
        case = random.choice(candidates) if candidates else None
    return stage, customer, difficulty, case


_STATE_SIGS: dict[str, str] = {}


def _dashboard_case_id(session_id: str | None, stage_id: str, difficulty_id: str, case_id: str | None = None) -> tuple[str, str, str]:
    """为 Dashboard 生成或复用 session_id，并绑定案例。

    当训练阶段或难度变化时（状态签名变了），清空旧 case 重新随机抽取；
    同一 (stage, difficulty) 组合下则复用已选案例。
    """
    session = session_id or uuid.uuid4().hex
    stage, _, difficulty = resolve_training(stage_id, None, difficulty_id)
    stage_label = _label(stage)
    diff_label = _label(difficulty)
    state_sig = f"{stage_label}|{diff_label}"
    key = f"dashboard-{session}"

    # 状态签名变了 → 清空旧 case，强制重新随机
    old_sig = _STATE_SIGS.get(key)
    if old_sig and old_sig != state_sig:
        case_id = None
    _STATE_SIGS[key] = state_sig

    context = get_or_create_session_context(
        key,
        stage_label,
        diff_label,
        preferred_case_id=case_id,
    )
    # 附上候选数，供 UI 展示
    raw_count = context.get("candidate_count", 0) if isinstance(context, dict) else 0
    candidate_count = int(raw_count) if isinstance(raw_count, (int, float)) else 0
    return session, str(context.get("case_id") or ""), str(candidate_count)


def _tag(text: Any) -> str:
    """将文本包装为 HTML chip 标签样式。"""
    value = _escape(text)
    if not value:
        return ""
    return f'<span class="chip">{value}</span>'


def render_header() -> str:
    """渲染 Dashboard 顶部导航栏 HTML。"""
    return """
<header class="dash-topbar">
  <div class="brand">
    <div class="brand-mark">训</div>
    <div>
      <h1>国际物流模拟客户陪练控制台</h1>
      <div class="subtitle">阶段 / 难度 / 客户音色 / 实时语音 / 智能复盘，一站式训练闭环</div>
    </div>
  </div>
  <div class="top-actions">
    <div class="pill"><span class="dot"></span> FastRTC 实时连接中</div>
    <div class="pill">本轮 <span id="round-time">00:00</span></div>
    <a class="button" href="/api/training-assets" target="_blank">导出复盘报告</a>
  </div>
</header>
"""


def render_persona(stage_id: str, difficulty_id: str, voice_id: str, case_id: str | None = None, candidate_count: str = "") -> str:
    """渲染客户人物卡片 HTML（头像、姓名、职位、关切标签）。"""
    stage, customer, difficulty, case = _resolve_case(stage_id, difficulty_id, case_id)
    assets = summarize_case_assets(case)
    voice = resolve_voice(voice_id)

    name = assets.get("customer_name") or customer.get("name", "客户")
    role = assets.get("customer_role") or customer.get("role", "")
    style = assets.get("communication_style") or customer.get("attitude", {}).get("label", "")
    concerns = assets.get("main_concerns") or customer.get("pain_points") or []
    if isinstance(concerns, str):
        concerns = [concerns]
    chips = list(concerns[:4])
    if assets.get("price_sensitivity"):
        chips.append(f"价格敏感度 {assets['price_sensitivity']}")
    if assets.get("few_shot_count"):
        chips.append(f"{assets['few_shot_count']} 个对话示例")

    desc = assets.get("scene") or assets.get("raw_call_summary") or customer.get("attitude", {}).get("style", "")
    role_line = " / ".join(part for part in (role, style) if part)

    # 候选数提示：候选=1 时提示"数据只有1个"，避免误判随机失效
    count_hint = ""
    count_int = int(candidate_count) if candidate_count.isdigit() else 0
    if count_int == 1:
        count_hint = '<span class="tag warn" title="当前组合仅有 1 个案例，不会随机">候选 1</span>'
    elif count_int > 1:
        count_hint = f'<span class="tag" title="每次从 {count_int} 个案例中随机抽取">候选 {count_int}</span>'

    return f"""
<div class="persona-card">
  <div class="avatar"><div class="face"></div></div>
  <div>
    <div class="persona-name">{_escape(name)}</div>
    <div class="persona-role">{_escape(role_line)}</div>
    <div class="persona-desc">{_escape(_clip(desc, 86))}</div>
  </div>
</div>
<div class="chips">{''.join(_tag(chip) for chip in chips) or _tag("等待客户画像")}</div>
<div class="mini-meta">
  <span>{_escape(_label(stage))}</span>
  <span>{_escape(_label(difficulty))}</span>
  <span>{_escape(voice.get("label", voice_id))}</span>
  {count_hint}
</div>
"""


def render_goal(stage_id: str, difficulty_id: str, case_id: str | None = None) -> str:
    """渲染训练目标 HTML。"""
    stage, _, _, case = _resolve_case(stage_id, difficulty_id, case_id)
    assets = summarize_case_assets(case)
    goal = stage.get("training_goal") or "完成有效开场，挖掘客户需求，并推进到下一步行动。"
    if assets.get("current_status"):
        goal = f"{goal}\n当前客户状态：{assets['current_status']}"
    return f'<div class="goal-box">{_escape(goal)}</div>'


def render_voice_room(stage_id: str, difficulty_id: str, voice_id: str) -> str:
    """渲染实时语音陪练房间 HTML。"""
    voice_label = _label_for(voice_choices(), voice_id)
    voice_name = voice_label.split("-", 1)[0].strip() or voice_label
    return f"""
<section class="voice-room">
  <div class="panel-header">
    <div class="panel-title">实时语音陪练</div>
    <div class="call-state"><span class="dot"></span>客户正在发言，VAD 已检测到语音</div>
  </div>
  <div class="voice-stage">
    <div class="voice-main">
      <div class="speaker">
        <div class="speaker-card ai"><div class="pulse-ring"></div><div class="headset">AI</div></div>
        <div class="speaker-name">模拟客户</div>
        <div class="speaker-note">{_escape(voice_name)}音色 · 情绪冷淡</div>
      </div>
      <div class="wave" aria-hidden="true">{''.join('<span class="bar"></span>' for _ in range(9))}</div>
      <div class="speaker">
        <div class="speaker-card client"><div class="headset">我</div></div>
        <div class="speaker-name">销售学员</div>
        <div class="speaker-note">麦克风正常 · 可随时插话</div>
      </div>
    </div>
    <div class="call-controls" aria-label="开始通话">
      <div class="stream-native-control">
        <iframe title="FastRTC 原生开始通话按钮" src="/stream" allow="microphone; autoplay; camera"></iframe>
      </div>
    </div>
  </div>
</section>
"""


def _current_scores(status: dict[str, Any], assets: dict[str, Any]) -> tuple[str, list[tuple[str, int]]]:
    evaluation = status.get("evaluation") if isinstance(status.get("evaluation"), dict) else None
    if evaluation:
        rows: list[tuple[str, int]] = []
        for item in (evaluation.get("dimension_scores") or [])[:4]:
            max_score = int(item.get("max_score") or 100)
            score = int(item.get("score") or 0)
            percent = round(score * 100 / max_score) if max_score else 0
            label = f"{item.get('dimension', '评分维度')} {score}/{max_score}"
            rows.append((label, percent))
        return str(evaluation.get("total_score", "--")), rows

    prompt = str(status.get("prompt") or "")
    response = str(status.get("response_text") or "")
    if not prompt and not response:
        dims = assets.get("rubric_dimensions") or [
            {"dimension": "开场清晰度", "score": 88},
            {"dimension": "需求挖掘", "score": 76},
            {"dimension": "异议处理", "score": 69},
        ]
        return "--", [(d.get("dimension", "评分维度"), int(d.get("score") or 0)) for d in dims[:3]]

    opening = 88 if any(word in prompt for word in ("您好", "你好", "我是", "联系")) else 62
    needs = 84 if any(word in prompt for word in ("需求", "出货", "美线", "欧洲", "货量", "时效", "报价")) else 66
    objection = 78 if any(word in prompt + response for word in ("货代", "价格", "贵", "固定", "供应商", "比")) else 61
    score = round((opening + needs + objection) / 3)
    return str(score), [("开场清晰度", opening), ("需求挖掘", needs), ("异议处理", objection)]


def render_score(
    stage_id: str,
    difficulty_id: str,
    status: dict[str, Any] | None = None,
    case_id: str | None = None,
) -> str:
    _, _, _, case = _resolve_case(stage_id, difficulty_id, case_id)
    assets = summarize_case_assets(case)
    status = status or LAST_STATUS
    evaluation = status.get("evaluation") if isinstance(status.get("evaluation"), dict) else None
    score, metrics = _current_scores(status, assets)
    rows = []
    for name, value in metrics[:3]:
        width = min(max(value, 0), 100)
        rows.append(
            f"""
<div class="metric">
  <div class="metric-head"><span>{_escape(name)}</span><span>{width}%</span></div>
  <div class="track"><div class="fill" style="width:{width}%"></div></div>
</div>
"""
        )

    sub = "等待本轮训练" if score == "--" else "正式复盘得分 / 100" if evaluation else "本轮综合得分 / 100"
    tag = "Rubric 已评分" if evaluation else "评分标准已接入"
    return f"""
<section class="score-card">
  <div class="score-row">
    <div>
      <div class="score">{_escape(score)}</div>
      <div class="score-sub">{_escape(sub)}</div>
    </div>
    <span class="tag">{_escape(tag)}</span>
  </div>
  {''.join(rows)}
</section>
"""


def render_coach_feedback(
    stage_id: str,
    difficulty_id: str,
    status: dict[str, Any] | None = None,
    case_id: str | None = None,
) -> str:
    _, _, _, case = _resolve_case(stage_id, difficulty_id, case_id)
    assets = summarize_case_assets(case)
    status = status or LAST_STATUS
    evaluation = status.get("evaluation") if isinstance(status.get("evaluation"), dict) else None
    prompt = str(status.get("prompt") or "")

    if evaluation:
        strengths = evaluation.get("strengths") or ["完成了本轮训练，可进入复盘。"]
        improvements = evaluation.get("improvements") or ["下一轮补足需求追问和下一步动作。"]
        summary = evaluation.get("summary") or "已按当前 rubric 完成评分。"
        feedback = [
            ("good", "做得好", strengths[0]),
            ("warn", "可加强", improvements[0]),
            ("bad", "复盘结论", summary),
        ]
    elif prompt:
        needs_hint = "需求问题更具体会更好，可追问货量、品名、目的港和近期发货节奏。"
        if any(word in prompt for word in ("出货", "货量", "美线", "时效", "价格")):
            needs_hint = "已经开始切到业务问题，下一步可以复述客户需求后给出备选方案。"
        feedback = [
            ("good", "做得好", "先围绕客户当前回答推进，没有跳出客户场景。"),
            ("warn", "可加强", needs_hint),
            ("bad", "风险点", "如果只介绍公司优势，容易被客户归类为普通报价推销。"),
        ]
    else:
        must = assets.get("must_do") or ["清晰说明来意和身份", "挖掘客户真实需求", "明确下一步行动计划"]
        critical = assets.get("critical_mistakes") or ["承诺无法兑现的价格或服务"]
        feedback = [
            ("good", "训练重点", must[0]),
            ("warn", "建议关注", must[1] if len(must) > 1 else "先确认客户当前状态，再给方案。"),
            ("bad", "扣分红线", critical[0]),
        ]

    items = []
    icons = {"good": "✓", "warn": "!", "bad": "×"}
    for kind, title, body in feedback:
        items.append(
            f"""
<div class="feedback">
  <div class="feedback-icon {kind}">{icons[kind]}</div>
  <div><strong>{_escape(title)}：</strong>{_escape(body)}</div>
</div>
"""
        )
    return f"""
<section class="feedback-panel">
  <div class="panel-header">
    <div class="panel-title">AI 教练反馈</div>
    <span class="tag">{_escape("正式复盘" if evaluation else "实时生成")}</span>
  </div>
  <div class="feedback-list">{''.join(items)}</div>
</section>
"""


def render_transcript(status: dict[str, Any] | None = None) -> str:
    """渲染对话记录 HTML（销售/客户气泡）。"""
    status = status or LAST_STATUS
    prompt = str(status.get("prompt") or "")
    response = str(status.get("response_text") or "")
    guardrail = str(status.get("guardrail") or "")
    stage = str(status.get("stage") or "waiting")
    error = str(status.get("error") or "")

    if error:
        return f"""
<div class="dialogue">
  <div class="bubble ai error">
    <div class="speaker-line"><span>系统提示</span><span>{_escape(stage)}</span></div>
    {_escape(error)}
  </div>
</div>
"""

    if not prompt and not response:
        return """
<div class="dialogue empty">
  <div class="bubble ai">
    <div class="speaker-line"><span>AI 实时建议</span><span>等待训练</span></div>
    开始语音练习后，这里会展示学员转写、客户回复和基于评分标准的提示。
  </div>
</div>
"""

    bubbles = []
    if prompt:
        bubbles.append(
            f"""
<div class="bubble trainee">
  <div class="speaker-line"><span>学员回复</span><span>刚刚</span></div>
  {_escape(prompt)}
</div>
"""
        )
    if response:
        bubbles.append(
            f"""
<div class="bubble client">
  <div class="speaker-line"><span>客户转写</span><span>{_escape(stage)}</span></div>
  {_escape(response)}
</div>
"""
        )
    if guardrail:
        bubbles.append(
            f"""
<div class="bubble ai">
  <div class="speaker-line"><span>AI 实时建议</span><span>Guardrail</span></div>
  已触发 {_escape(guardrail)}，系统已自动回到客户角色边界。
</div>
"""
        )
    return f'<div class="dialogue">{"".join(bubbles)}</div>'


def render_history() -> str:
    records = recent_training_records(limit=3)
    if not records:
        records = [
            {"time": "今天", "title": "陌call · 进阶异议", "desc": "客户拒绝推销，成功推进至发送方案", "score": "82"},
            {"time": "昨天", "title": "报价跟进 · 中等", "desc": "价格解释充分，成交推进不足", "score": "74"},
            {"time": "周五", "title": "老客户唤醒 · 简单", "desc": "开场自然，案例引用准确", "score": "91"},
        ]

    rows = []
    for index, item in enumerate(records[:3], 1):
        score = item.get("score") or str(max(68, 92 - index * 7))
        rows.append(
            f"""
<div class="history-item">
  <div class="time">{_escape(item.get("time"))}</div>
  <div>
    <div class="history-title">{_escape(_clip(item.get("title"), 18))}</div>
    <div class="history-desc">{_escape(_clip(item.get("desc"), 28))}</div>
  </div>
  <div class="mini-score">{_escape(score)}</div>
</div>
"""
        )
    return f"""
<section class="history-panel">
  <div class="panel-header">
    <div class="panel-title">训练复盘记录</div>
    <span class="tag">自动沉淀</span>
  </div>
  <div class="history">{''.join(rows)}</div>
</section>
"""


def render_session_info(status: dict[str, Any] | None = None) -> str:
    status = status or LAST_STATUS
    training = status.get("training") or {}
    timings = status.get("timings") or {}
    details = [
        f"状态：{status.get('stage', 'waiting')}",
        f"阶段：{training.get('stage', training.get('stage_id', '-'))}",
        f"客户：{training.get('customer', '-')}",
    ]
    if timings.get("total_s"):
        details.append(f"总耗时：{timings['total_s']}s")
    if timings.get("asr_s"):
        details.append(f"ASR：{timings['asr_s']}s")
    if timings.get("qwen_s"):
        details.append(f"LLM：{timings['qwen_s']}s")
    if training.get("current_state"):
        details.append(f"客户状态：{training.get('current_state')}")
    if training.get("final_state"):
        result = "成功" if training.get("is_success") else "失败" if training.get("is_failure") else "结束"
        details.append(f"终局：{training.get('final_state')} / {result}")
    evaluation = status.get("evaluation") if isinstance(status.get("evaluation"), dict) else None
    if evaluation:
        details.append(f"复盘得分：{evaluation.get('total_score')}/100")
    if status.get("error"):
        details.append(f"错误：{_clip(status.get('error'), 120)}")
    return f'<div class="session-info">{" · ".join(_escape(item) for item in details)}</div>'


def update_training_view(
    stage_id: str,
    difficulty_id: str,
    voice_id: str,
    session_id: str | None,
    case_id: str | None,
) -> tuple[str, str, str, str, str, str, str]:
    session_id, selected_case_id, candidate_count = _dashboard_case_id(session_id, stage_id, difficulty_id, case_id)
    set_active_stream_selection(stage_id, difficulty_id, voice_id, selected_case_id)
    return (
        session_id,
        selected_case_id,
        render_persona(stage_id, difficulty_id, voice_id, selected_case_id, candidate_count),
        render_goal(stage_id, difficulty_id, selected_case_id),
        render_voice_room(stage_id, difficulty_id, voice_id),
        render_score(stage_id, difficulty_id, case_id=selected_case_id),
        render_coach_feedback(stage_id, difficulty_id, case_id=selected_case_id),
    )


def poll_live_view(stage_id: str, difficulty_id: str, case_id: str | None) -> tuple[str, str, str, str]:
    status = dict(LAST_STATUS)
    live_case_id = (status.get("training") or {}).get("case_id") or case_id
    return (
        render_transcript(status),
        render_score(stage_id, difficulty_id, status, live_case_id),
        render_coach_feedback(stage_id, difficulty_id, status, live_case_id),
        render_session_info(status),
    )


DASHBOARD_CSS = """
:root {
  --bg: #eef3f8;
  --panel: #ffffff;
  --ink: #10243f;
  --muted: #64748b;
  --line: #d9e2ee;
  --teal: #13b8a6;
  --teal-soft: #d8faf5;
  --blue: #2563eb;
  --blue-soft: #e5edff;
  --amber-soft: #fff5d6;
  --red: #ef4444;
  --red-soft: #ffe5e5;
  --green: #22c55e;
  --green-soft: #defbe6;
  --shadow: 0 18px 48px rgba(18, 38, 63, 0.13);
}

* {
  box-sizing: border-box;
}

.gradio-container {
  min-height: 100vh !important;
  max-width: none !important;
  background:
    radial-gradient(circle at 12% 12%, rgba(19, 184, 166, 0.16), transparent 32%),
    radial-gradient(circle at 88% 8%, rgba(37, 99, 235, 0.14), transparent 34%),
    linear-gradient(135deg, #f8fbff 0%, var(--bg) 100%) !important;
  color: var(--ink);
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
  letter-spacing: 0;
}

main.app {
  width: 100% !important;
  max-width: 1440px !important;
  margin: 0 auto;
  padding: 34px 42px 18px !important;
}

.html-container.padding {
  padding: 0 !important;
}

.html-container .prose {
  max-width: none !important;
}

.dash-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.brand {
  display: flex;
  gap: 14px;
  align-items: center;
}

.brand-mark {
  width: 48px;
  height: 48px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--teal), var(--blue));
  color: #fff;
  font-size: 25px;
  font-weight: 800;
  box-shadow: 0 14px 28px rgba(19, 184, 166, 0.28);
}

h1 {
  margin: 0;
  font-size: 30px;
  line-height: 1.2;
  font-weight: 850;
  color: #061938;
}

.subtitle {
  margin-top: 6px;
  color: var(--muted);
  font-size: 15px;
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.pill,
.button,
.tag,
.kbd {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  font-weight: 700;
  text-decoration: none;
}

.pill {
  gap: 7px;
  height: 32px;
  padding: 0 12px;
  background: #fff;
  border: 1px solid var(--line);
  color: var(--muted);
  font-size: 13px;
}

.button {
  height: 38px;
  padding: 0 16px;
  border-radius: 8px;
  border: 0;
  background: #10243f;
  color: #fff !important;
  font-size: 14px;
  box-shadow: 0 10px 20px rgba(16, 36, 63, 0.18);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.16);
}

.dashboard-grid {
  display: grid !important;
  grid-template-columns: 318px minmax(520px, 1fr) 346px;
  gap: 18px;
  align-items: stretch;
}

.dashboard-grid > .form {
  min-width: 0 !important;
}

.left-stack,
.center-stack,
.right-stack {
  min-width: 0;
}

.left-panel,
.voice-room,
.conversation-panel,
.score-card,
.feedback-panel,
.history-panel {
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid var(--line);
  border-radius: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
}

.left-panel {
  padding: 0;
}

.panel-header {
  min-height: 54px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 18px 12px;
  border-bottom: 1px solid #edf2f7;
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 16px;
  font-weight: 820;
  color: var(--ink);
}

.tag {
  height: 26px;
  padding: 0 9px;
  border-radius: 7px;
  background: var(--teal-soft);
  color: #078b7f;
  font-size: 12px;
}

.tag.warn {
  background: #fff5d6;
  color: #b45309;
}

.config-section,
.persona-section,
.goal-section {
  padding: 14px 18px;
  border-bottom: 1px solid #edf2f7;
}

.goal-section {
  border-bottom: 0;
}

.left-panel .wrap {
  gap: 0 !important;
}

.left-panel label span {
  color: #34506f !important;
  font-size: 13px !important;
  font-weight: 780 !important;
}

.left-panel .input-container,
.left-panel .wrap-inner {
  border-radius: 8px !important;
}

.persona-card {
  display: grid;
  grid-template-columns: 86px 1fr;
  gap: 14px;
  align-items: center;
  padding: 12px;
  border: 1px solid #dce5f0;
  border-radius: 10px;
  background: linear-gradient(180deg, #fff, #f8fbff);
}

.avatar {
  position: relative;
  width: 78px;
  height: 78px;
  border-radius: 50%;
  background: #ecfeff;
  overflow: hidden;
  border: 1px solid #bdeee8;
}

.avatar::before {
  content: "";
  position: absolute;
  left: 20px;
  top: 16px;
  width: 42px;
  height: 34px;
  border-radius: 20px 20px 14px 14px;
  background: #111827;
}

.avatar::after {
  content: "";
  position: absolute;
  left: 18px;
  bottom: 10px;
  width: 46px;
  height: 44px;
  border-radius: 18px 18px 12px 12px;
  background: linear-gradient(140deg, #0f766e, #14b8a6);
  box-shadow: inset 18px 0 0 rgba(255, 255, 255, 0.14);
}

.face {
  position: absolute;
  left: 25px;
  top: 33px;
  width: 32px;
  height: 32px;
  z-index: 2;
  border-radius: 11px 11px 16px 16px;
  background: #ffc7a8;
}

.face::before,
.face::after {
  content: "";
  position: absolute;
  top: 14px;
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #334155;
}

.face::before {
  left: 8px;
}

.face::after {
  right: 8px;
}

.persona-name {
  font-size: 20px;
  font-weight: 880;
  margin-bottom: 3px;
}

.persona-role {
  color: #078b7f;
  font-size: 14px;
  font-weight: 820;
  line-height: 1.35;
}

.persona-desc {
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.chip {
  min-height: 26px;
  display: inline-flex;
  align-items: center;
  padding: 4px 8px;
  border-radius: 7px;
  background: #f1f5f9;
  color: #475569;
  font-size: 12px;
  font-weight: 700;
}

.mini-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 12px;
  color: #64748b;
  font-size: 12px;
}

.goal-box {
  min-height: 84px;
  padding: 11px 12px;
  border: 1px solid #dce5f0;
  background: #fff;
  border-radius: 8px;
  color: #405875;
  font-size: 13px;
  line-height: 1.55;
  white-space: pre-line;
}

.center-stack {
  display: grid !important;
  grid-template-rows: auto auto;
  gap: 16px;
}

.voice-room {
  min-height: 416px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,251,255,0.95)),
    radial-gradient(circle at 24% 48%, rgba(20,184,166,0.10), transparent 28%),
    radial-gradient(circle at 78% 42%, rgba(245,158,11,0.08), transparent 24%);
}

.call-state {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #078b7f;
  font-size: 13px;
  font-weight: 820;
}

.voice-stage {
  padding: 14px 22px 18px;
}

.voice-main {
  min-height: 274px;
  display: grid;
  grid-template-columns: minmax(150px, 1fr) 156px minmax(150px, 1fr);
  align-items: center;
  gap: 22px;
  padding: 0 34px;
}

.speaker {
  display: grid;
  justify-items: center;
  gap: 9px;
  min-width: 0;
}

.speaker-card {
  width: 176px;
  height: 176px;
  position: relative;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: #fff;
  border: 1px solid #dce5f0;
  box-shadow: 0 18px 38px rgba(18, 38, 63, 0.10);
}

.speaker-card.ai {
  background:
    radial-gradient(circle at center, rgba(236,254,255,0.96) 0 45%, rgba(236,254,255,0.45) 46%),
    linear-gradient(135deg, #ecfeff, #eef6ff);
}

.speaker-card.client {
  background:
    radial-gradient(circle at center, #fff 0 48%, rgba(255,247,237,0.72) 49%),
    linear-gradient(135deg, #fff7ed, #fff);
}

.headset {
  width: 78px;
  height: 78px;
  border-radius: 23px;
  background: linear-gradient(135deg, #2563eb, var(--teal));
  display: grid;
  place-items: center;
  color: #fff;
  font-size: 39px;
  font-weight: 900;
  line-height: 1;
  box-shadow: 0 14px 26px rgba(37, 99, 235, 0.18);
}

.pulse-ring,
.pulse-ring::before,
.pulse-ring::after {
  position: absolute;
  inset: 16px;
  border-radius: 50%;
  border: 2px solid rgba(20, 184, 166, 0.22);
}

.pulse-ring::before,
.pulse-ring::after {
  content: "";
  inset: -14px;
  opacity: 0.66;
}

.pulse-ring::after {
  inset: -30px;
  opacity: 0.34;
}

.speaker-name {
  font-size: 18px;
  font-weight: 860;
  color: var(--ink);
}

.speaker-note {
  color: #7186a2;
  font-size: 13px;
  font-weight: 680;
  text-align: center;
  line-height: 1.3;
}

.wave {
  height: 118px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
}

.bar {
  width: 9px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--teal), #2563eb);
  box-shadow: 0 9px 18px rgba(37, 99, 235, 0.20);
}

.bar:nth-child(1) { height: 36px; }
.bar:nth-child(2) { height: 64px; }
.bar:nth-child(3) { height: 92px; }
.bar:nth-child(4) { height: 56px; }
.bar:nth-child(5) { height: 112px; }
.bar:nth-child(6) { height: 78px; }
.bar:nth-child(7) { height: 48px; }
.bar:nth-child(8) { height: 88px; }
.bar:nth-child(9) { height: 64px; }

.call-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 70px;
  margin-top: 8px;
}

.stream-native-control {
  width: 270px;
  height: 88px;
  position: relative;
  overflow: hidden;
  border-radius: 14px;
  background: transparent;
}

.stream-native-control iframe {
  position: absolute;
  left: 50%;
  top: 0;
  width: 640px;
  height: 420px;
  border: 0;
  transform: translate(-50%, -244px);
  transform-origin: center top;
}

.conversation-panel {
  min-height: 330px;
}

.dialogue {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.bubble {
  max-width: 78%;
  padding: 10px 12px;
  border-radius: 11px;
  font-size: 13px;
  line-height: 1.45;
  box-shadow: 0 8px 20px rgba(18, 38, 63, 0.06);
}

.bubble.client {
  justify-self: start;
  background: #fff7ed;
  border: 1px solid #fed7aa;
}

.bubble.trainee {
  justify-self: end;
  background: #eaf7ff;
  border: 1px solid #bfdbfe;
}

.bubble.ai {
  max-width: 92%;
  background: #f0fdfa;
  border: 1px solid #99f6e4;
}

.bubble.error {
  max-width: 96%;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  color: #9a3412;
}

.speaker-line {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 5px;
  color: #506a89;
  font-size: 12px;
  font-weight: 800;
}

.right-stack {
  display: grid !important;
  grid-template-rows: auto auto auto;
  gap: 16px;
}

.score-card {
  padding: 16px 18px;
}

.score-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 14px;
}

.score {
  font-size: 50px;
  line-height: 1;
  font-weight: 900;
  color: #0f766e;
}

.score-sub {
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.metric {
  margin-top: 12px;
}

.metric-head {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
  color: #405875;
  font-size: 12px;
  font-weight: 760;
}

.track {
  height: 8px;
  border-radius: 999px;
  background: #e8eef6;
  overflow: hidden;
}

.fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--teal), var(--blue));
}

.feedback-list {
  padding: 16px 18px 18px;
  display: grid;
  gap: 10px;
}

.feedback {
  display: grid;
  grid-template-columns: 30px 1fr;
  gap: 10px;
  padding: 10px;
  border-radius: 9px;
  background: #f8fbff;
  border: 1px solid #e2e8f0;
  font-size: 13px;
  line-height: 1.45;
}

.feedback-icon {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  font-weight: 900;
}

.good {
  background: var(--green-soft);
  color: #15803d;
}

.warn {
  background: var(--amber-soft);
  color: #b45309;
}

.bad {
  background: var(--red-soft);
  color: #dc2626;
}

.history {
  padding: 14px 18px 18px;
}

.history-item {
  display: grid;
  grid-template-columns: 50px 1fr 44px;
  gap: 10px;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #edf2f7;
}

.history-item:last-child {
  border-bottom: 0;
}

.time {
  color: var(--muted);
  font-size: 12px;
  font-weight: 720;
}

.history-title {
  font-size: 13px;
  font-weight: 800;
}

.history-desc {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}

.mini-score {
  height: 28px;
  display: grid;
  place-items: center;
  border-radius: 7px;
  background: var(--blue-soft);
  color: var(--blue);
  font-size: 13px;
  font-weight: 850;
}

.session-info {
  margin-top: 10px;
  padding: 10px 12px;
  color: #405875;
  background: #fff;
  border: 1px solid #dce5f0;
  border-radius: 8px;
  font-size: 12px;
  line-height: 1.5;
}

.footer-note {
  margin-top: 12px;
  display: flex;
  justify-content: center;
  gap: 10px;
  color: #8ca0bb;
  font-size: 13px;
  font-weight: 650;
}

.kbd {
  color: #405875;
  background: #fff;
  border: 1px solid #dce5f0;
  border-radius: 6px;
  padding: 3px 7px;
}

@media (max-width: 1320px) {
  main.app {
    padding: 24px !important;
  }

  .dashboard-grid {
    grid-template-columns: 300px minmax(470px, 1fr) 320px !important;
    gap: 14px;
  }

  .dash-topbar {
    gap: 16px;
  }

  .brand-mark {
    width: 42px;
    height: 42px;
  }

  h1 {
    font-size: 26px;
  }

  .subtitle {
    font-size: 13px;
  }

  .top-actions {
    gap: 8px;
  }

  .pill {
    height: 30px;
    padding: 0 10px;
    font-size: 12px;
  }

  .button {
    height: 34px;
    padding: 0 12px;
    font-size: 12px;
  }

  .voice-main {
    grid-template-columns: 1fr 120px 1fr;
    gap: 14px;
    padding: 0 22px;
  }

  .speaker-card {
    width: 146px;
    height: 146px;
  }

  .headset {
    width: 66px;
    height: 66px;
    font-size: 32px;
  }

  .wave {
    gap: 5px;
  }

  .bar {
    width: 7px;
  }
}

@media (max-width: 1080px) {
  .dashboard-grid {
    grid-template-columns: 1fr !important;
  }

  .voice-main {
    grid-template-columns: 1fr 116px 1fr;
    padding: 0;
  }

  .speaker-card {
    width: 136px;
    height: 136px;
  }

  .call-controls {
    flex-wrap: wrap;
  }
}
"""


def build_dashboard() -> gr.Blocks:
    with gr.Blocks(css=DASHBOARD_CSS, title="国际物流模拟客户陪练控制台") as dashboard:
        dashboard_session = gr.State("")
        selected_case = gr.State("")
        gr.HTML(render_header())
        with gr.Row(elem_classes=["dashboard-grid"]):
            with gr.Column(elem_classes=["left-stack"]):
                with gr.Group(elem_classes=["left-panel"]):
                    gr.HTML('<div class="panel-header"><div class="panel-title">训练配置</div><span class="tag">可实时切换</span></div>')
                    with gr.Group(elem_classes=["config-section"]):
                        stage = gr.Dropdown(stage_choices(), value=DEFAULT_STAGE_ID, label="训练阶段", interactive=True)
                        difficulty = gr.Dropdown(
                            difficulty_choices(),
                            value=DEFAULT_DIFFICULTY_ID,
                            label="难度等级",
                            interactive=True,
                        )
                        voice = gr.Dropdown(voice_choices(), value=DEFAULT_VOICE_ID, label="客户音色", interactive=True)
                    with gr.Group(elem_classes=["persona-section"]):
                        gr.HTML('<span class="panel-title">本轮客户画像</span>')
                        persona = gr.HTML(render_persona(DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID))
                    with gr.Group(elem_classes=["goal-section"]):
                        gr.HTML('<span class="panel-title">陪练目标</span>')
                        goal = gr.HTML(render_goal(DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID))

            with gr.Column(elem_classes=["center-stack"]):
                voice_room = gr.HTML(render_voice_room(DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID, DEFAULT_VOICE_ID))
                with gr.Group(elem_classes=["conversation-panel"]):
                    gr.HTML('<div class="panel-header"><div class="panel-title">实时转写与话术辅助</div><span class="tag">边练边提示</span></div>')
                    transcript = gr.HTML(render_transcript())
                    session_info = gr.HTML(render_session_info())

            with gr.Column(elem_classes=["right-stack"]):
                score = gr.HTML(render_score(DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID))
                coach = gr.HTML(render_coach_feedback(DEFAULT_STAGE_ID, DEFAULT_DIFFICULTY_ID))
                history = gr.HTML(render_history())

        gr.HTML(
            """
<div class="footer-note">
  <span class="kbd">Gradio + FastRTC</span>
  <span class="kbd">实时 ASR</span>
  <span class="kbd">客户画像 Prompt</span>
  <span class="kbd">TTS 音色切换</span>
  <span class="kbd">训练评分与复盘</span>
</div>
"""
        )

        inputs = [stage, difficulty, voice, dashboard_session, selected_case]
        for component in (stage, difficulty, voice):
            component.change(
                update_training_view,
                inputs=inputs,
                outputs=[dashboard_session, selected_case, persona, goal, voice_room, score, coach],
                show_progress="hidden",
            )

        dashboard.load(
            update_training_view,
            inputs=inputs,
            outputs=[dashboard_session, selected_case, persona, goal, voice_room, score, coach],
            show_progress="hidden",
        )

        timer = gr.Timer(value=2.5)
        timer.tick(
            poll_live_view,
            inputs=[stage, difficulty, selected_case],
            outputs=[transcript, score, coach, session_info],
            show_progress="hidden",
        )

    return dashboard


app = FastAPI()


@app.get("/")
async def root():
    return RedirectResponse("/app")


@app.get("/api/config")
async def api_config():
    return {
        "stages": _choice(stage_choices()),
        "difficulties": _choice(difficulty_choices()),
        "voices": _choice(voice_choices()),
        "defaults": {
            "stage_id": DEFAULT_STAGE_ID,
            "difficulty_id": DEFAULT_DIFFICULTY_ID,
            "voice_id": DEFAULT_VOICE_ID,
        },
    }


@app.get("/api/persona")
async def api_persona(stage_id: str = DEFAULT_STAGE_ID, difficulty_id: str = DEFAULT_DIFFICULTY_ID, voice_id: str = DEFAULT_VOICE_ID):
    try:
        stage, customer, difficulty, case = _resolve_case(stage_id, difficulty_id)
        voice = resolve_voice(voice_id)
        assets = summarize_case_assets(case)
        return {
            "name": assets.get("customer_name") or customer.get("name", "?"),
            "role": assets.get("customer_role") or customer.get("role", ""),
            "style": assets.get("communication_style") or customer.get("attitude", {}).get("label", ""),
            "desc": assets.get("scene") or customer.get("attitude", {}).get("style", ""),
            "traits": assets.get("main_concerns") or [],
            "goal": stage.get("training_goal", ""),
            "voice_label": voice.get("label", voice_id),
            "stage": _label(stage),
            "difficulty": _label(difficulty),
            "case_fewshot": assets.get("few_shot_count", 0),
            "case_total": get_case_count(),
            "raw_call_id": assets.get("raw_call_id"),
            "rubric_id": assets.get("rubric_id"),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/training-assets")
async def api_training_assets(stage_id: str = DEFAULT_STAGE_ID, difficulty_id: str = DEFAULT_DIFFICULTY_ID):
    try:
        _, _, _, case = _resolve_case(stage_id, difficulty_id)
        return summarize_case_assets(case)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/debug/status")
async def debug_status():
    return LAST_STATUS


stream.mount(app)
app = gr.mount_gradio_app(app, stream.ui, path="/stream")
app = gr.mount_gradio_app(app, build_dashboard(), path="/app")


if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8520"))
    print(f"\n  http://127.0.0.1:{port}/app  |  http://127.0.0.1:{port}/stream  |  Gradio + FastRTC\n")
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
