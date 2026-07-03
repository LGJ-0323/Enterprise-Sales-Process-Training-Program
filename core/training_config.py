from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object.")
    return data


@lru_cache
def _load_items(group: str) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in sorted((TRAINING_DIR / group).glob("*.yaml")):
        data = _read_yaml(path)
        item_id = str(data.get("id") or path.stem)
        data["id"] = item_id
        items[item_id] = data
    return items


@lru_cache
def load_voices() -> dict[str, dict[str, Any]]:
    voices: dict[str, dict[str, Any]] = {}
    for path in sorted((TRAINING_DIR / "voices").glob("*.yaml")):
        data = _read_yaml(path)
        entries = data.get("voices", [data])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            voice_id = str(entry.get("id") or entry.get("voice") or path.stem)
            entry["id"] = voice_id
            voices[voice_id] = entry
    return voices


@lru_cache
def load_avatars() -> dict[str, dict[str, Any]]:
    avatars: dict[str, dict[str, Any]] = {}
    for path in sorted((TRAINING_DIR / "avatars").glob("*.yaml")):
        data = _read_yaml(path)
        entries = data.get("avatars", [data])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            avatar_id = str(entry.get("id") or path.stem)
            entry["id"] = avatar_id
            avatars[avatar_id] = entry
    return avatars


def load_stages() -> dict[str, dict[str, Any]]:
    return _load_items("stages")


def load_customers() -> dict[str, dict[str, Any]]:
    return _load_items("customers")


def load_difficulties() -> dict[str, dict[str, Any]]:
    return _load_items("difficulties")


def _get_or_first(items: dict[str, dict[str, Any]], item_id: str | None) -> dict[str, Any]:
    if item_id and item_id in items:
        return items[item_id]
    return next(iter(items.values()))


def _label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("name") or item["id"])


def stage_choices() -> list[tuple[str, str]]:
    stages = sorted(load_stages().values(), key=lambda item: item.get("order", 999))
    return [(_label(stage), stage["id"]) for stage in stages]


def customer_choices() -> list[tuple[str, str]]:
    stages = load_stages()
    choices = [("按阶段自动匹配客户", "auto")]
    for customer in load_customers().values():
        stage = stages.get(customer.get("stage_id"), {})
        stage_label = _label(stage) if stage else str(customer.get("stage_id", ""))
        attitude = customer.get("attitude", {}).get("label", "未设置语气")
        choices.append((f"{customer.get('name', customer['id'])} - {stage_label} / {attitude}", customer["id"]))
    return choices


def difficulty_choices() -> list[tuple[str, str]]:
    difficulties = sorted(load_difficulties().values(), key=lambda item: item.get("level", 999))
    return [(_label(difficulty), difficulty["id"]) for difficulty in difficulties]


def voice_choices() -> list[tuple[str, str]]:
    return [(_label(voice), voice["id"]) for voice in load_voices().values()]


def avatar_choices() -> list[tuple[str, str]]:
    return [(_label(avatar), avatar["id"]) for avatar in load_avatars().values()]


def _select_customer(customer_id: str | None, stage_id: str) -> dict[str, Any]:
    customers = load_customers()
    if customer_id and customer_id != "auto" and customer_id in customers:
        return customers[customer_id]
    for customer in customers.values():
        if customer.get("stage_id") == stage_id:
            return customer
    return next(iter(customers.values()))


def resolve_training(
    stage_id: str | None,
    customer_id: str | None,
    difficulty_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage = _get_or_first(load_stages(), stage_id)
    customer = _select_customer(customer_id, stage["id"])
    difficulty = _get_or_first(load_difficulties(), difficulty_id or customer.get("default_difficulty"))
    return stage, customer, difficulty


def resolve_voice(voice_id: str | None) -> dict[str, Any]:
    return _get_or_first(load_voices(), voice_id)


def resolve_avatar(avatar_id: str | None) -> dict[str, Any]:
    return _get_or_first(load_avatars(), avatar_id)


def resolve_avatar_for_customer(customer_id: str | None, avatar_id: str | None = None) -> dict[str, Any]:
    avatars = load_avatars()
    customers = load_customers()

    if avatar_id and avatar_id != "auto" and avatar_id in avatars:
        avatar = avatars[avatar_id]
        if not customer_id or avatar.get("customer_id") == customer_id:
            return avatar

    if customer_id:
        customer = customers.get(customer_id, {})
        customer_avatar_id = customer.get("avatar_id")
        if customer_avatar_id in avatars:
            return avatars[customer_avatar_id]

        for avatar in avatars.values():
            if avatar.get("customer_id") == customer_id:
                return avatar

    return _get_or_first(avatars, None if avatar_id == "auto" else avatar_id)


def _dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()


@lru_cache(maxsize=16)
def _build_training_prompt_v2_cached(
    stage_id: str,
    difficulty_id: str,
) -> str:
    """缓存版：只返回 prompt 字符串，按 (stage_id, difficulty_id) 缓存。

    首次调用时加载 JSONL + 组装 few-shot prompt（~500ms），
    后续调用直接返回缓存结果（<1ms）。
    """
    stage, customer, difficulty = resolve_training(stage_id, None, difficulty_id)
    stage_label = _label(stage)

    # 尝试从 JSONL 加载
    try:
        from .case_loader import find_case
        from .prompt_assembler import assemble_prompt
        from .training_data_context import build_model_context_for_case
    except ImportError:
        try:
            from case_loader import find_case
            from prompt_assembler import assemble_prompt
            from training_data_context import build_model_context_for_case
        except ImportError:
            find_case = None
            assemble_prompt = None
            build_model_context_for_case = None

    if find_case and assemble_prompt:
        case = find_case(
            training_type=stage_label,
            difficulty=_label(difficulty),
        )
        if case:
            prompt = assemble_prompt(
                case,
                current_state="guarded",
                history=[],
                difficulty=_label(difficulty),
            )
            if build_model_context_for_case:
                data_context, _ = build_model_context_for_case(case)
                if data_context:
                    prompt = f"{prompt}\n\n{data_context}"
            return prompt

    # Fallback: 使用原有 YAML 方式
    prompt, _ = build_training_prompt(stage_id, None, difficulty_id)
    return prompt


def build_training_prompt_v2(
    stage_id: str | None,
    customer_id: str | None,
    difficulty_id: str | None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """V2 版本：优先从 roleplay_cases.jsonl 加载案例，fallback 到 YAML。

    通过 @lru_cache 缓存，首次调用后不再重复加载 JSONL/组装 prompt。
    延迟: 首次 ~500ms，后续 <1ms。
    """
    stage, customer, difficulty = resolve_training(stage_id, customer_id, difficulty_id)

    # 使用缓存获取 prompt
    prompt = _build_training_prompt_v2_cached(
        stage["id"], difficulty["id"],
    )

    # 构建 summary（轻量，不做缓存）
    summary = {
        "stage_id": stage["id"],
        "stage": _label(stage),
        "customer_id": customer["id"],
        "customer": customer.get("name", customer["id"]),
        "difficulty_id": difficulty["id"],
        "difficulty": _label(difficulty),
    }

    try:
        from .case_loader import find_case
        from .training_data_context import build_model_context_for_case
    except ImportError:
        try:
            from case_loader import find_case
            from training_data_context import build_model_context_for_case
        except ImportError:
            find_case = None
            build_model_context_for_case = None

    if find_case and build_model_context_for_case:
        case = find_case(training_type=_label(stage), difficulty=_label(difficulty))
        if case:
            _, data_meta = build_model_context_for_case(case)
            summary.update(
                {
                    "case_id": case.get("case_id"),
                    "source_call_id": case.get("source_call_id"),
                    **data_meta,
                }
            )

    return prompt, summary


def build_training_prompt_from_case(
    case: dict[str, Any],
    current_state: str | None = None,
    history: list[dict[str, Any]] | None = None,
    stage_id: str | None = None,
    difficulty_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a non-cached prompt for a fixed JSONL case and runtime state."""
    try:
        from .prompt_assembler import assemble_prompt
        from .training_data_context import build_model_context_for_case
    except ImportError:
        from prompt_assembler import assemble_prompt
        from training_data_context import build_model_context_for_case

    state_machine = case.get("state_machine", {})
    state = current_state or state_machine.get("initial_state") or "guarded"
    difficulty = str(case.get("difficulty") or "")
    if difficulty_id:
        difficulty_item = load_difficulties().get(difficulty_id)
        if difficulty_item:
            difficulty = str(difficulty_item.get("id") or difficulty_item.get("label") or difficulty)

    prompt = assemble_prompt(case, current_state=state, history=history or [], difficulty=difficulty)
    data_context, data_meta = build_model_context_for_case(case)
    if data_context:
        prompt = f"{prompt}\n\n{data_context}"

    role = case.get("customer_role_card", {})
    stage = load_stages().get(stage_id or "")
    summary = {
        "stage_id": stage_id or "",
        "stage": _label(stage) if stage else str(case.get("training_type", "")),
        "customer_id": case.get("case_id", ""),
        "customer": role.get("name") or "客户",
        "difficulty_id": difficulty_id or str(case.get("difficulty", "")),
        "difficulty": str(case.get("difficulty", "")),
        "case_id": case.get("case_id"),
        "source_call_id": case.get("source_call_id"),
        "current_state": state,
        "training_type": case.get("training_type"),
        **data_meta,
    }
    return prompt, summary


def build_training_prompt(
    stage_id: str | None,
    customer_id: str | None,
    difficulty_id: str | None,
) -> tuple[str, dict[str, Any]]:
    stage, customer, difficulty = resolve_training(stage_id, customer_id, difficulty_id)
    attitude = customer.get("attitude", {})
    state_curve = customer.get("state_curve", {})
    professional_probes = customer.get("professional_probes", [])
    blockers = difficulty.get("blockers", [])

    summary = {
        "stage_id": stage["id"],
        "stage": _label(stage),
        "customer_id": customer["id"],
        "customer": customer.get("name", customer["id"]),
        "avatar_id": customer.get("avatar_id", ""),
        "attitude": attitude.get("label", ""),
        "difficulty_id": difficulty["id"],
        "difficulty": _label(difficulty),
        "blocker_count": len(blockers),
    }

    prompt = "\n\n".join(
        [
            "【系统定位】\n"
            "你正在扮演真实企业客户，服务于公司内部国际物流销售训练。"
            "对话中的员工/用户是雄达物流销售，正在争取让客户使用雄达物流的国际物流服务。"
            "你只扮演被拜访企业里的客户人员，不扮演雄达物流销售，不替雄达物流推销服务，"
            "不扮演销售教练，不解释系统规则，不跳出角色。",
            "【角色边界】\n"
            "- 员工/用户身份：雄达物流销售。\n"
            "- 你的身份：企业客户方人员，代表自己的企业评估物流供应商。\n"
            "- 你的沟通目标：根据本企业发货需求、成本、时效、风险和信任顾虑，判断是否继续了解雄达物流。\n"
            "- 你不能说成自己是雄达物流的人，也不能主动向员工推销物流服务。\n"
            "- 你说“我们”时，只能指你的企业、工厂、贸易公司、平台或现有合作货代。",
            "【反向错误示例】\n"
            "- 错误：你好，我们是雄达物流，想了解一下你们最近有没有发货需求？\n"
            "- 错误：我们这边雄达物流价格比较低，可以给你们报价。\n"
            "- 正确：嗯，你们是做哪条线的？我现在有合作货代，你先说重点。\n"
            "- 正确：价格低我会关注，但你们费用边界和旺季涨价怎么保证？",
            "【行业范围】\n"
            "国际物流业务，包括海运、空运、询价、报价、订舱、舱位、时效、目的港费用、报关、查验、异常处理、旺季风险和售后跟进。",
            "【当前训练阶段】\n"
            + _dump(
                {
                    "阶段": _label(stage),
                    "训练目标": stage.get("training_goal"),
                    "客户默认反应": stage.get("customer_behavior"),
                    "员工应训练能力": stage.get("employee_skills"),
                }
            ),
            "【当前客户画像】\n"
            + _dump(
                {
                    "姓名": customer.get("name"),
                    "所属阶段": stage.get("label"),
                    "公司类型": customer.get("company_type"),
                    "职位": customer.get("role"),
                    "运输需求": customer.get("logistics_needs"),
                    "核心痛点": customer.get("pain_points"),
                    "默认难度": customer.get("default_difficulty"),
                }
            ),
            "【客户状态曲线】\n"
            + _dump(
                {
                    "初始状态": state_curve.get("initial_state"),
                    "可选状态": state_curve.get("states", {}),
                    "状态转移线索": state_curve.get("transitions", []),
                }
            ),
            "【客户专业问题库】\n"
            + _dump(
                {
                    "可选专业追问": professional_probes,
                }
            ),
            "【客户语气态度】\n"
            + _dump(
                {
                    "语气": attitude.get("label"),
                    "表达方式": attitude.get("style"),
                    "沟通原则": attitude.get("rules"),
                }
            ),
            "【难度与卡点】\n"
            + _dump(
                {
                    "难度": _label(difficulty),
                    "卡点数量": len(blockers),
                    "卡点策略": difficulty.get("policy"),
                    "本轮固定卡点": blockers,
                }
            ),
            "【本轮客户状态判定】\n"
            "- 回复前先结合员工当前发言、客户状态曲线、当前阶段、难度卡点和最近 10 轮会话记忆，判断客户此刻状态。\n"
            "- 你只能在状态曲线中选择最贴近的状态；如果没有明显线索，使用初始状态。\n"
            "- 员工越具体地问到航线、货量、品名、时效、费用、当前供应商问题，客户越可能从防备/平淡转向好奇或愿意继续。\n"
            "- 员工泛泛介绍公司、说太长、只讲低价或回避专业问题，客户要降温、质疑或施压。\n"
            "- 员工提出报价、方案、试单或服务承诺时，客户应进入追问风险、费用边界、责任边界的状态。\n"
            "- 不要输出状态名，只输出该状态下客户会自然说的话。",
            "【专业问题使用规则】\n"
            "- 每轮最多选择 1 个与当前员工发言最相关的专业问题，不能把问题库连续罗列出来。\n"
            "- 专业问题必须贴合客户画像中的公司类型、岗位、货物、航线、运输方式和痛点。\n"
            "- 如果员工没有触及具体业务，先用短句降温或要求说重点，不急着抛复杂专业问题。\n"
            "- 如果员工给出具体方案或承诺，优先追问费用构成、报价有效期、时效、异常处理、责任人或旺季保障。\n"
            "- 如果员工回答专业、具体、可执行，可以少量释放客户需求信息，但不要一次性把全部背景说完。",
            "【回复规则】\n"
            "- 每次回复控制在 1 到 3 句话，适合语音播放。\n"
            "- 直接输出客户会说的话，不要添加姓名、角色或说话人前缀；禁止使用“王女士：”“客户：”“外贸主管：”这类格式。\n"
            "- 用户消息中的“我们”通常指雄达物流销售团队，不是你；你回复里的“我们”只能指客户自己的企业。\n"
            "- 你要像真实客户一样表达需求、犹豫、比较、反问或拒绝。\n"
            "- 如果当前难度有卡点，你要在对话中自然制造这些卡点，但不要说出“卡点”二字。\n"
            "- 如果员工表现专业、具体、有推进动作，你可以逐步放松态度并透露更多信息。\n"
            "- 如果员工泛泛介绍、不问需求、回避专业问题，你要降低兴趣或变得更冷淡。",
        ]
    )
    return prompt, summary
