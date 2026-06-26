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


def _dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False).strip()


def build_training_prompt(
    stage_id: str | None,
    customer_id: str | None,
    difficulty_id: str | None,
) -> tuple[str, dict[str, Any]]:
    stage, customer, difficulty = resolve_training(stage_id, customer_id, difficulty_id)
    attitude = customer.get("attitude", {})
    blockers = difficulty.get("blockers", [])

    summary = {
        "stage_id": stage["id"],
        "stage": _label(stage),
        "customer_id": customer["id"],
        "customer": customer.get("name", customer["id"]),
        "attitude": attitude.get("label", ""),
        "difficulty_id": difficulty["id"],
        "difficulty": _label(difficulty),
        "blocker_count": len(blockers),
    }

    prompt = "\n\n".join(
        [
            "【系统定位】\n"
            "你是一个智能模拟客户陪练系统，服务于公司内部国际物流销售训练。"
            "你只扮演客户，不扮演销售教练，不解释系统规则，不跳出角色。",
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
            "【回复规则】\n"
            "- 每次回复控制在 1 到 3 句话，适合语音播放。\n"
            "- 你要像真实客户一样表达需求、犹豫、比较、反问或拒绝。\n"
            "- 如果当前难度有卡点，你要在对话中自然制造这些卡点，但不要说出“卡点”二字。\n"
            "- 如果员工表现专业、具体、有推进动作，你可以逐步放松态度并透露更多信息。\n"
            "- 如果员工泛泛介绍、不问需求、回避专业问题，你要降低兴趣或变得更冷淡。",
        ]
    )
    return prompt, summary
