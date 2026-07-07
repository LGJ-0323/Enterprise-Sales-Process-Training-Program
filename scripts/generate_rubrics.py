"""
generate_rubrics.py — 根据 roleplay_cases.jsonl 批量生成 evaluation_rubrics.jsonl

策略：
1. 不再手写 237 条专属 rubric，而是定义 8 个场景族模板
2. 读 roleplay_cases.jsonl，按 training_type 匹配模板 → 自动实例化
3. case 的 failure_conditions / training_goals 直接复制进 rubric
4. few_shot_examples → 自动生成为 ideal_sales_flow
5. difficulty 字段 → 写入 difficulty_variants.pass_threshold
6. 输出覆盖 docs/evaluation_rubrics.jsonl

用法：
    python scripts/generate_rubrics.py
"""

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass


# ═══════════════════════════════════════════════════════
# 场景族模板定义
# ═══════════════════════════════════════════════════════

# training_type → scenaroio_family 映射表
TYPE_TO_FAMILY = OrderedDict([
    ("新客户开发", "新客户开发/陌call"),
    ("陌call", "新客户开发/陌call"),
    ("prospecting_outbound", "新客户开发/陌call"),
    ("discovery_call", "新客户开发/陌call"),
    ("报价后回访", "报价后回访"),
    ("需求确认回访", "报价后回访"),
    ("货物异常处理", "货物异常/问题处理回访"),
    ("问题处理回访", "货物异常/问题处理回访"),
    ("scenario_recovery", "货物异常/问题处理回访"),
    ("obstacle_recognition_and_recovery", "货物异常/问题处理回访"),
    ("informational_service_handling", "货物异常/问题处理回访"),
    ("老客户维护", "老客户维护"),
    ("旺季舱位提醒", "旺季舱位提醒"),
    ("流失客户挽回", "流失客户挽回"),
    ("投诉处理回访", "投诉处理回访"),
    ("合作续约沟通", "合作续约沟通"),
    ("voice_mail_outreach", "新客户开发/陌call"),
])

# 7 要素维度（所有场景共用，仅权重不同）
SEVEN_ELEMENTS_DIM = {
    "dimension": "需求挖掘",
    "description": "评估销售是否系统性地挖掘了国际物流客户的7项核心业务信息",
    "seven_elements": {
        "1_市场区域": {"element": "客户市场是否做美加？", "weight": 3, "excellent": "主动问出目标市场+追问美/加/欧/东南亚具体区域", "pass": "问到了出口目的地", "fail": "全程未提及出口市场"},
        "2_外贸模式": {"element": "外贸模式：传统外贸还是跨境电商？", "weight": 3, "excellent": "问清传统/电商模式+追问平台(亚马逊/FBA/Temu等)", "pass": "问到了业务模式", "fail": "未了解业务模式"},
        "3_出口产品": {"element": "出口产品是什么？", "weight": 3, "excellent": "问出具体品名+判断普货/敏感货/危险品", "pass": "大致了解产品类型", "fail": "完全不知道客户卖什么"},
        "4_运输方式": {"element": "运输方式：海运还是空运？", "weight": 3, "excellent": "确认海空比例+追问时效要求+是否考虑联运", "pass": "问到了常用运输方式", "fail": "没问运输方式就报价"},
        "5_出货渠道": {"element": "常出货的港口/渠道？", "weight": 2, "excellent": "问出起运港+目的港+结合公司优势港口给建议", "pass": "问到了起运城市或地区", "fail": "未涉及出货渠道"},
        "6_货量频次": {"element": "货量频次：多久出一次，一次出多少？", "weight": 3, "excellent": "问出月/周出货量+单次货量(CBM/KG)+判断体量", "pass": "了解了出货频次", "fail": "没问货量就报价"},
        "7_出货计划": {"element": "最近有没有货要出？", "weight": 3, "excellent": "问出具体出货时间+货量+顺势推进报价/试单", "pass": "询问了近期是否有出货", "fail": "全程未推进到具体出货时间"},
    },
    "excellent": "覆盖>=6项且>=4项达到excellent标准",
    "pass": "覆盖>=4项，客户画像基本清晰",
    "fail": "覆盖<3项，客户画像模糊",
    "scoring_guide": "单项excellent=3分 pass=2分 fail=0分 满分21分折算",
}

SEVEN_ELEMENTS_LIGHT = {
    "dimension": "需求诊断",
    "description": "评估销售是否准确识别了客户的核心问题和当前状态",
    "excellent": "精准定位客户痛点，能关联具体业务场景给出针对性回应",
    "pass": "基本理解客户当前状态和需求",
    "fail": "对客户问题的判断出现明显偏差",
}


# 模板 1：新客户开发/陌call
FAMILY_NEW_CUSTOMER = {
    "scenario_family": "新客户开发/陌call",
    "scoring_dimensions": [
        {"dimension": "开场破冰", "score": 10,
         "excellent": "自然说明来意和来源，快速建立信任",
         "pass": "说清楚了自己是谁和为什么联系", "fail": "开场生硬，客户有防备心"},
        {"dimension": "需求挖掘", "score": 25, **SEVEN_ELEMENTS_DIM},
        {"dimension": "价值呈现", "score": 20,
         "excellent": "针对客户痛点精准呈现公司解决方案",
         "pass": "有介绍公司优势", "fail": "自说自话，与客户需求脱节"},
        {"dimension": "异议处理", "score": 15,
         "excellent": "对'已有固定货代'等常见异议有成熟应对话术",
         "pass": "能回应客户异议", "fail": "遇到异议就退缩或强推"},
        {"dimension": "推进下一步", "score": 20,
         "excellent": "明确下一步（加微信/发报价/约面谈），降低决策门槛",
         "pass": "有后续跟进计划", "fail": "没有明确下一步"},
        {"dimension": "专业形象", "score": 10,
         "excellent": "展现行业知识和专业度，让客户觉得可靠",
         "pass": "基本专业", "fail": "表现不专业"},
    ],
    "must_do": ["清晰说明来意和身份", "确认客户当前合作状态", "至少完成 4 项 7 要素信息挖掘", "明确下一步动作"],
    "critical_mistakes": ["承诺无法兑现的价格或服务", "攻击同行或否定客户判断", "没问产品和货量就盲目报价"],
}

# 模板 2：报价后回访
FAMILY_QUOTE_FOLLOWUP = {
    "scenario_family": "报价后回访",
    "scoring_dimensions": [
        {"dimension": "开场与来意确认", "score": 10,
         "excellent": "自然说明身份和回访目的，客户不会觉得突兀",
         "pass": "能说明自己是谁及回访原因", "fail": "开场含糊，客户不知道打电话做什么"},
        {"dimension": "报价进展确认", "score": 15,
         "excellent": "问清客户是否已报给终端客户、当前反馈状态",
         "pass": "问到客户对价格的基本反馈", "fail": "没有确认报价进展，直接进入推销"},
        {"dimension": "价格异议处理", "score": 15,
         "excellent": "理解客户比价心理，用ALL IN费用对比法，不攻击同行",
         "pass": "能解释价格高的原因", "fail": "回避价格问题或攻击同行"},
        {"dimension": "需求挖掘", "score": 15, **SEVEN_ELEMENTS_DIM},
        {"dimension": "价值匹配", "score": 15,
         "excellent": "将公司优势精准匹配客户需求点",
         "pass": "有介绍公司优势", "fail": "泛泛介绍与客户无关"},
        {"dimension": "推进下一步", "score": 15,
         "excellent": "明确下一步动作（发资料/锁舱/试单），有具体时间节点",
         "pass": "表达了后续跟进意愿", "fail": "通话结束没有任何后续动作"},
        {"dimension": "信息补全", "score": 5,
         "excellent": "拿到客户称呼、地址、出货量等关键信息",
         "pass": "至少补充了一项客户信息", "fail": "没有补充任何客户信息"},
        {"dimension": "专业度体现", "score": 10,
         "excellent": "展现行业知识（旺季、清关、费用结构），让客户感到专业",
         "pass": "提到了部分行业常识", "fail": "表现出不专业或给错误信息"},
    ],
    "must_do": ["清晰说明来意", "确认报价当前状态", "回答客户对价格的疑问", "明确下一步动作"],
    "critical_mistakes": ["承诺无法兑现的价格或服务", "攻击同行", "没问货量就报价"],
}

# 模板 3：货物异常/问题处理回访
FAMILY_ISSUE_HANDLING = {
    "scenario_family": "货物异常/问题处理回访",
    "scoring_dimensions": [
        {"dimension": "沟通开场", "score": 15,
         "excellent": "开场自然，说明来意清晰", "pass": "能说清楚来意", "fail": "开场生硬或无目的"},
        {"dimension": "问题理解", "score": 20,
         "excellent": "准确理解客户问题/需求的核心",
         "pass": "基本理解客户意图", "fail": "误解或忽视客户核心问题"},
        {"dimension": "解决方案", "score": 25,
         "excellent": "提供清晰、可行、有诚意的解决方案",
         "pass": "提供了基本解决方案", "fail": "没有实质性解决方案"},
        {"dimension": "情绪处理", "score": 15,
         "excellent": "能识别客户情绪并恰当回应",
         "pass": "没有激化客户负面情绪", "fail": "激怒客户或无视客户情绪"},
        {"dimension": "推进闭环", "score": 15,
         "excellent": "有明确下一步和时间节点",
         "pass": "表达了后续跟进意愿", "fail": "问题悬而未决"},
        {"dimension": "专业度", "score": 10,
         "excellent": "展现行业知识和专业态度", "pass": "基本专业", "fail": "表现不专业"},
        {"dimension": "需求诊断", "score": 20, **SEVEN_ELEMENTS_LIGHT},
    ],
    "must_do": ["主动告知问题原因", "给出明确的处理时间预期", "提供补偿或备选方案", "保持主动跟进姿态"],
    "critical_mistakes": ["隐瞒问题严重性", "推卸责任", "承诺无法兑现的时效", "未主动跟进造成空窗期"],
}

# 模板 4：老客户维护
FAMILY_EXISTING_CUSTOMER = {
    "scenario_family": "老客户维护",
    "scoring_dimensions": [
        {"dimension": "关系维护", "score": 20,
         "excellent": "真诚关心客户近况，不只谈业务",
         "pass": "有问候和关心", "fail": "上来就谈业务，显得功利"},
        {"dimension": "服务反馈收集", "score": 20,
         "excellent": "主动询问服务问题和改进建议，认真记录",
         "pass": "问到了客户对服务的评价", "fail": "没有询问客户反馈"},
        {"dimension": "增值信息提供", "score": 15,
         "excellent": "提供有价值的行业信息（旺季提醒/渠道变化等）",
         "pass": "提供了一些有用信息", "fail": "没有提供增值内容"},
        {"dimension": "业务拓展", "score": 15,
         "excellent": "在合适的时机自然地介绍新服务或升级方案",
         "pass": "提到了业务拓展方向", "fail": "强行推销让客户反感"},
        {"dimension": "转介绍引导", "score": 10,
         "excellent": "自然引导客户做转介绍",
         "pass": "有提及转介绍", "fail": "完全没有拓展人脉意识"},
        {"dimension": "结束跟进", "score": 10,
         "excellent": "有明确的后续跟进安排",
         "pass": "表达了保持联系意愿", "fail": "通话草草结束"},
        {"dimension": "需求挖掘", "score": 10, **SEVEN_ELEMENTS_DIM},
    ],
    "must_do": ["先问候再谈业务", "收集至少一条服务反馈", "提供一条增值信息", "铺垫下一次触达"],
    "critical_mistakes": ["客户已表示满意后仍强行推销", "忽视客户提出的改进建议", "通话变成纯闲聊无商业价值"],
}

# 模板 5：旺季舱位提醒
FAMILY_PEAK_SEASON = {
    "scenario_family": "旺季舱位提醒",
    "scoring_dimensions": [
        {"dimension": "价值信息传递", "score": 15,
         "excellent": "以提供有价值信息为由联系，非直接推销",
         "pass": "说清楚了舱位紧张的情况", "fail": "信息空洞，客户感觉被推销"},
        {"dimension": "需求确认", "score": 20,
         "excellent": "准确了解客户近期出货计划和时间窗口",
         "pass": "问到了客户是否有出货计划", "fail": "未了解客户出货时间"},
        {"dimension": "方案匹配", "score": 25,
         "excellent": "针对客户出货计划给出锁舱/报价方案",
         "pass": "能给出基本方案", "fail": "无针对性建议"},
        {"dimension": "推进闭环", "score": 20,
         "excellent": "明确下一步（发报价/锁舱/发资料），有时间节点",
         "pass": "有后续跟进意愿", "fail": "通话结束无后续动作"},
        {"dimension": "专业度", "score": 20,
         "excellent": "展现对市场行情的深入了解，让客户信任",
         "pass": "提到了基本的行业信息", "fail": "信息有误或不专业"},
    ],
    "must_do": ["以行业信息为切入点", "了解客户近期出货计划", "给出具体建议或方案"],
    "critical_mistakes": ["纯推销无信息价值", "夸大舱位紧张制造焦虑", "承诺无法保证的舱位"],
}

# 模板 6：流失客户挽回
FAMILY_LOST_CUSTOMER = {
    "scenario_family": "流失客户挽回",
    "scoring_dimensions": [
        {"dimension": "关系重建", "score": 20,
         "excellent": "真诚关心客户近况，不急于推销，先重建对话基础",
         "pass": "有基本问候和关系铺垫", "fail": "上来就直接推销"},
        {"dimension": "问题理解", "score": 20,
         "excellent": "准确理解客户流失原因和当前合作情况",
         "pass": "问到客户当前服务情况", "fail": "未了解流失原因就推销"},
        {"dimension": "挽回方案", "score": 25,
         "excellent": "提出有针对性的改进方案或补偿，体现诚意",
         "pass": "表达了希望重新合作的意愿", "fail": "没有实质性挽回措施"},
        {"dimension": "情绪处理", "score": 15,
         "excellent": "对客户过去的不满表达理解，不辩解",
         "pass": "没有激化负面情绪", "fail": "推卸责任或激怒客户"},
        {"dimension": "推进闭环", "score": 20,
         "excellent": "明确下一步（发报价/约拜访），但有分寸不强迫",
         "pass": "表达了后续联系意愿", "fail": "强行推进让客户反感"},
    ],
    "must_do": ["先理解流失原因再谈方案", "展现改进诚意而非辩解", "用低姿态方式留出窗口"],
    "critical_mistakes": ["客户明确拒绝后仍继续推销", "推卸责任", "攻击当前合作货代", "通话超过5分钟未给客户说话机会"],
}

# 模板 7：投诉处理回访
FAMILY_COMPLAINT = {
    "scenario_family": "投诉处理回访",
    "scoring_dimensions": [
        {"dimension": "沟通开场", "score": 15,
         "excellent": "开场主动道歉并说明整改措施，客户感受到重视",
         "pass": "能说清楚来意", "fail": "开场无道歉或回避问题"},
        {"dimension": "问题理解", "score": 20,
         "excellent": "准确理解投诉原因和客户的损失",
         "pass": "基本理解客户投诉内容", "fail": "回避或低估问题严重性"},
        {"dimension": "解决方案", "score": 25,
         "excellent": "给出具体的补偿方案+改进承诺+可验证措施",
         "pass": "提供了基本补偿或道歉", "fail": "没有实质性解决方案"},
        {"dimension": "补偿方案", "score": 15,
         "excellent": "补偿有具体价值（减免费用/赠服务），超出客户预期",
         "pass": "表达了歉意和补偿意愿", "fail": "仅口头道歉无实际行动"},
        {"dimension": "推进闭环", "score": 15,
         "excellent": "明确下一步跟进节点和改进验证方式",
         "pass": "表达了后续跟进意愿", "fail": "问题悬而未决"},
        {"dimension": "专业度", "score": 10,
         "excellent": "展现诚恳态度和专业处理能力",
         "pass": "基本专业", "fail": "推卸责任或态度不端正"},
    ],
    "must_do": ["开场道歉", "承认问题并说明原因", "给出具体补偿方案", "承诺改进措施"],
    "critical_mistakes": ["推卸责任", "轻视客户损失", "敷衍式道歉无实际行动", "激化客户情绪"],
}

# 模板 8：合作续约沟通
FAMILY_RENEWAL = {
    "scenario_family": "合作续约沟通",
    "scoring_dimensions": [
        {"dimension": "数据复盘", "score": 20,
         "excellent": "用具体数据回顾合作成果（票数/准点率/问题处理）",
         "pass": "有基本的年度回顾", "fail": "无数据回顾直接谈续约"},
        {"dimension": "服务反馈收集", "score": 20,
         "excellent": "主动收集服务评价和改进建议",
         "pass": "问到了客户对服务的满意度", "fail": "未关心客户反馈"},
        {"dimension": "增值方案", "score": 20,
         "excellent": "给出续约专属优惠或升级方案",
         "pass": "介绍了续约的基本条件", "fail": "仅通知续约无额外价值"},
        {"dimension": "推进闭环", "score": 25,
         "excellent": "给出续约方案+明确回复时间节点",
         "pass": "表达了续约意愿", "fail": "通话结束无明确结论"},
        {"dimension": "专业度", "score": 15,
         "excellent": "展现对客户业务的关注和长期合作诚意",
         "pass": "基本专业", "fail": "表现不专业或过度推销"},
    ],
    "must_do": ["回顾合作数据", "问询服务满意度", "给出续约方案", "约定回复时间"],
    "critical_mistakes": ["未提及历史服务表现直接谈续约", "强势逼迫客户立即决策", "忽视客户提出的改进诉求"],
}

# 优先匹配顺序（精确 training_type 匹配优先）
FAMILY_TEMPLATES = {
    "新客户开发/陌call": FAMILY_NEW_CUSTOMER,
    "报价后回访": FAMILY_QUOTE_FOLLOWUP,
    "货物异常/问题处理回访": FAMILY_ISSUE_HANDLING,
    "老客户维护": FAMILY_EXISTING_CUSTOMER,
    "旺季舱位提醒": FAMILY_PEAK_SEASON,
    "流失客户挽回": FAMILY_LOST_CUSTOMER,
    "投诉处理回访": FAMILY_COMPLAINT,
    "合作续约沟通": FAMILY_RENEWAL,
}


DIFFICULTY_THRESHOLDS = {
    "easy": {"pass_threshold": 45, "coach_focus": "基础流程与产品信息完整度"},
    "medium": {"pass_threshold": 60, "coach_focus": "异议处理深度与服务差异化呈现"},
    "hard": {"pass_threshold": 75, "coach_focus": "价格防御、价值谈判与综合压力应对"},
    "expert": {"pass_threshold": 85, "coach_focus": "高阶客户心理学与策略性关系管理"},
}


# ═══════════════════════════════════════════════════════
# 实例化函数
# ═══════════════════════════════════════════════════════

def get_scenario_family(training_type: str) -> str:
    """根据 training_type 获取场景族名称。"""
    return TYPE_TO_FAMILY.get(training_type, "新客户开发/陌call")


def build_ideal_sales_flow(case: dict) -> list[str]:
    """从 case 的 few_shot_examples 提取 ideal_sales_flow。"""
    examples = case.get("few_shot_examples") or []
    flow = []
    for ex in examples:
        text = ex.get("sales_input", "")
        if len(text) > 20:
            flow.append(text)
    if not flow:
        flow.append("开场：说明身份和来意")
        flow.append("推进：确认客户当前状态并提供价值")
        flow.append("闭环：明确下一步动作")
    return flow[:5]


def build_bonus_behaviors(case: dict, family: str) -> list[str]:
    """为案例生成特定的 bonus_behaviors。"""
    bonuses = []
    scene = (case.get("scene") or "").lower()
    training_type = (case.get("training_type") or "")

    if "美" in scene or "美线" in scene or "美西" in scene or "美东" in scene:
        bonuses.append("通过引导式提问获取美加市场信息(客户未主动提及时)")
    if "产品" in scene or "家居" in scene or "小商品" in scene:
        bonuses.append("根据产品类型判断是否普货/敏感货并提示清关要求")
    if "货量" in scene or "方" in scene or "公斤" in scene:
        bonuses.append("问了货量后主动建议拼箱/整柜方案而非报一口价")
    if "急" in scene or "下周" in scene or "本月" in scene:
        bonuses.append("确认出货计划后立即推进具体报价时间节点")

    # 缺省的添加兜底
    if not bonuses:
        if "新客户" in training_type or "陌call" in training_type:
            bonuses.append("开场即说明来电来源（如展会/官网/朋友介绍）")
        elif "回访" in training_type:
            bonuses.append("引用上一次沟通的细节（如日期/具体报价）建立延续感")

    return bonuses[:5]


def build_difficulty_variants(case: dict) -> dict:
    """根据 case 的 difficulty 字段生成 difficulty_variants。"""
    difficulty = str(case.get("difficulty") or "medium").strip()
    variants = case.get("difficulty_variants")

    if variants and isinstance(variants, dict):
        # 已有 difficulty_variants，补充 pass_threshold 和 coach_focus
        for d, v in variants.items():
            if d in DIFFICULTY_THRESHOLDS:
                v["pass_threshold"] = DIFFICULTY_THRESHOLDS[d]["pass_threshold"]
                v.setdefault(
                    "coach_focus", DIFFICULTY_THRESHOLDS[d]["coach_focus"])
        # 确保当前难度也在里面
        if difficulty not in variants:
            variants[difficulty] = DIFFICULTY_THRESHOLDS.get(
                difficulty, DIFFICULTY_THRESHOLDS["medium"])
        return variants

    return {
        difficulty: DIFFICULTY_THRESHOLDS.get(difficulty, DIFFICULTY_THRESHOLDS["medium"]),
    }


def generate_rubric(case: dict, template: dict) -> dict:
    """将模板实例化为一条 case-specific rubric。"""
    family_name = template["scenario_family"]
    case_id = str(case.get("case_id") or "")
    training_type = str(case.get("training_type") or "")
    difficulty = str(case.get("difficulty") or "medium").strip()

    rubric = {
        "rubric_id": f"rubric_{case_id.replace('case_', '')}" if case_id else f"rubric_{training_type}",
        "case_id": case_id,
        "source_call_id": case.get("source_call_id"),
        "schema_version": "2.0",
        "training_type": training_type,
        "total_score": 100,
        "scenario_family": family_name,
        "scoring_dimensions": template["scoring_dimensions"],
        "must_do": template["must_do"],
        "critical_mistakes": template["critical_mistakes"],
        "bonus_behaviors": build_bonus_behaviors(case, family_name),
        "ideal_sales_flow": build_ideal_sales_flow(case),
        "difficulty_variants": build_difficulty_variants(case),
        "coach_feedback_template": {
            "summary": "本轮你在【{strong_points}】上做得不错，但【{main_gap}】还需要加强。",
            "score_explanation": "该难度下及格线为 {pass_threshold}/{total} 分，你的分数 {score}/{total}。",
            "next_training_advice": "下一轮重点练：{focus_area}。",
        },
    }

    # 复制 case 中的可选字段（如果存在）
    if case.get("failure_conditions"):
        rubric["failure_conditions"] = case["failure_conditions"]
    if case.get("training_goals"):
        rubric["training_goals"] = case["training_goals"]
    if case.get("customer_behavior_rules"):
        rubric["customer_behavior_rules"] = case["customer_behavior_rules"]

    return rubric


def main():
    cases_path = PROJECT_DIR / "docs" / "roleplay_cases.jsonl"
    output_path = PROJECT_DIR / "docs" / "evaluation_rubrics.jsonl"

    if not cases_path.exists():
        print(f"[错误] 未找到: {cases_path}")
        sys.exit(1)

    # 读所有 case
    cases = []
    with open(cases_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 跳过来解析的行: {e}")

    print(f"[读取] roleplay_cases.jsonl: {len(cases)} 条")

    # 统计场景族分布
    family_counts = {}
    for case in cases:
        tt = str(case.get("training_type") or "")
        family = get_scenario_family(tt)
        family_counts[family] = family_counts.get(family, 0) + 1
    print(f"[场景族分布]:")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {family}: {count}")

    # 生成 rubric
    rubrics = []
    missing_templates = set()
    for case in cases:
        training_type = str(case.get("training_type") or "")
        family = get_scenario_family(training_type)
        template = FAMILY_TEMPLATES.get(family)

        if not template:
            missing_templates.add(training_type)
            # fallback 到默认模板
            template = FAMILY_TEMPLATES["新客户开发/陌call"]

        rubrics.append(generate_rubric(case, template))

    if missing_templates:
        print(f"[警告] 以下 training_type 无精确模板，已回退默认:")
        for t in sorted(missing_templates):
            print(f"  - {t}")

    # 写入
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rubric in rubrics:
            f.write(json.dumps(rubric, ensure_ascii=False) + "\n")

    print(f"[写入] evaluation_rubrics.jsonl: {len(rubrics)} 条 (schema v2.0)")
    print("[完成]")


if __name__ == "__main__":
    main()
