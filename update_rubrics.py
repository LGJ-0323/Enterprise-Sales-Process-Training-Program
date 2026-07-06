import json
import sys

RUBRICS_PATH = sys.argv[1] if len(sys.argv) > 1 else "docs/evaluation_rubrics.jsonl"

with open(RUBRICS_PATH, encoding="utf-8") as f:
    rubrics = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(rubrics)} entries")

# 7 elements for international logistics need discovery
NEED_PROBE = {
    "dimension": "需求挖掘",
    "score": 20,
    "description": "评估销售是否系统性地挖掘了国际物流客户的7项核心业务信息",
    "seven_elements": {
        "1_市场区域": {
            "element": "客户市场是否做美加？",
            "weight": 3,
            "why_matters": "美加线是核心利润线，确认市场决定方案匹配度",
            "excellent": "主动问出目标市场+追问美/加/欧/东南亚具体区域",
            "pass": "问到了出口目的地",
            "fail": "全程未提及出口市场"
        },
        "2_外贸模式": {
            "element": "外贸模式：传统外贸还是跨境电商？",
            "weight": 3,
            "why_matters": "B2B整柜和B2C小包物流需求完全不同",
            "excellent": "问清传统/电商模式+追问平台(亚马逊/FBA/Temu等)",
            "pass": "问到了业务模式",
            "fail": "未了解业务模式"
        },
        "3_出口产品": {
            "element": "出口产品是什么？",
            "weight": 3,
            "why_matters": "产品决定HS编码/是否敏感货/是否需FDA等认证",
            "excellent": "问出具体品名+判断普货/敏感货/危险品",
            "pass": "大致了解产品类型",
            "fail": "完全不知道客户卖什么"
        },
        "4_运输方式": {
            "element": "运输方式：海运还是空运？",
            "weight": 3,
            "why_matters": "海运/空运产品线和报价逻辑完全不同",
            "excellent": "确认海空比例+追问时效要求+是否考虑联运",
            "pass": "问到了常用运输方式",
            "fail": "没问运输方式就报价"
        },
        "5_出货渠道": {
            "element": "常出货的港口/渠道？",
            "weight": 2,
            "why_matters": "不同港口舱位紧张度和服务能力差异大",
            "excellent": "问出起运港+目的港+结合公司优势港口给建议",
            "pass": "问到了起运城市或地区",
            "fail": "未涉及出货渠道"
        },
        "6_货量频次": {
            "element": "货量频次：多久出一次，一次出多少？",
            "weight": 3,
            "why_matters": "货量决定拼箱/整柜/合约价和客户等级",
            "excellent": "问出月/周出货量+单次货量(CBM/KG)+判断体量",
            "pass": "了解了出货频次",
            "fail": "没问货量就报价"
        },
        "7_出货计划": {
            "element": "最近有没有货要出？",
            "weight": 3,
            "why_matters": "直接决定是否可立即转化，区分闲聊和有效销售",
            "excellent": "问出具体出货时间+货量+顺势推进报价/试单",
            "pass": "询问了近期是否有出货",
            "fail": "全程未推进到具体出货时间"
        }
    },
    "excellent": "覆盖≥6项且≥4项达到excellent标准，能给出针对性方案",
    "pass": "覆盖≥4项，客户画像基本清晰",
    "fail": "覆盖<3项，客户画像模糊，无法给有效方案",
    "scoring_guide": "单项excellent=3分 pass=2分 fail=0分 满分21分折算20分制"
}

MUST_DO = [
    "清晰说明来意和身份",
    "确认客户当前状态/进展",
    "系统挖掘客户7项核心信息:市场/模式/产品/运输方式/港口/货量/出货计划",
    "针对客户需求提供匹配方案",
    "明确下一步行动计划"
]

BONUS = [
    "通过引导式提问获取美加市场信息(客户未主动提及时)",
    "根据产品类型判断是否普货/敏感货并提示清关要求",
    "问了货量后主动建议拼箱/整柜方案而非报一口价",
    "确认出货计划后立即推进具体报价时间节点",
    "将7项信息串联成完整客户画像并复述确认"
]

CRITICAL = [
    "承诺无法兑现的价格或服务",
    "攻击同行或否定客户判断",
    "客户明确拒绝后继续强行推销",
    "没问产品和货量就盲目报价",
    "把传统外贸客户的货推荐到电商小包渠道(或反之)",
    "提供错误的行业信息"
]

for r in rubrics:
    dims = r.get('scoring_dimensions', [])
    found = False
    for i, d in enumerate(dims):
        if d.get('dimension', '') in ('需求挖掘', '需求探索'):
            new_d = json.loads(json.dumps(NEED_PROBE))
            new_d['score'] = d.get('score', 20)
            dims[i] = new_d
            found = True
            break
    if not found:
        dims.append(json.loads(json.dumps(NEED_PROBE)))
    r['scoring_dimensions'] = dims
    r['must_do'] = MUST_DO
    r['bonus_behaviors'] = BONUS
    r['critical_mistakes'] = CRITICAL

    has_val = any(d.get('dimension') in ('价值匹配','价值呈现') for d in dims)
    if not has_val:
        dims.append({
            "dimension": "价值匹配", "score": 15,
            "description": "将挖掘到的客户信息转化为针对性价值主张",
            "excellent": "将≥4项核心信息与公司优势一一对应给出具体方案",
            "pass": "能说出2-3个公司优势对应客户需求",
            "fail": "泛泛介绍公司，与客户需求脱节"
        })

with open(RUBRICS_PATH, 'w', encoding='utf-8') as f:
    for r in rubrics:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Updated {len(rubrics)} entries")
print("\n第一条的 需求挖掘 维度预览:")
nd = rubrics[0]['scoring_dimensions']
for d in nd:
    if '七' in d.get('description','') or '需求' in d.get('dimension',''):
        for k, v in d.get('seven_elements', {}).items():
            print(f"  {k}: {v['element']} (权重{v['weight']})")
        print(f"  总分: {d['score']}, 规则: {d['scoring_guide']}")
