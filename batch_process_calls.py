"""
batch_process_calls.py — 批量处理销售录音 → raw_calls.jsonl + roleplay_cases.jsonl

用法: D:\Anaconda3\envs\fastrtc_env\python.exe D:\workspace\personal_project\batch_process_calls.py

流程:
  1. 扫描回访录音/陌call录音目录
  2. 跳过已在 raw_calls.jsonl 中的文件（按文件名中的日期时间匹配）
  3. ffmpeg 转码 → 阿里云 ASR 转写
  4. 千问 LLM 分析 → 生成 raw_calls 条目
  5. 千问 LLM 再分析 → 生成 roleplay_cases 条目（含 few_shot_examples）
  6. 追加到 JSONL 文件

安全: 每条之间间隔 2s，失败跳过不中断。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────
AUDIO_DIRS = [
    r"C:\Users\11253\Desktop\我的项目\项目md\customer_service\回访录音",
    r"C:\Users\11253\Desktop\我的项目\项目md\customer_service\陌call录音",
]

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"
RAW_CALLS_PATH = DOCS_DIR / "raw_calls.jsonl"
ROLEPLAY_PATH = DOCS_DIR / "roleplay_cases.jsonl"

# 每次运行最多处理 N 条（防止费用失控）
MAX_PER_RUN = int(os.getenv("BATCH_MAX", "10"))

# ffmpeg 路径
FFMPEG_BIN = os.getenv("FFMPEG_BIN", r"D:\tools\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin")
if not shutil.which("ffmpeg") and os.path.exists(os.path.join(FFMPEG_BIN, "ffmpeg.exe")):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# ── 初始化 dashscope ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass

import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    print("❌ 请设置 DASHSCOPE_API_KEY")
    sys.exit(1)

LLM_MODEL = os.getenv("DASHSCOPE_LLM_MODEL", "qwen-plus")
ASR_MODEL = os.getenv("DASHSCOPE_ASR_MODEL", "fun-asr")

# ── 工具函数 ────────────────────────────────────────────

def get_existing_ids() -> set[str]:
    """从 raw_calls.jsonl 读取已处理的 call_id 集合"""
    ids = set()
    if RAW_CALLS_PATH.exists():
        with open(RAW_CALLS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    ids.add(d.get("call_id", ""))
                except json.JSONDecodeError:
                    pass
    ids.discard("")
    return ids


def scan_audio_files() -> list[tuple[str, str]]:
    """扫描音频目录，返回 [(完整路径, 文件名), ...]，跳过已处理的"""
    existing = get_existing_ids()
    date_pat = re.compile(r"(\d{8})(\d{6})")
    new_files = []

    for audio_dir in AUDIO_DIRS:
        if not os.path.isdir(audio_dir):
            print(f"  ⚠ 目录不存在: {audio_dir}")
            continue

        for fname in os.listdir(audio_dir):
            if not fname.lower().endswith((".mp3", ".m4a", ".wav", ".webm")):
                continue

            # 检查是否已处理（按日期时间匹配）
            m = date_pat.search(fname)
            if m:
                call_dt = m.group(1) + m.group(2)
                if any(call_dt in eid for eid in existing):
                    continue

            full_path = os.path.join(audio_dir, fname)
            new_files.append((full_path, fname))

    return new_files


def transcribe(audio_path: str) -> str:
    """ASR 转写 — Recognition API + fun-asr 模型"""
    # ffmpeg 转 mp3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000",
             "-codec:a", "libmp3lame", "-b:a", "64k", mp3_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
    except subprocess.CalledProcessError:
        if audio_path.lower().endswith(".mp3"):
            shutil.copy2(audio_path, mp3_path)
        else:
            try: os.remove(mp3_path)
            except OSError: pass
            raise RuntimeError("ffmpeg 转码失败")

    try:
        rec = Recognition(model=ASR_MODEL, callback=None, format="mp3", sample_rate=16000)
        resp = rec.call(mp3_path)
        if resp.status_code != 200:
            raise RuntimeError(f"ASR 失败: {resp.message}")

        sentences = resp.get_sentence()
        if isinstance(sentences, list):
            return " ".join(s.get("text", "") for s in sentences).strip()
        elif isinstance(sentences, dict):
            return sentences.get("text", "").strip()
        return ""
    finally:
        try: os.remove(mp3_path)
        except OSError: pass


def call_llm(system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    """调用千问 LLM"""

    resp = Generation.call(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        result_format="message",
    )
    if resp.status_code == 200:
        return resp.output.choices[0].message.content.strip()
    raise RuntimeError(f"LLM 调用失败: code={resp.status_code} msg={resp.message}")


# ── 核心: 生成 raw_calls 条目 ──────────────────────────

RAW_CALL_SYSTEM = """你是一个销售通话分析专家。根据通话转写文本，生成结构化的 raw_calls 条目。

要求:
1. 输出纯 JSON（不要 markdown 代码块包裹）
2. call_type 从以下选择: 陌call、报价后回访、需求确认回访、问题处理回访、老客户维护、新客户开发、流失客户挽回、货物异常处理、投诉处理回访、旺季舱位提醒、合作续约沟通
3. transcript_turns 至少包含 6 轮对话
4. 如果无法确定的信息用 "未知" 填充
5. 严格按照以下 JSON schema:

{
  "call_id": "call_<日期时间>",
  "schema_version": "2.0",
  "source": {"audio_file": "<原始路径>", "duration_seconds": 0},
  "call_metadata": {"call_type": "<类型>", "sales_stage": "<阶段>", "business_line": "国际物流/<线路>", "scenario": "<一句话场景>", "language": "zh-CN", "is_training_simulation": false},
  "participants": {"sales": {"name_or_alias": "<销售名>", "company": "雄达国际物流", "experience_level": "mid"}, "customer": {"name_or_alias": "<客户名>", "role_inferred": "<角色>", "customer_lifecycle_stage": "<阶段>"}},
  "customer_profile": {"known_needs": ["<需求1>"], "current_status": ["<状态1>"], "price_sensitivity": "medium", "decision_criteria": ["<标准1>"], "competitor_reference": "<竞品信息或空>"},
  "result_label": "<next_step_secured | stalled | lost | info_only>",
  "result_detail": {"outcome": "<结果描述>", "next_step_type": "<下一步类型>", "info_captured": ["<信息1>"]},
  "summary": {"one_sentence": "<一句话总结>", "key_points": ["<要点1>", "<要点2>"], "outcome": "<结果>"},
  "transcript_turns": [{"turn_id": 1, "speaker": "sales", "text": "<话术>"}, {"turn_id": 2, "speaker": "customer", "text": "<话术>"}],
  "tags": ["<标签1>", "<标签2>"]
}
"""


def generate_raw_call(audio_path: str, transcript: str, fname: str) -> dict:
    """用 LLM 分析转写文本，生成 raw_calls 条目"""
    # 从文件名提取日期
    date_pat = re.search(r"(\d{8})(\d{6})", fname)
    call_dt = date_pat.group(1) + date_pat.group(2) if date_pat else datetime.now().strftime("%Y%m%d%H%M%S")

    user_prompt = f"""音频文件: {fname}
call_id 请用: call_{call_dt}
source.audio_file 请用: {audio_path}

转写文本:
{transcript[:4000]}

请生成 JSON:"""

    for attempt in range(3):
        try:
            raw = call_llm(RAW_CALL_SYSTEM, user_prompt)
            # 清理可能的 markdown 代码块
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            data = json.loads(raw)
            return data
        except (json.JSONDecodeError, RuntimeError) as e:
            print(f"    ⚠ 第 {attempt+1} 次生成失败: {e}")
            time.sleep(2)
    raise RuntimeError("3 次尝试均失败")


# ── 核心: 生成 roleplay_cases 条目 ─────────────────────

ROLEPLAY_STRUCT_SYSTEM = """你是一个销售培训案例设计师。根据真实通话信息，生成 AI 陪练系统的角色扮演配置（不含 few_shot_examples）。

输出纯 JSON，包含：case_id, source_call_id, schema_version("2.0"), training_type, difficulty, business_line, scene(1句话), customer_role_card({name,role,company_location,business_context,personality,communication_style,decision_style}), hidden_customer_state({main_concerns:[],price_sensitivity,trust_level_at_start,known_facts_can_reveal_if_asked:[],do_not_reveal_unless_deep_trust:[]}), state_machine({initial_state,states:{guarded/price_concerned/warming_up/open_to_next/shut_down:各含description/tone/reply_length/max_turns},transitions:[]}), customer_behavior_rules({global:[],by_state:{}}), failure_conditions:[{condition,result_state,customer_reaction}], difficulty_variants:{easy/medium/hard:{trust_start,objection_count,objection_intensity}}, training_goals:[{goal,maps_to_rubric_dimension}], conversation_opening:{customer_first_reply,initial_context_for_model}

要求:
- 3-5 个状态
- 4-5 条转移规则
- 3 条 failure_conditions
- difficulty_variants 三档
- training_goals 3-5 条，每条 maps_to_rubric_dimension"""

ROLEPLAY_FEWSHOT_SYSTEM = """你是一个销售培训案例设计师。根据对话片段，生成 few_shot_examples 数组。

输出纯 JSON 数组:
[{"state":"guarded","sales_input":"<销售原话>","customer_reply":"<客户原话>","why":"<1句话解释>"}, ...]

要求:
- 3-4 个片段，覆盖不同状态（guarded/price_concerned/warming_up 至少各一个）
- sales_input 和 customer_reply 尽量用真实对话原话
- why 解释客户为什么这么回复（1句话）"""


def generate_roleplay_case(raw_call: dict) -> dict:
    """基于 raw_call 分两步生成 roleplay_case（避免单次调用超时）"""
    call_type = raw_call.get("call_metadata", {}).get("call_type", "回访")
    result = raw_call.get("result_label", "info_only")
    difficulty = "easy" if result == "next_step_secured" else ("hard" if result == "lost" else "medium")
    call_id = raw_call.get("call_id", "")
    case_id = f"case_{call_id.replace('call_','')}"

    participants = raw_call.get("participants", {}).get("customer", {})
    customer = raw_call.get("customer_profile", {})
    summary = raw_call.get("summary", {})
    turns = raw_call.get("transcript_turns", [])

    # ── Step 1: 生成结构（不含 few_shot） ──
    struct_prompt = f"""case_id: {case_id}
source_call_id: {call_id}
training_type: {call_type}
difficulty: {difficulty}
business_line: {raw_call.get('call_metadata',{}).get('business_line','国际物流')}

客户: 姓名={participants.get('name_or_alias','?')}, 角色={participants.get('role_inferred','')}, 位置={participants.get('company_location','')}
需求: {json.dumps(customer.get('known_needs',[]), ensure_ascii=False)}
标准: {json.dumps(customer.get('decision_criteria',[]), ensure_ascii=False)}
结果: {summary.get('outcome','')}
场景: {raw_call.get('call_metadata',{}).get('scenario','')}

请生成完整 JSON（不含 few_shot_examples）:"""

    print("    📐 Step 1/2: 生成结构...")
    struct_data = None
    for attempt in range(3):
        try:
            raw = call_llm(ROLEPLAY_STRUCT_SYSTEM, struct_prompt)
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            struct_data = json.loads(raw)
            break
        except (json.JSONDecodeError, RuntimeError) as e:
            print(f"      ⚠ 第{attempt+1}次失败: {e}")
            time.sleep(2)

    if not struct_data:
        raise RuntimeError("3次尝试均无法生成结构")

    struct_data.setdefault("case_id", case_id)
    struct_data.setdefault("source_call_id", call_id)
    struct_data.setdefault("schema_version", "2.0")
    struct_data.setdefault("training_type", call_type)
    struct_data.setdefault("difficulty", difficulty)

    # ── Step 2: 生成 few_shot_examples ──
    customer_turns = [t for t in turns if t.get("speaker") == "customer"]
    sales_turns = [t for t in turns if t.get("speaker") == "sales"]

    # 取出关键对话片段（最多6轮，前3轮 + 中间异议 + 最后推进）
    key_turns = []
    for i, ct in enumerate(customer_turns[:6]):
        # 找对应的销售话术
        st_idx = next((j for j, st in enumerate(sales_turns) if st.get("turn_id", 0) == ct.get("turn_id", 0) - 1), i)
        st_text = sales_turns[min(st_idx, len(sales_turns)-1)].get("text", "") if sales_turns else ""
        key_turns.append({"sales": st_text, "customer": ct.get("text", "")})

    fewshot_prompt = f"""对话片段:
{json.dumps(key_turns, ensure_ascii=False, indent=2)}

请为这些状态生成 few_shot_examples: {list(struct_data.get('state_machine',{}).get('states',{}).keys())[:4]}"""

    print("    🎯 Step 2/2: 生成 few_shot...")
    fewshot_data = []
    for attempt in range(3):
        try:
            raw = call_llm(ROLEPLAY_FEWSHOT_SYSTEM, fewshot_prompt)
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            fewshot_data = json.loads(raw)
            if isinstance(fewshot_data, list) and len(fewshot_data) >= 2:
                break
        except (json.JSONDecodeError, RuntimeError) as e:
            print(f"      ⚠ 第{attempt+1}次失败: {e}")
            time.sleep(2)

    struct_data["few_shot_examples"] = fewshot_data if isinstance(fewshot_data, list) else []
    return struct_data


# ── 追加到 JSONL ───────────────────────────────────────

def append_jsonl(path: Path, data: dict):
    """原子追加一行 JSON 到 JSONL 文件"""
    line = json.dumps(data, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ── 主流程 ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  批量处理销售录音 → JSONL")
    print(f"  LLM: {LLM_MODEL} | ASR: {ASR_MODEL} | 最多: {MAX_PER_RUN} 条")
    print("=" * 60)

    # 1. 扫描
    audio_files = scan_audio_files()
    print(f"\n📁 待处理: {len(audio_files)} 条（已跳过已入库的）")

    if not audio_files:
        print("✅ 全部已处理，无需操作")
        return

    # 2. 限制每次运行数量
    batch = audio_files[:MAX_PER_RUN]
    print(f"🎯 本轮处理: {len(batch)} 条\n")

    success = 0
    for i, (audio_path, fname) in enumerate(batch, 1):
        print(f"[{i}/{len(batch)}] {fname}")
        try:
            # 2a. ASR
            print("  🎤 转写中...")
            transcript = transcribe(audio_path)
            if not transcript or len(transcript) < 10:
                print(f"  ⚠ 转写内容太短，跳过")
                continue
            print(f"  ✓ 转写: {transcript[:80]}...")

            # 2b. 生成 raw_calls
            print("  📝 生成 raw_calls...")
            raw_call = generate_raw_call(audio_path, transcript, fname)
            append_jsonl(RAW_CALLS_PATH, raw_call)
            print(f"  ✓ raw_calls: call_type={raw_call.get('call_metadata',{}).get('call_type','?')}, result={raw_call.get('result_label','?')}")

            # 2c. 生成 roleplay_cases
            print("  🎭 生成 roleplay_cases...")
            roleplay = generate_roleplay_case(raw_call)
            append_jsonl(ROLEPLAY_PATH, roleplay)
            fs_count = len(roleplay.get("few_shot_examples", []))
            print(f"  ✓ roleplay_cases: states={len(roleplay.get('state_machine',{}).get('states',{}))}, few_shot={fs_count}")

            success += 1

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            traceback.print_exc()

        # 间隔
        if i < len(batch):
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"  完成: {success}/{len(batch)} 条成功")
    print(f"  累计 raw_calls: {len(get_existing_ids())} 条")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
