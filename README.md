# 国际物流销售 Realtime AI 语音陪练系统

面向国际物流销售训练场景的实时语音陪练系统。系统支持销售通过 PC 浏览器、手机 H5 或企业微信内置浏览器，与 AI 客户进行低延迟语音对话；AI 客户基于真实通话案例、客户画像、状态机和评分 Rubric，模拟报价追问、客户异议、需求试探、老客回访等业务场景，并在训练后生成评分与反馈。

## 核心能力

- Realtime 语音对话：支持浏览器麦克风采集、PCM 音频流传输、Server VAD / 本地 VAD 断句、实时 ASR、LLM 理解与流式语音回复。
- 客户角色实时注入：按训练阶段、客户类型、难度和真实案例动态组装 Prompt，注入客户画像、业务背景、隐藏状态、异议规则和 few-shot 示例。
- 销售状态机陪练：根据销售表现推动客户在谨慎、价格敏感、逐步升温、愿意推进、终止沟通等状态间流转，增强对抗感和真实感。
- 多端训练入口：提供 PC 桌面控制台、企业微信三页面训练流程、移动 H5 Omni 实时语音、移动录音上传兼容模式和经典 WebSocket 实时语音模式。
- 训练数据闭环：围绕真实通话、客户案例、评分标准和历史会话沉淀训练数据，支持 SQLite / MySQL 存储和会话复盘。
- RAG 与领域适配：真实案例、原始通话和 Rubric 作为业务上下文来源；LoRA/QLoRA 适合放在语言大模型业务智能层，用于客户回复生成、异议识别和训练反馈生成的领域适配。

## 技术栈

- 后端：FastAPI、WebSocket、asyncio、Uvicorn
- 语音链路：Qwen-Omni Realtime、DashScope ASR/TTS、Server VAD、PCM Audio、AudioContext
- 智能编排：Prompt Engineering、状态机、Rubric 评分、RAG
- 数据存储：SQLite、MySQL
- 前端：HTML/CSS/JavaScript、移动 H5、企业微信 WebView 兼容
- 可选微调：LoRA / QLoRA（语言大模型层，不直接微调 DashScope Realtime 语音接口）

## 运行入口

| 入口 | 路径 | 说明 |
| --- | --- | --- |
| PC 桌面控制台 | `http://127.0.0.1:8520/app` | FastRTC + Gradio 主训练入口 |
| 桌面实时流 | `http://127.0.0.1:8520/stream` | PC WebRTC 实时语音训练 |
| 移动 Omni 实时语音 | `http://127.0.0.1:8511/mobile` | 浏览器 PCM -> Qwen-Omni -> 文本 + 音频 |
| 移动录音上传兼容模式 | `http://127.0.0.1:8511/mobile/upload` | 适合企业微信 WebView 兼容兜底 |
| 经典实时语音模式 | `http://127.0.0.1:8511/realtime` | WebSocket -> VAD -> ASR -> LLM -> TTS |
| 企微配置页 | `http://127.0.0.1:8511/wechat/config` | 训练配置 |
| 企微对话页 | `http://127.0.0.1:8511/wechat/chat` | 对话训练 |
| 企微评分页 | `http://127.0.0.1:8511/wechat/score` | 训练评分 |
| 管理后台 | `http://127.0.0.1:8521/admin` | 训练会话查询与复盘 |

## 目录结构

```text
core/
  app_new_web.py              PC 桌面端入口
  app_new_wechat.py           移动 H5 / 企业微信入口
  admin_dashboard.py          训练管理员后台
  fastrtc_new_web.py          训练语义单一真相源，含 run_customer_turn()
  voice_ws.py                 经典 WebSocket 实时语音链路
  omni_voice_ws.py            Qwen-Omni Realtime 移动语音代理
  prompt_assembler.py         客户画像 / 状态机 / few-shot Prompt 组装
  case_loader.py              roleplay_cases.jsonl 加载与校验
  training_evaluator.py       LLM + heuristic 评分引擎
  conversation_store.py       数据库门面层
  db_sqlite.py                SQLite 后端
  db_mysql.py                 MySQL 后端
  templates/
    mobile.html               移动 Omni 实时语音页面
    mobile_compat.html        移动兼容页面
    realtime.html             经典实时语音页面
    wechat_config.html        企微训练配置页
    wechat_chat.html          企微训练对话页
    wechat_score.html         企微训练评分页
docs/
  roleplay_cases.jsonl        训练案例主数据
  raw_calls.jsonl             真实通话参考
  evaluation_rubrics.jsonl    评分 Rubric
  SPEC.md                     详细技术规范
scripts/
  generate_rubrics.py         基于案例批量生成 Rubric
  migrate_sqlite_to_mysql.py  SQLite -> MySQL 数据迁移
tests/
  test_case_loader_validation.py
```

## 快速开始

```powershell
cd D:\workspace\personal_project
conda activate fastrtc_env
pip install -r requirements.txt
copy .env.example .env
```

至少配置：

```text
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

如需切换 MySQL：

```text
DB_ENGINE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=training_memory
```

## 启动方式

### PC 桌面端

```powershell
conda activate fastrtc_env
python -m core.app_new_web
```

默认地址（如未通过 `APP_PORT` 覆盖）：

```text
http://127.0.0.1:8520/app
http://127.0.0.1:8520/stream
http://127.0.0.1:8520/debug/status
```

### 移动端 / 企业微信 H5

```powershell
conda activate fastrtc_env
python -m core.app_new_wechat
```

默认地址（如未通过 `WECHAT_APP_PORT` 覆盖）：

```text
http://127.0.0.1:8511/mobile
http://127.0.0.1:8511/mobile/upload
http://127.0.0.1:8511/realtime
http://127.0.0.1:8511/wechat/config
http://127.0.0.1:8511/wechat/chat
http://127.0.0.1:8511/wechat/score
```

### 管理后台

```powershell
conda activate fastrtc_env
python -m core.admin_dashboard
```

默认地址（如未通过 `ADMIN_APP_PORT` 覆盖）：

```text
http://127.0.0.1:8521/admin
```

## LoRA/QLoRA 在本项目中的定位

LoRA 不放在语音采集、WebSocket、ASR、TTS 或 DashScope Realtime 语音接口本身，而是放在语言大模型业务智能层。推荐用途包括：

- AI 客户回复生成：学习国际物流客户的价格追问、时效质疑、清关顾虑和老客不信任表达。
- 销售意图识别与异议分类：识别报价、方案介绍、需求挖掘、过度承诺、推进下一步等销售动作。
- 槽位抽取与状态机驱动：抽取航线、品类、目的港、时效、价格、下一步动作等关键信息，辅助客户状态流转。
- 对话评分与训练反馈：基于 Rubric 学习需求确认、异议处理、方案匹配、推进闭环等维度的结构化评价。

推荐架构表达：

```text
Realtime 语音链路负责“实时听说”
RAG 负责“事实知识和案例依据”
LoRA/QLoRA 负责“领域表达、业务判断和反馈风格适配”
```

## 关键实现说明

### 训练语义单一真相源

`core/fastrtc_new_web.py` 中的 `run_customer_turn()` 是客户回复、状态机推进和训练语义的核心入口。`voice_ws.py`、`omni_voice_ws.py` 和 `app_new_wechat.py` 都应通过它或同一套 Prompt/案例装配逻辑委托，避免不同入口产生行为分叉。

### Realtime 语音链路

移动端当前同时保留两条实时链路：

- `/mobile`：优先使用 Qwen-Omni Realtime，浏览器发送 PCM 音频，由 Omni 完成实时语音理解和语音回复。
- `/realtime`：经典拆分链路，浏览器 PCM 经 WebSocket 传输，服务端完成 VAD、ASR、LLM、TTS 和音频分块下发。

企业微信或移动 WebView 不稳定时，可使用 `/mobile/upload` 录音上传模式兜底。

### 数据库存储

- 默认：SQLite，文件在 `data/training_memory.sqlite3`
- 可切换：MySQL
- 迁移脚本：

```powershell
python scripts/migrate_sqlite_to_mysql.py
```

## 测试

```powershell
python -m pytest tests -q
```

当前关键测试覆盖：

- `roleplay_cases.jsonl` 必填字段和 schema 校验
- 状态机合法性与可达性
- few-shot 基本格式

## 详细规范

- [SPEC.md](SPEC.md)
- [docs/SPEC.md](docs/SPEC.md)
