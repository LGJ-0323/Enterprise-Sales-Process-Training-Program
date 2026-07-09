# 销售陪练系统规范文档 (SPEC)

> 版本：1.4
> 最后更新：2026-07-09
> 状态：活跃维护

---

## 1. 系统概述

### 1.1 系统定位

本项目是面向国际物流销售团队的 Realtime AI 语音陪练系统。系统支持销售通过 PC、移动 H5 或企业微信内置浏览器，与 AI 客户进行低延迟语音对话。AI 客户基于真实通话案例、客户画像、隐藏状态、状态机和评分 Rubric，模拟客户异议、报价追问、需求试探、老客回访和推进闭环等训练场景。

系统目标：

- 让销售在接近真实客户的语音互动中练习话术。
- 通过状态机控制客户态度变化，提升训练对抗感。
- 通过 Rubric 和历史会话生成结构化训练反馈。
- 支持 PC 与企业微信 H5 多端使用。
- 为后续 RAG、LoRA/QLoRA 和训练数据闭环预留扩展位。

### 1.2 核心技术栈

| 层级 | 技术 |
| --- | --- |
| 后端服务 | FastAPI、WebSocket、asyncio、Uvicorn |
| PC 实时语音 | FastRTC、Gradio、WebRTC |
| 移动实时语音 | Qwen-Omni Realtime、DashScope、PCM Audio、Server VAD |
| 经典语音链路 | DashScope ASR、DashScope TTS、本地 VAD、流式音频下发 |
| 训练智能层 | Prompt Engineering、客户画像、状态机、Rubric 评分、RAG |
| 数据存储 | SQLite、MySQL |
| 可选微调 | LoRA / QLoRA，用于语言大模型业务智能层 |

### 1.3 运行入口

| 入口 | 文件 | 路径 | 说明 |
| --- | --- | --- | --- |
| PC 控制台 | `core/app_new_web.py` | `/app` | FastRTC + Gradio 主训练入口 |
| PC 实时流 | `core/app_new_web.py` | `/stream` | 桌面实时语音训练 |
| 移动 Omni 实时语音 | `core/app_new_wechat.py` | `/mobile` | 浏览器 PCM -> Qwen-Omni -> 文本 + 音频 |
| 移动录音上传 | `core/app_new_wechat.py` | `/mobile/upload` | 企业微信 / H5 兼容兜底 |
| 移动兼容页 | `core/app_new_wechat.py` | `/mobile/compat` | 兼容检测与降级入口 |
| 经典实时语音 | `core/app_new_wechat.py` | `/realtime` | WebSocket -> VAD -> ASR -> LLM -> TTS |
| 企微配置页 | `core/app_new_wechat.py` | `/wechat/config` | 三页面训练流程：配置 |
| 企微对话页 | `core/app_new_wechat.py` | `/wechat/chat` | 三页面训练流程：对话 |
| 企微评分页 | `core/app_new_wechat.py` | `/wechat/score` | 三页面训练流程：评分 |
| 管理后台 | `core/admin_dashboard.py` | `/admin` | 会话查询、训练复盘 |

### 1.4 架构原则

`core/fastrtc_new_web.py` 中的 `run_customer_turn()` 是客户回复、状态流转和训练语义的核心入口。所有 HTTP、WebSocket、WebRTC 和移动端入口都应复用同一套案例加载、Prompt 组装、状态机和评分逻辑，避免不同入口出现行为分叉。

---

## 2. 总体架构

### 2.1 训练主链路

```text
docs/roleplay_cases.jsonl
  -> core/case_loader.py
  -> core/prompt_assembler.py
  -> core/training_data_context.py
  -> core/fastrtc_new_web.py: run_customer_turn()
  -> core/training_evaluator.py
  -> core/conversation_store.py
```

### 2.2 实时语音架构

```text
浏览器麦克风
  -> PCM / WebRTC / WebSocket
  -> VAD 或 Server VAD
  -> ASR / Qwen-Omni Realtime
  -> 语言大模型业务智能层
  -> 客户状态机与 Prompt 上下文
  -> TTS / Omni 音频输出
  -> 浏览器播放
```

### 2.3 语言大模型业务智能层

语言大模型层负责以下任务：

- AI 客户角色扮演与回复生成
- 销售意图识别
- 客户异议识别
- 槽位抽取
- 客户状态流转判断
- 训练评分与反馈生成
- 与 RAG 上下文、客户画像、Rubric 的融合

LoRA/QLoRA 的推荐位置就在这一层。

---

## 3. LoRA/QLoRA 微调定位

### 3.1 不建议微调的位置

本项目不建议把 LoRA 描述为用于以下位置：

- 浏览器麦克风采集
- PCM / WebSocket 传输
- Server VAD 或本地 VAD
- DashScope ASR
- DashScope TTS
- DashScope / Qwen-Omni Realtime 语音 API 本身

这些模块主要解决实时听说、断句、音频传输和播放问题，通常通过服务配置、链路优化和前端兼容处理来提升体验。

### 3.2 推荐微调的位置

LoRA/QLoRA 推荐用于开源文本大模型或独立业务模型，作为语言大模型业务智能层的一部分。

推荐任务：

1. AI 客户回复生成

   学习国际物流客户在价格、时效、清关、舱位、派送、历史合作不信任等场景下的真实表达方式。

2. 销售意图识别与异议分类

   识别销售是否在报价、介绍方案、确认需求、处理异议、推进下一步，判断客户异议类型。

3. 槽位抽取与状态机辅助

   抽取航线、目的港、货物类型、时效、价格、出货计划、下一步动作等字段，辅助客户状态流转。

4. 训练评分与反馈生成

   基于 Rubric 学习需求确认、方案匹配、异议处理、风险表达、推进闭环等维度的评分与反馈风格。

### 3.3 推荐架构表达

```text
Realtime 语音链路负责“实时听说”
RAG 负责“事实知识和案例依据”
LoRA/QLoRA 负责“领域表达、业务判断和反馈风格适配”
```

### 3.4 可落地训练数据来源

可用于构造指令微调数据的数据来源：

- `docs/raw_calls.jsonl`：真实通话节奏、客户回应、销售推进方式
- `docs/roleplay_cases.jsonl`：客户画像、隐藏状态、状态机、few-shot 示例
- `docs/evaluation_rubrics.jsonl`：评分维度、must-do、critical mistakes
- 历史训练会话：销售输入、AI 客户回复、状态变化、评分结果

---

## 4. 核心模块

### 4.1 后端入口

| 文件 | 作用 |
| --- | --- |
| `core/app_new_web.py` | PC 桌面端入口，提供 Gradio / FastRTC 页面和桌面实时训练 |
| `core/app_new_wechat.py` | 移动 H5 与企业微信入口，提供 `/mobile`、`/realtime`、`/wechat/*` 等页面 |
| `core/admin_dashboard.py` | 管理后台，查询训练会话和回合明细 |
| `core/voice_ws.py` | 经典 WebSocket 实时语音链路 |
| `core/omni_voice_ws.py` | Qwen-Omni Realtime 移动语音代理 |

### 4.2 训练智能模块

| 文件 | 作用 |
| --- | --- |
| `core/fastrtc_new_web.py` | 训练语义核心，包含 `run_customer_turn()` |
| `core/case_loader.py` | 加载、校验和匹配训练案例 |
| `core/prompt_assembler.py` | 组装客户画像、状态机、few-shot 和业务上下文 |
| `core/training_data_context.py` | 桥接原始通话和 Rubric 上下文 |
| `core/training_evaluator.py` | LLM 评分与启发式评分 fallback |
| `core/training_config.py` | YAML 训练配置加载 |

### 4.3 数据库层

数据库采用门面层 + 后端实现：

```text
conversation_store.py
  ├─ db_sqlite.py
  └─ db_mysql.py
```

切换方式：

- `DB_ENGINE=sqlite`：本地默认
- `DB_ENGINE=mysql`：多用户、长期保存或部署环境

---

## 5. 数据文件契约

### 5.1 roleplay_cases.jsonl

- 文件：`docs/roleplay_cases.jsonl`
- 当前 schema：`2.0`
- 作用：训练案例主数据源

核心结构：

- `customer_role_card`
- `hidden_customer_state`
- `state_machine`
- `few_shot_examples`
- `failure_conditions`
- `customer_behavior_rules`
- `difficulty_variants`
- `training_goals`
- `conversation_opening`

### 5.2 raw_calls.jsonl

- 文件：`docs/raw_calls.jsonl`
- 作用：真实通话节奏、关键销售动作、结果参考

主要字段：

- `call_metadata`
- `summary.one_sentence`
- `summary.key_points`
- `summary.outcome`
- `transcript_turns`

### 5.3 evaluation_rubrics.jsonl

- 文件：`docs/evaluation_rubrics.jsonl`
- 作用：评分维度、must-do、critical mistakes、理想销售推进路径

当前状态：

- `roleplay_cases.jsonl` 已使用 `schema_version: 2.0`
- `evaluation_rubrics.jsonl` 仍处于升级和补齐阶段
- 当缺少 `case_id` 精确匹配时，评分逻辑会 fallback 到 `source_call_id` 或 `training_type`

### 5.4 Rubric 生成脚本

```powershell
python scripts/generate_rubrics.py
```

用途：

- 基于 `roleplay_cases.jsonl` 批量生成或补齐 `evaluation_rubrics.jsonl`
- 为评分覆盖率提升和后续 LoRA 反馈数据构造提供基础

---

## 6. Prompt、RAG 与客户状态机

### 6.1 Prompt 组装

`core/prompt_assembler.py` 负责把以下内容合并成模型输入：

- 客户角色信息
- 业务背景
- 隐藏客户状态
- 当前状态机状态
- few-shot 示例
- 失败红线
- 难度信息
- 历史对话

`core/training_data_context.py` 额外补充：

- `raw_calls.jsonl` 的真实通话上下文
- `evaluation_rubrics.jsonl` 的评分维度摘要

### 6.2 RAG 定位

RAG 用于注入外部事实和案例依据，不用于替代状态机或微调。

适合纳入 RAG 的内容：

- 国际物流销售案例
- 历史真实通话片段
- 常见客户异议
- 报价、时效、清关、舱位等业务知识
- 评分标准和优秀话术示例

规划能力：

- 混合检索
- 上下文拼接
- Rerank 精排
- 增量索引

### 6.3 状态推进

状态推进由训练核心逻辑驱动：

- 读取当前会话状态
- 根据销售输入和客户规则判断客户反应
- 生成客户回复
- 产出或校验 `next_state`
- 更新会话状态
- 到达终局状态时标记训练结束

---

## 7. 语音链路

### 7.1 PC 桌面端

```text
浏览器麦克风
-> FastRTC / WebRTC
-> DashScope ASR
-> run_customer_turn()
-> TTS
-> 浏览器播放
```

### 7.2 移动 Omni 实时语音

路径：`/mobile`、`/ws/mobile-omni`

```text
浏览器 getUserMedia
-> PCM 采集
-> WebSocket
-> Qwen-Omni Realtime
-> Server VAD
-> 实时文本与音频输出
-> H5 AudioContext 播放
```

特点：

- 更接近端到端 Realtime 语音体验
- 使用 `DASHSCOPE_OMNI_*` 环境变量控制模型、声音、采样率和 VAD 参数
- 适合移动端主链路

### 7.3 经典 WebSocket 实时语音

路径：`/realtime`、`/ws/voice`

```text
getUserMedia
-> ScriptProcessor
-> WebSocket PCM
-> SimpleVAD
-> DashScope ASR
-> run_customer_turn()
-> DashScope TTS
-> PCM / 音频分块下发
-> H5 AudioContext 播放
```

当前优化：

- 本地停顿触发 `flush`
- 服务端支持 `asr_partial`
- 手机端更保守的切句参数
- TTS 流式预缓冲和卡顿缓解

### 7.4 移动录音上传兼容模式

路径：`/mobile/upload`

```text
MediaRecorder
-> 上传整段录音
-> 后端 ASR / LLM / TTS
-> 返回文字 + 音频
```

适合：

- 企业微信 WebView 权限异常
- 移动端实时链路不稳定
- 演示或兜底训练

---

## 8. API 与路由

### 8.1 PC 端

- `GET /app`
- `GET /stream`
- `GET /api/training-assets`
- `GET /debug/status`

### 8.2 移动端与企微端

- `GET /mobile`
- `GET /mobile/upload`
- `GET /mobile/compat`
- `GET /realtime`
- `GET /wechat/config`
- `GET /wechat/chat`
- `GET /wechat/score`
- `GET /api/training/config`
- `POST /api/training/voice-turn`
- `GET /api/persona`
- `GET /api/training-assets`
- `GET /api/training/history`
- `GET /api/training/session-turns`
- `GET /api/training/resolve-case`
- `GET /api/training/session-summary`
- `GET /debug/status`
- `GET /debug/session-turns`
- `POST /debug/client-log`

### 8.3 WebSocket

- `/ws/audio-probe`：音频探针
- `/ws/voice`：经典实时语音管道
- `/ws/mobile-omni`：Qwen-Omni Realtime 移动语音代理

`/ws/voice` 常见事件：

- `ready`
- `status`
- `persona`
- `goal`
- `asr_partial`
- `asr_final`
- `response_text`
- `tts_done`
- `evaluation`
- `final_evaluation`
- `state_change`
- `session_info`
- `session_end`
- `error`

---

## 9. 环境变量

### 9.1 基础变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 必填 | DashScope Key |
| `DASHSCOPE_LLM_MODEL` | `qwen-plus` | LLM 模型 |
| `DASHSCOPE_ASR_MODEL` | `paraformer-realtime-v2` | ASR 模型 |
| `DASHSCOPE_TTS_MODEL` | `cosyvoice-v1` | TTS 模型 |
| `DASHSCOPE_TTS_VOICE` | `longxiaochun` | TTS 音色 |
| `APP_PORT` | `8520` / 环境变量可覆盖 | PC 端口 |
| `WECHAT_APP_HOST` | `127.0.0.1` | 移动端监听主机 |
| `WECHAT_APP_PORT` | `8511` | 移动端端口 |
| `DB_ENGINE` | `sqlite` | `sqlite` / `mysql` |
| `TRAINING_DB_PATH` | 空 | SQLite 文件路径，空时使用默认路径 |
| `MYSQL_HOST` | `127.0.0.1` | MySQL 主机 |
| `MYSQL_PORT` | `3306` | MySQL 端口 |
| `MYSQL_USER` | `root` | MySQL 用户 |
| `MYSQL_PASSWORD` | 空 | MySQL 密码 |
| `MYSQL_DATABASE` | `training_memory` | MySQL 数据库名 |

### 9.2 Qwen-Omni Realtime 变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_OMNI_REALTIME_MODEL` | `qwen3.5-omni-flash-realtime` | Omni Realtime 模型 |
| `DASHSCOPE_OMNI_VOICE` | `Tina` | Omni 输出音色 |
| `DASHSCOPE_OMNI_INPUT_SAMPLE_RATE` | `16000` | 输入采样率 |
| `DASHSCOPE_OMNI_OUTPUT_SAMPLE_RATE` | `24000` | 输出采样率 |
| `DASHSCOPE_OMNI_TRANSCRIPTION_MODEL` | `qwen3-asr-flash-realtime` | 转写模型 |
| `DASHSCOPE_OMNI_TURN_DETECTION` | `server_vad` | 断句模式 |
| `DASHSCOPE_OMNI_VAD_THRESHOLD` | `0.2` | VAD 阈值 |
| `DASHSCOPE_OMNI_VAD_PREFIX_PADDING_MS` | `300` | VAD 前置 padding |
| `DASHSCOPE_OMNI_VAD_SILENCE_DURATION_MS` | `800` | 静音断句时长 |

### 9.3 经典 WebSocket 语音变量

- `WS_ASR_SAMPLE_RATE`
- `WS_VAD_THRESHOLD`
- `WS_VAD_MIN_SPEECH_MS`
- `WS_VAD_MIN_SILENCE_MS`
- `WS_VAD_MAX_UTTERANCE_MS`
- `WS_MIN_UTTERANCE_MS`
- `WS_MIN_AUDIO_RMS`
- `WS_ASR_TRAILING_SILENCE_MS`
- `WS_ASR_MAX_SENTENCE_SILENCE_MS`
- `WS_ASR_CALLBACK_FRAME_BYTES`
- `WS_ASR_CALLBACK_FIRST_TIMEOUT_S`
- `WS_ASR_CALLBACK_GRACE_S`
- `WS_TTS_STREAM_CHUNK_MS`
- `WS_TTS_STREAM_CHUNK_PAUSE_MS`

---

## 10. 数据库与会话存储

### 10.1 SQLite

默认用于本地开发和单机演示。

```text
data/training_memory.sqlite3
```

### 10.2 MySQL

适合多用户训练、长期保存和部署环境。

```text
DB_ENGINE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=training_memory
```

### 10.3 迁移脚本

```powershell
python scripts/migrate_sqlite_to_mysql.py
```

---

## 11. 质量门禁

当前活跃测试：

```powershell
python -m pytest tests -q
```

主要覆盖：

- `roleplay_cases.jsonl` 可解析
- `schema_version == 2.0`
- 状态机状态有效
- 转移可达
- few-shot 基本格式正确

---

## 12. 已知状态与后续重点

### 12.1 已知状态

- PC、移动 H5、企业微信三页面入口已形成。
- `/mobile` 已接入 Qwen-Omni Realtime 移动语音链路。
- `/realtime` 保留经典 ASR/LLM/TTS 拆分链路，便于调试和兜底。
- `roleplay_cases.jsonl` 是当前案例主数据源。
- `evaluation_rubrics.jsonl` 仍处于补齐和升级阶段。
- 数据库存储支持 SQLite / MySQL 双后端。

### 12.2 后续建议

优先级较高的工作：

1. 补齐 `evaluation_rubrics` 覆盖率。
2. 升级评分逻辑，让 evaluator 更充分消费 `failure_conditions`、`customer_behavior_rules`、`training_goals`。
3. 建立 RAG 索引，支持真实通话、案例和 Rubric 的混合检索。
4. 构建 LoRA/QLoRA 指令微调数据，用于客户回复生成、异议识别和反馈生成。
5. 为企业微信 WebView 建立更完整的兼容性检测和降级策略。

---

## 13. 文件索引

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `README.md` | 项目首页说明 | 活跃 |
| `SPEC.md` | 根目录规范入口 | 活跃 |
| `docs/SPEC.md` | 详细规范 | 活跃 |
| `docs/roleplay_cases.jsonl` | case 主数据 | 活跃 |
| `docs/raw_calls.jsonl` | 原始通话参考 | 活跃 |
| `docs/evaluation_rubrics.jsonl` | 评分 Rubric | 待补齐 |
| `core/app_new_web.py` | PC 端入口 | 核心 |
| `core/app_new_wechat.py` | 移动 H5 / 企业微信入口 | 核心 |
| `core/admin_dashboard.py` | 管理后台 | 核心 |
| `core/fastrtc_new_web.py` | 训练语义核心 | 核心 |
| `core/voice_ws.py` | 经典实时语音链路 | 核心 |
| `core/omni_voice_ws.py` | Qwen-Omni Realtime 代理 | 核心 |
| `core/templates/mobile.html` | 移动 Omni 页面 | 活跃 |
| `core/templates/mobile_compat.html` | 移动兼容页面 | 活跃 |
| `core/templates/realtime.html` | 经典实时语音页面 | 活跃 |
| `core/templates/wechat_config.html` | 企微配置页 | 活跃 |
| `core/templates/wechat_chat.html` | 企微对话页 | 活跃 |
| `core/templates/wechat_score.html` | 企微评分页 | 活跃 |
| `core/conversation_store.py` | 数据库门面层 | 核心 |
| `core/db_sqlite.py` | SQLite 后端 | 核心 |
| `core/db_mysql.py` | MySQL 后端 | 核心 |
| `scripts/generate_rubrics.py` | Rubric 批量生成 | 工具 |
| `scripts/migrate_sqlite_to_mysql.py` | SQLite -> MySQL 迁移 | 工具 |
| `tests/test_case_loader_validation.py` | case schema 测试 | 门禁 |
