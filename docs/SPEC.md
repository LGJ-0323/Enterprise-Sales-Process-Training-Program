# 销售陪练系统规范文档 (SPEC)

> 版本：1.3  
> 最后更新：2026-07-08  
> 状态：活跃维护

---

## 1. 系统概述

### 1.1 系统定位

本项目是一个面向国际物流销售团队的语音陪练系统。系统用案例驱动的客户画像、状态机和评分规则，帮助销售在不同阶段完成：

- 陌 call
- 报价后回访
- 深入回访
- 逼单
- 老客维护
- 异常处理

### 1.2 运行入口

| 入口 | 文件 | 说明 |
| --- | --- | --- |
| 桌面控制台 | `core/app_new_web.py` | FastRTC + Gradio，主实时训练入口 |
| 移动 H5 - 对讲机 | `core/app_new_wechat.py` → `/mobile` | 录音上传式陪练 |
| 移动 H5 - 实时 | `core/app_new_wechat.py` → `/realtime` | WebSocket 实时语音模式 |

### 1.3 架构铁律

`core/fastrtc_new_web.py` 中的 `run_customer_turn()` 是训练语义的**单一真相源**。

以下模块都必须通过它委托，而不能自行复制 prompt / 护栏 / 状态机逻辑：

- `core/voice_ws.py`
- `core/app_new_wechat.py`
- 其他 HTTP / WebSocket / WebRTC 入口

---

## 2. 核心模块

### 2.1 训练主链路

```text
docs/roleplay_cases.jsonl
  → core/case_loader.py
  → core/prompt_assembler.py
  → core/training_data_context.py
  → core/fastrtc_new_web.py: run_customer_turn()
  → core/training_evaluator.py
  → core/conversation_store.py
```

### 2.2 前端模板

移动前端模板已从内嵌字符串迁移到：

- `core/templates/mobile.html`
- `core/templates/realtime.html`

`core/app_new_wechat.py` 通过 `_render_template()` 读取模板文件。

### 2.3 数据库层

数据库采用门面层 + 后端实现的结构：

```text
conversation_store.py
  ├─ db_sqlite.py
  └─ db_mysql.py
```

切换方式：

- `DB_ENGINE=sqlite`：本地默认
- `DB_ENGINE=mysql`：多用户 / 持久环境

---

## 3. 数据文件契约

### 3.1 roleplay_cases.jsonl

- 文件：`docs/roleplay_cases.jsonl`
- 当前 schema：`2.0`
- 作用：案例主数据源

当前 case 已普遍包含以下结构：

- `customer_role_card`
- `hidden_customer_state`
- `state_machine`
- `few_shot_examples`
- `failure_conditions`
- `customer_behavior_rules`
- `difficulty_variants`
- `training_goals`
- `conversation_opening`

### 3.2 raw_calls.jsonl

- 文件：`docs/raw_calls.jsonl`
- 当前 schema：`1.0`
- 作用：真实通话节奏、关键销售动作、结果参考

主要字段：

- `call_metadata`
- `summary.one_sentence`
- `summary.key_points`
- `summary.outcome`
- `transcript_turns`

### 3.3 evaluation_rubrics.jsonl

- 文件：`docs/evaluation_rubrics.jsonl`
- 当前线上活跃 schema：以 `1.0` 为主
- 作用：评分维度、must-do、critical mistakes、理想销售推进路径

**当前真实状态**

- `roleplay_cases.jsonl` 已基本是 `v2.0`
- `evaluation_rubrics.jsonl` 仍主要是 `v1.0`
- rubric 总量明显少于 case 总量
- `find_rubric()` 会在缺少 `case_id` 精确匹配时，退回到 `source_call_id` 或 `training_type` fallback

这意味着：当前评分体系已可运行，但并非所有 case 都有专属 rubric。

### 3.4 生成脚本

已有批量生成脚本：

```powershell
python scripts/generate_rubrics.py
```

用途：

- 基于 `roleplay_cases.jsonl` 批量生成 `evaluation_rubrics.jsonl`
- 输出目标是 rubric `v2.0`

---

## 4. Prompt 与客户状态机

### 4.1 Prompt 组装

`core/prompt_assembler.py` 负责把以下内容合并成模型输入：

- 客户角色信息
- 隐藏状态
- 当前状态机状态
- few-shot 示例
- 失败红线
- 难度信息
- 历史对话

`core/training_data_context.py` 额外补充：

- `raw_calls.jsonl` 的真实通话上下文
- `evaluation_rubrics.jsonl` 的评分维度摘要

### 4.2 状态推进

状态推进由 `run_customer_turn()` 内部驱动：

- 读取当前会话状态
- 让模型产出 `next_state`
- 校验状态是否可达
- 更新会话状态
- 若到达终局状态，则标记训练完成

---

## 5. 评分体系

### 5.1 当前模式

评分引擎在 `core/training_evaluator.py`，支持两种模式：

1. `LLM 评分`
2. `启发式评分`

流程：

- 优先走 `_llm_evaluation()`
- 失败时 fallback 到 `_heuristic_evaluation()`

### 5.2 当前 LLM 评分真正读取的字段

当前 LLM 评分 prompt 主要使用：

- `scoring_dimensions`
- `critical_mistakes`
- `case.scene`
- 对话文本

这意味着：

- 即使 case 已经有 `failure_conditions / customer_behavior_rules / training_goals`
- 如果 rubric schema 和 evaluator 不升级，这些字段也不会自动参与评分

### 5.3 当前启发式评分特点

启发式评分不是严格的规则引擎，而是基于维度名关键词的保守匹配。  
适合：

- 实时反馈
- LLM 评分失败时 fallback

不适合：

- 精细判断 case v2.0 的复杂行为规则
- 精确利用 `failure_conditions / training_goals / difficulty_variants`

### 5.4 当前已知评分缺口

当前评分体系最重要的现实问题是：

1. `case v2.0` 与 `rubric v1.0` 代际断层
2. rubric 覆盖率不足
3. evaluator 尚未消费大部分 case v2.0 新字段
4. `must_do / bonus_behaviors / critical_mistakes` 在现有 rubric 中同质化明显

---

## 6. WebRTC / WebSocket / H5 音频链路

### 6.1 桌面端

桌面控制台主要走：

```text
浏览器麦克风
→ FastRTC / WebRTC
→ DashScope ASR
→ run_customer_turn()
→ TTS
→ 浏览器播放
```

### 6.2 移动端实时模式

移动实时模式走：

```text
getUserMedia
→ ScriptProcessor
→ WebSocket PCM
→ SimpleVAD
→ DashScope ASR callback
→ run_customer_turn()
→ PCM TTS chunks
→ H5 AudioContext 播放
```

当前已做的手机端优化包括：

- 本地停顿触发 `flush`
- 服务端支持 `asr_partial`
- 更保守的手机端切句策略
- TTS 流式预缓冲和卡顿缓解

### 6.3 移动端对讲机模式

对讲机模式走：

```text
MediaRecorder
→ 上传整段录音
→ 后端 ASR / LLM / TTS
→ 返回文字 + 音频
```

适合：

- 企业微信 / H5 环境下优先保证稳定性

---

## 7. API 与路由

### 7.1 桌面端

主要路由：

- `/app`
- `/stream`
- `/api/training-assets`
- `/debug/status`

### 7.2 移动端

主要路由：

- `GET /mobile`
- `GET /realtime`
- `GET /api/training/config`
- `POST /api/training/voice-turn`
- `GET /api/persona`
- `GET /api/training-assets`
- `GET /api/training/history`
- `GET /api/training/session-turns`
- `GET /debug/status`
- `GET /debug/session-turns`

### 7.3 WebSocket 消息

`/ws/voice` 目前客户端可能接收到的主要事件：

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

## 8. 环境变量

关键变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 必填 | DashScope Key |
| `DASHSCOPE_LLM_MODEL` | `qwen-plus` | LLM 模型 |
| `DASHSCOPE_ASR_MODEL` | `paraformer-realtime-v2` | ASR 模型 |
| `DASHSCOPE_TTS_MODEL` | `cosyvoice-v1` | TTS 模型 |
| `APP_PORT` | `8520` | 桌面端端口 |
| `WECHAT_APP_PORT` | `8511` | 移动端端口 |
| `DB_ENGINE` | `sqlite` | `sqlite` / `mysql` |
| `TRAINING_DB_PATH` | `data/training_memory.sqlite3` | SQLite 文件路径 |
| `MYSQL_HOST` | `127.0.0.1` | MySQL 主机 |
| `MYSQL_PORT` | `3306` | MySQL 端口 |
| `MYSQL_USER` | `root` | MySQL 用户 |
| `MYSQL_PASSWORD` | 空 | MySQL 密码 |
| `MYSQL_DATABASE` | `training_memory` | MySQL 数据库名 |

移动实时链路常调变量：

- `WS_VAD_THRESHOLD`
- `WS_VAD_MIN_SPEECH_MS`
- `WS_VAD_MIN_SILENCE_MS`
- `WS_VAD_MAX_UTTERANCE_MS`
- `WS_MIN_UTTERANCE_MS`
- `WS_ASR_TRAILING_SILENCE_MS`
- `WS_ASR_CALLBACK_FRAME_BYTES`
- `WS_ASR_CALLBACK_FIRST_TIMEOUT_S`
- `WS_ASR_CALLBACK_GRACE_S`
- `WS_TTS_STREAM_CHUNK_MS`

---

## 9. 质量门禁

当前活跃测试：

- `tests/test_case_loader_validation.py`

主要校验：

- `roleplay_cases.jsonl` 可解析
- `schema_version == 2.0`
- 状态机状态有效
- 转移可达
- few-shot 基本格式正确

运行：

```powershell
python -m pytest tests -q
```

---

## 10. 当前已知事实与建议

### 10.1 已知事实

- 项目运行入口已经稳定迁移到：
  - `core.app_new_web`
  - `core.app_new_wechat`
- 移动前端真实来源已经迁移到 `core/templates/*`
- 数据库存储已经支持 SQLite / MySQL 双后端
- `roleplay_cases` 已是 v2.0 主数据源
- `evaluation_rubrics` 仍处于从 v1 向 v2 迁移阶段

### 10.2 后续建议

优先级较高的工作：

1. 补齐 `evaluation_rubrics` 覆盖率
2. 设计并落地 `rubric v2.0 schema`
3. 升级 `training_evaluator.py`，让它真正消费 `failure_conditions / customer_behavior_rules / training_goals`
4. 按场景族重构评分权重

---

## 11. 文件索引

| 文件 | 作用 | 状态 |
| --- | --- | --- |
| `README.md` | 项目首页说明 | 活跃 |
| `SPEC.md` | 根目录导航 | 活跃 |
| `docs/SPEC.md` | 详细规范 | 活跃 |
| `docs/roleplay_cases.jsonl` | case 主数据 | 活跃 |
| `docs/raw_calls.jsonl` | 原始通话参考 | 活跃 |
| `docs/evaluation_rubrics.jsonl` | 评分 rubric | 活跃，待升级 |
| `core/fastrtc_new_web.py` | 单一训练语义源 | 核心 |
| `core/voice_ws.py` | 移动实时语音链路 | 核心 |
| `core/app_new_wechat.py` | 移动端路由层 | 核心 |
| `core/templates/mobile.html` | 对讲机模式前端 | 活跃 |
| `core/templates/realtime.html` | 实时模式前端 | 活跃 |
| `core/conversation_store.py` | 数据库门面层 | 核心 |
| `core/db_sqlite.py` | SQLite 后端 | 核心 |
| `core/db_mysql.py` | MySQL 后端 | 核心 |
| `scripts/generate_rubrics.py` | 批量生成 rubric | 工具 |
| `scripts/migrate_sqlite_to_mysql.py` | SQLite → MySQL 迁移 | 工具 |
| `tests/test_case_loader_validation.py` | case schema 测试 | 门禁 |
