# Enterprise Sales Process Training Program

基于 FastAPI、FastRTC、Gradio 和阿里云 DashScope 的国际物流销售语音陪练系统。项目同时提供桌面端实时陪练控制台，以及面向手机 / 企业微信 H5 的对讲机模式和实时语音模式。

## 当前能力

- 案例驱动陪练：主数据来自 `docs/roleplay_cases.jsonl`，按训练类型、难度和状态机驱动客户回复。
- 双评分链路：`training_evaluator.py` 支持 LLM 评分，失败时 fallback 到启发式评分。
- 双数据库后端：`conversation_store.py` 门面层支持 SQLite / MySQL 切换。
- 双移动入口：
  - `/mobile`：录音上传式“对讲机模式”
  - `/realtime`：WebSocket 实时语音模式
- 前端模板独立化：移动端页面已经迁移到 `core/templates/mobile.html` 和 `core/templates/realtime.html`。
- 客户画像与训练记录：`/realtime` 页面支持配置预览客户画像、实时评分、历史会话卡片。
- WebSocket 实时增强：
  - 前端本地停顿触发 `flush`
  - 服务端支持 `asr_partial`
  - 手机端切句策略做了保守优化，优先减少误切

## 目录结构

```text
core/
  app_new_web.py              桌面端入口（FastRTC + Gradio）
  app_new_wechat.py           移动端入口（/mobile + /realtime + WebSocket）
  fastrtc_new_web.py          训练语义单一真相源，含 run_customer_turn()
  fastrtc_new_wechat.py       旧 H5 同步链路，仍保留作兼容参考
  voice_ws.py                 WebSocket 实时语音处理（VAD → ASR → LLM → TTS）
  prompt_assembler.py         案例 / 状态机 / few-shot prompt 组装
  case_loader.py              roleplay_cases.jsonl 校验与匹配
  training_config.py          YAML 配置加载与 fallback prompt
  training_data_context.py    raw_call / rubric 上下文桥接
  training_evaluator.py       评分引擎（LLM + heuristic）
  conversation_store.py       数据库门面层
  db_sqlite.py                SQLite 后端
  db_mysql.py                 MySQL 后端
  templates/
    mobile.html               对讲机模式前端
    realtime.html             实时语音模式前端
docs/
  roleplay_cases.jsonl        案例主数据（v2.0）
  raw_calls.jsonl             原始通话参考
  evaluation_rubrics.jsonl    当前评分 rubric（仍以 v1.0 为主）
  SPEC.md                     详细规范
scripts/
  generate_rubrics.py         基于 roleplay_cases 批量生成 rubric
  migrate_sqlite_to_mysql.py  SQLite → MySQL 数据迁移
tests/
  test_case_loader_validation.py  case schema / 状态机校验
```

## 快速开始

```powershell
cd D:\workspace\personal_project
conda activate fastrtc_env
pip install -r requirements.txt
copy .env.example .env
```

至少配置以下环境变量：

```text
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

如果要切到 MySQL：

```text
DB_ENGINE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=training_memory
```

## 启动方式

### 1. 桌面端控制台

```powershell
conda activate fastrtc_env
python -m core.app_new_web
```

默认地址：

```text
http://127.0.0.1:8520/app
http://127.0.0.1:8520/stream
http://127.0.0.1:8520/debug/status
```

### 2. 移动端 H5

```powershell
conda activate fastrtc_env
python -m core.app_new_wechat
```

默认地址：

```text
http://127.0.0.1:8511/mobile
http://127.0.0.1:8511/realtime
http://127.0.0.1:8511/api/training/config
http://127.0.0.1:8511/api/training/history?limit=10
```

## 关键实现说明

### 训练语义单一真相源

`core/fastrtc_new_web.py` 里的 `run_customer_turn()` 是唯一训练语义实现。
`voice_ws.py` 和 `app_new_wechat.py` 必须通过它委托，不再各自复制 prompt / 护栏 / 状态机逻辑。

### 数据现状

- `roleplay_cases.jsonl`：当前主数据源，已全面使用 `schema_version: 2.0`
- `raw_calls.jsonl`：真实通话节奏、话术和 outcome 参考
- `evaluation_rubrics.jsonl`：当前线上仍以 `schema_version: 1.0` 为主，和 `case v2.0` 还存在代际断层

项目已经有批量生成脚本：

```powershell
python scripts/generate_rubrics.py
```

该脚本会按 `roleplay_cases.jsonl` 生成 `evaluation_rubrics v2.0`，适合后续补齐 rubric 覆盖率。

### 实时语音现状

`/realtime` 目前做了几类稳定性优化：

- 本地停顿触发 `flush`
- 服务端 `asr_partial` 中间识别文本
- 手机端更保守的切句参数，减少一句话被切成两段
- iOS / 企微播放侧的预缓冲与卡顿缓解

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
- 状态机合法性 / 可达性
- `schema_version` 必须为 `2.0`

## 已知现状 / 后续重点

- `evaluation_rubrics.jsonl` 目前覆盖率明显低于 `roleplay_cases.jsonl`，属于后续优先补齐项。
- 手机端实时语音体验已做多轮优化，但和桌面 WebRTC 链路相比仍更依赖浏览器 / WebView 稳定性。
- Cloudflare quick tunnel 适合临时验证，不适合长期生产使用。

## 详细规范

详见：

- [SPEC.md](SPEC.md)
- [docs/SPEC.md](docs/SPEC.md)
