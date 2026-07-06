"""
core 包 — 企业销售语音陪练系统的核心模块。

包含以下子模块：
- app_new_web:        桌面 WebRTC 控制台入口（Gradio + FastAPI）
- app_new_wechat:     移动 H5 录音上传入口（FastAPI + WebSocket）
- fastrtc_new_web:    WebRTC 实时语音链路（ASR → LLM → TTS）
- fastrtc_new_wechat: H5 版语音处理链路（录音上传 → ASR → LLM → TTS）
- training_config:    YAML 训练配置加载与 prompt 构建
- case_loader:        JSONL 案例加载与模糊匹配
- prompt_assembler:   将案例数据动态组装为 LLM prompt
- training_session:   运行时会话状态管理（状态机推进）
- training_data_context: 真实通话数据与评分标准注入
- training_evaluator: 训练评分（LLM 评分 + 启发式评分）
- conversation_store: SQLite 会话、轮次和记忆持久化
- voice_ws:           WebSocket 实时语音管道（VAD → ASR → LLM → TTS）
"""
