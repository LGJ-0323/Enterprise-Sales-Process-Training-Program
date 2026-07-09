# SPEC

本文件是根目录规范入口，详细技术规范见：

- [docs/SPEC.md](docs/SPEC.md)

## 项目摘要

本项目是一个面向国际物流销售训练的 Realtime AI 语音陪练系统。系统支持 PC、移动 H5 和企业微信入口，基于真实通话案例、客户画像、状态机和评分 Rubric，模拟客户异议、报价追问、需求试探和推进闭环等销售场景。

## 架构快照

```text
浏览器 / 企业微信 H5
  -> WebRTC / WebSocket / PCM Audio
  -> Qwen-Omni Realtime 或 DashScope ASR/TTS
  -> 语言大模型业务智能层
  -> 客户画像 + 状态机 + Rubric + RAG
  -> 训练反馈 + 会话存储 + 管理后台
```

## LoRA/QLoRA 定位

LoRA/QLoRA 适合用于语言大模型业务智能层，而不是语音采集、ASR、TTS 或 DashScope Realtime 接口本身。可用于 AI 客户回复生成、销售意图识别、异议分类、槽位抽取、训练评分和反馈生成。

## 当前核心入口

- PC 桌面端：`core/app_new_web.py`
- 移动 / 企业微信端：`core/app_new_wechat.py`
- 管理后台：`core/admin_dashboard.py`
- 经典实时语音 WebSocket：`core/voice_ws.py`
- Qwen-Omni Realtime 代理：`core/omni_voice_ws.py`
- 详细规范：[docs/SPEC.md](docs/SPEC.md)
