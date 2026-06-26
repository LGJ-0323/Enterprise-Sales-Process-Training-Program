# Enterprise Sales Process Training Program

这是一个基于 FastRTC、Gradio、FastAPI 和阿里云百炼 DashScope 的实时语音模拟客户陪练原型。

## 当前结构

```text
core/
  app_new.py                  # FastAPI + Gradio/FastRTC 挂载入口
  fastrtc_test.py             # 语音识别、千问回复、语音合成主链路
  index.html                  # 当前自定义 WebRTC 调试页面
  prompts/
    customer_profile.md       # 默认模拟客户身份背景
  training/
    stages/                   # 后续放四个销售训练阶段配置
    customers/                # 后续放客户画像配置
    difficulties/             # 后续放难度配置
    rubrics/                  # 后续放评分规则
    voices/                   # 后续放音色配置
log/                          # 本地运行日志，GitHub 只保留目录
run_app.py                    # 推荐启动入口
```

## 本地环境变量

复制 `.env.example` 为 `.env`，再填入自己的真实密钥。`.env` 已被 `.gitignore` 排除，不会上传。

必填：

```text
DASHSCOPE_API_KEY=你的阿里云百炼 API Key
```

## 启动

```powershell
D:\Anaconda3\envs\agent_new\python.exe D:\workspace\personal_project\run_app.py
```

默认访问：

```text
http://127.0.0.1:8510/ui
```

## GitHub 上传

如果远程仓库是空的，使用：

```powershell
git remote set-url origin https://github.com/LGJ-0323/Enterprise-Sales-Process-Training-Program.git
git push -u origin main
```

如果 HTTPS 推送失败，建议改用 SSH：

```powershell
git remote set-url origin git@github.com:LGJ-0323/Enterprise-Sales-Process-Training-Program.git
git push -u origin main
```
