"""
app_new_web.py — 智能销售陪练控制台

使用 fastrtc_new_web（guardrails + 会话记忆 + WebRTC 流式 + P0 v2）
布局: 左栏(阶段/难度/音色 + 客户画像) 中栏(WebRTC语音 + 实时转写) 右栏(评分占位)
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
import uvicorn
from fastapi import FastAPI
from starlette.responses import HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# ── 导入 fastrtc_new_web（融合版） ─────────────────────
try:
    from .fastrtc_new_web import LAST_STATUS, stream
    from .training_config import _label, difficulty_choices, resolve_training, resolve_voice, stage_choices, voice_choices
    from .case_loader import case_count as get_case_count, find_case
except ImportError:
    from fastrtc_new_web import LAST_STATUS, stream
    from training_config import _label, difficulty_choices, resolve_training, resolve_voice, stage_choices, voice_choices
    from case_loader import case_count as get_case_count, find_case

app = FastAPI()
stream.mount(app)
app = gr.mount_gradio_app(app, stream.ui, path="/stream")

DEFAULT_STAGE_ID = os.getenv("TRAINING_STAGE_ID", "cold_call")
DEFAULT_DIFFICULTY_ID = os.getenv("TRAINING_DIFFICULTY_ID", "easy")
DEFAULT_VOICE_ID = os.getenv("TRAINING_VOICE_ID", "longsanshu_v3")

# ── API ────────────────────────────────────────────────
def _choice(l): return [{"label":lb,"value":v} for lb,v in l]

@app.get("/api/config")
async def api_config():
    return {"stages":_choice(stage_choices()),"difficulties":_choice(difficulty_choices()),"voices":_choice(voice_choices()),
            "defaults":{"stage_id":DEFAULT_STAGE_ID,"difficulty_id":DEFAULT_DIFFICULTY_ID,"voice_id":DEFAULT_VOICE_ID}}

@app.get("/api/persona")
async def api_persona(stage_id="cold_call",difficulty_id="easy",voice_id="longsanshu_v3"):
    try:
        s,c,d=resolve_training(stage_id,None,difficulty_id);v=resolve_voice(voice_id)
        att=c.get("attitude",{});sc=c.get("state_curve",{});traits=list(sc.get("states",{}).keys())[:5]
        if att.get("label"):traits.insert(0,att["label"])
        cn=0
        try:
            cc=find_case(_label(s),_label(d))
            if cc:cn=len(cc.get("few_shot_examples",[]))
        except:pass
        return {"name":c.get("name","?"),"role":c.get("role",""),"style":att.get("label",""),"desc":att.get("style",""),
                "traits":traits,"goal":s.get("training_goal",""),"voice_label":v.get("label",voice_id),
                "stage":_label(s),"difficulty":_label(d),"case_fewshot":cn,"case_total":get_case_count()}
    except Exception as e: return JSONResponse({"error":str(e)},500)

@app.get("/debug/status")
async def debug_status(): return LAST_STATUS


# ── 仪表盘 ─────────────────────────────────────────────
DASHBOARD = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>智能销售陪练 · Guardrails</title><style>
:root{--bg:#eef3f8;--panel:#fff;--ink:#10243f;--muted:#64748b;--line:#d9e2ee;--teal:#13b8a6;--blue:#2563eb;--red:#ef4444;--green:#22c55e;--amber:#f59e0b;--shadow:0 14px 36px rgba(18,38,63,0.10)}
*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;background:radial-gradient(circle at 10% 10%,rgba(19,184,166,0.06),transparent 28%),radial-gradient(circle at 85% 5%,rgba(37,99,235,0.04),transparent 30%),linear-gradient(135deg,#f8fbff 0%,var(--bg) 100%);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",sans-serif;overflow:hidden}
.stage{display:flex;flex-direction:column;height:100vh;max-width:1560px;margin:0 auto;padding:10px 14px 6px}
.topbar{display:flex;align-items:center;justify-content:space-between;flex-shrink:0;margin-bottom:6px}
.brand{display:flex;gap:8px;align-items:center}
.icon{width:32px;height:32px;border-radius:7px;background:linear-gradient(135deg,var(--teal),var(--blue));color:#fff;display:grid;place-items:center;font-size:15px;font-weight:800}
h1{font-size:18px;font-weight:850;color:#061938}
.sub{font-size:9px;color:var(--muted)}
.pills{display:flex;gap:5px}
.pill{display:inline-flex;align-items:center;gap:4px;height:22px;padding:0 7px;border-radius:999px;background:#fff;border:1px solid var(--line);font-size:9px;color:var(--muted);font-weight:650}
.dot{width:5px;height:5px;border-radius:50%}.dot.on{background:var(--green);box-shadow:0 0 0 3px rgba(34,197,94,0.08)}.dot.off{background:var(--muted)}

.layout{display:grid;grid-template-columns:260px 1fr 260px;gap:8px;flex:1;min-height:0}
.panel{background:rgba(255,255,255,0.88);border:1px solid var(--line);border-radius:7px;box-shadow:var(--shadow);overflow:hidden;display:flex;flex-direction:column}
.ph{padding:7px 9px;border-bottom:1px solid #edf2f7;font-size:11px;font-weight:820;display:flex;align-items:center;gap:5px;flex-shrink:0}
.tag{display:inline-flex;align-items:center;height:16px;padding:0 4px;border-radius:3px;background:#d8faf5;color:#078b7f;font-size:8px;font-weight:800}
.sec{padding:7px 9px;border-bottom:1px solid #edf2f7}.sec:last-child{border-bottom:0}
.lbl{font-size:9px;color:#34506f;font-weight:780;margin-bottom:2px}
select{width:100%;height:30px;border:1px solid #dce5f0;border-radius:5px;padding:0 6px;font-size:10px;background:#fff;margin-bottom:4px}
select:focus{outline:none;border-color:var(--teal)}

.persona{display:grid;grid-template-columns:36px 1fr;gap:7px;align-items:center;padding:6px;border:1px solid #dce5f0;border-radius:6px;background:linear-gradient(180deg,#fff,#f8fbff)}
.av{width:34px;height:34px;border-radius:50%;background:linear-gradient(140deg,#0f766e,#14b8a6);color:#fff;display:grid;place-items:center;font-size:14px;font-weight:900}
.pname{font-size:13px;font-weight:880}.prole{font-size:8px;color:#078b7f;font-weight:820}
.pdesc{margin-top:2px;font-size:8px;color:var(--muted);line-height:1.3}
.chips{display:flex;flex-wrap:wrap;gap:2px;margin-top:4px}
.chip{padding:1px 4px;border-radius:3px;background:#f1f5f9;color:#475569;font-size:8px;font-weight:700}
.info{margin-top:4px;font-size:8px;color:var(--muted)}
.goal{margin-top:4px;font-size:8px;color:#405875;line-height:1.3;background:#f8fbff;padding:3px 5px;border-radius:4px}

.stream-frame{flex:1;border:none;width:100%;height:100%;min-height:300px}
.transcript-box{height:180px;overflow-y:auto;padding:6px;font-size:10px;line-height:1.5;background:#fafbfd;border-top:1px solid var(--line)}
.transcript-box .t-line{padding:3px 5px;margin:1px 0;border-radius:3px}
.t-user{background:#eaf7ff;text-align:right}
.t-ai{background:#fff7ed}
.t-system{color:var(--muted);font-style:italic;text-align:center}
.right-card{padding:7px 9px}
.score-num{font-size:28px;font-weight:900;color:#0f766e;text-align:center}
.footer{text-align:center;padding:4px;font-size:8px;color:#8ca0bb;flex-shrink:0}
.footer span{margin:0 1px;padding:1px 3px;background:#fff;border:1px solid #dce5f0;border-radius:3px;color:#405875}
</style></head><body>
<main class="stage">
<header class="topbar"><div class="brand"><div class="icon">训</div><div><h1>智能销售陪练 · Guardrails 增强版</h1><div class="sub">WebRTC · Silero VAD · 角色校验 · 会话记忆 · P0 few-shot</div></div></div>
<div class="pills"><div class="pill"><span class="dot on" id="wsDot"></span><span id="wsText">系统就绪</span></div><div class="pill">Guardrails: <span style="color:var(--green)">✓</span></div><div class="pill">P0: <span id="p0Status">--</span></div></div></header>
<section class="layout">
<!-- ═══ LEFT: 阶段/难度/音色 + 客户画像 ═══ -->
<aside class="panel" style="overflow-y:auto">
<div class="ph">⚙ 训练配置</div>
<div class="sec">
<div class="lbl">训练阶段</div>
<select id="stageSel"><option>加载中...</option></select>
<div class="lbl">难度等级</div>
<select id="diffSel"><option>加载中...</option></select>
<div class="lbl">客户音色</div>
<select id="voiceSel"><option>加载中...</option></select>
</div>
<div class="sec">
<div class="lbl">客户画像</div>
<div class="persona"><div class="av" id="avInit">客</div><div><div class="pname" id="pName">加载中...</div><div class="prole" id="pRole"></div></div></div>
<div class="pdesc" id="pDesc"></div><div class="chips" id="pChips"></div>
<div class="info" id="pInfo"></div><div class="goal" id="pGoal"></div>
</div>
<div class="sec"><div class="lbl">Guardrails 状态</div><div style="font-size:8px;color:#078b7f;line-height:1.5">✓ 角色防反转<br>✓ 脏话/混淆拦截<br>✓ 身份前缀清洗<br>✓ TTS 3次重试</div></div>
</aside>
<!-- ═══ CENTER: WebRTC 语音 + 实时转写 ═══ -->
<div class="panel">
<div class="ph">🎙 实时语音陪练 <span class="tag">WebRTC 流式</span></div>
<iframe class="stream-frame" src="/stream" allow="microphone;autoplay;camera" id="streamFrame"></iframe>
<div class="ph" style="border-top:1px solid var(--line);border-bottom:0">💬 实时语音转文字</div>
<div class="transcript-box" id="transcript"></div>
</div>
<!-- ═══ RIGHT: 评分 ═══ -->
<aside class="panel" style="overflow-y:auto">
<div class="ph">📊 评分与复盘</div>
<div class="right-card"><div class="score-num" style="color:#94a3b8">--</div><div style="text-align:center;font-size:9px;color:var(--muted)">综合得分 / 100</div></div>
<div class="sec"><div class="lbl">评分维度</div><div style="font-size:8px;color:var(--muted);line-height:1.8">开场清晰度 · 需求挖掘 · 价值表达<br>异议处理 · 专业可信度 · 成交推进</div></div>
<div class="ph">🎯 AI 教练反馈</div><div class="sec"><div style="font-size:9px;color:var(--muted);text-align:center;padding:8px">完成训练后生成</div></div>
<div class="ph">📋 会话状态</div><div class="sec"><div style="font-size:8px;color:var(--muted);line-height:1.5" id="sessionInfo">等待中...</div></div>
</aside>
</section>
<div class="footer"><span>Gradio WebRTC</span><span>Silero VAD</span><span>Guardrails</span><span>SQLite Memory</span><span>阿里云 ASR</span><span>千问 LLM</span><span>P0 few-shot</span><span>CosyVoice TTS</span></div>
</main>
<script>
let stageId='cold_call',diffId='easy',voiceId='longsanshu_v3';

async function loadConfig(){try{const r=await fetch('/api/config');const d=await r.json();
['stageSel','diffSel','voiceSel'].forEach((id,idx)=>{const keys=['stages','difficulties','voices'];const defs=['cold_call','easy','longsanshu_v3'];
const s=document.getElementById(id);s.innerHTML='';d[keys[idx]].forEach(o=>{const opt=document.createElement('option');opt.value=o.value;opt.textContent=o.label;s.appendChild(opt)});
s.value=defs[idx]});
}catch(e){}}

async function loadPersona(){try{const r=await fetch('/api/persona?stage_id='+stageId+'&difficulty_id='+diffId+'&voice_id='+voiceId);const d=await r.json();
document.getElementById('pName').textContent=d.name||'?';
document.getElementById('pRole').textContent=(d.role||'')+' / '+(d.style||'');
document.getElementById('pDesc').textContent=d.desc||'';
document.getElementById('avInit').textContent=(d.name||'客').charAt(0);
document.getElementById('pChips').innerHTML=(d.traits||[]).map(t=>'<span class="chip">'+t+'</span>').join('');
document.getElementById('pInfo').textContent='📍 '+(d.stage||'')+' · '+(d.difficulty||'')+' · '+(d.voice_label||'');
document.getElementById('pGoal').innerHTML='<strong>🎯</strong> '+(d.goal||'');
document.getElementById('p0Status').textContent=(d.case_fewshot||0)+' 示例/'+(d.case_total||0)+'案例';
}catch(e){}}

function onConfigChange(){stageId=document.getElementById('stageSel').value;diffId=document.getElementById('diffSel').value;voiceId=document.getElementById('voiceSel').value;loadPersona()}
['stageSel','diffSel','voiceSel'].forEach(id=>document.getElementById(id).addEventListener('change',onConfigChange));

// ── 实时转写轮询 ──
let lastPrompt='',lastResponse='';
async function pollTranscript(){try{const r=await fetch('/debug/status');const d=await r.json();
const box=document.getElementById('transcript');let updated=false;
if(d.prompt&&d.prompt!==lastPrompt){lastPrompt=d.prompt;box.innerHTML+='<div class="t-line t-user">🎤 '+d.prompt.replace(/</g,'&lt;')+'</div>';updated=true}
if(d.response_text&&d.response_text!==lastResponse){lastResponse=d.response_text;box.innerHTML+='<div class="t-line t-ai">🤖 '+d.response_text.replace(/</g,'&lt;')+'</div>';updated=true}
if(d.guardrail)box.innerHTML+='<div class="t-line t-system">⚠ Guardrail: '+d.guardrail+'</div>';
if(updated)box.scrollTop=box.scrollHeight;
if(d.training)document.getElementById('sessionInfo').textContent='阶段: '+(d.training.stage||'')+' · 客户: '+(d.training.customer||'')+' · 来源: '+(d.training.source||'?');
}catch(e){}}

loadConfig();loadPersona();setInterval(pollTranscript,1500);
</script></body></html>"""

@app.get("/")
async def dashboard(): return HTMLResponse(DASHBOARD)

if __name__=="__main__":
    port=int(os.getenv("APP_PORT","8520"))
    print(f"\n  🏠 http://127.0.0.1:{port}  |  🎙 http://127.0.0.1:{port}/stream  |  fastrtc_new_web (guardrails+P0)\n")
    uvicorn.run(app,host="127.0.0.1",port=port,reload=False)
