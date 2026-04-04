"""
CROM-IA V3.5b - Rosa Chat (HuggingFace Spaces Edition)
Motor de Inferencia Organica com RAG Injector DNA O(1)
"""

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import multiprocessing
import time
import json
import re

# ====================================================================
# 1. DOWNLOAD DO MODELO E CODEBOOK
# ====================================================================
print("[CROM-IA] Baixando modelo GGUF V3.5b...")
model_path = hf_hub_download(
    repo_id="CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic",
    filename="Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf"
)
codebook_path = hf_hub_download(
    repo_id="CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic",
    filename="logs_and_codebooks/bugged_codebook_v3_recipes.json"
)

# ====================================================================
# 2. CARREGAR CODEBOOK DNA
# ====================================================================
with open(codebook_path, "r", encoding="utf-8") as f:
    codebook_data = json.load(f)
dna_entries = codebook_data.get("entries", {})
total_bytes = sum(e["bytes"] for e in dna_entries.values())
print(f"[CROM-IA] {len(dna_entries)} Ponteiros DNA ({total_bytes} bytes)")

# ====================================================================
# 3. INSTANCIAR LLM
# ====================================================================
cores = max(multiprocessing.cpu_count(), 2)
print(f"[CROM-IA] {cores} threads | n_batch=512 | n_ctx=4096")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=cores,
    n_threads_batch=cores,
    n_batch=512,
    use_mmap=True,
    verbose=False
)
print("[CROM-IA] Pronto!")

# ====================================================================
# 4. RAG INJECTOR DNA
# ====================================================================
def rag_inject(text):
    hits = 0
    saved = 0
    result = []
    i = 0
    while i < len(text):
        if text[i] == '@' and i + 1 < len(text) and text[i + 1] == '@':
            found = False
            for kl in [4, 3, 2]:
                end = i + 2 + kl
                if end <= len(text):
                    key = text[i + 2:end]
                    if key in dna_entries:
                        result.append(dna_entries[key]["text"])
                        saved += dna_entries[key]["bytes"]
                        hits += 1
                        i = end
                        found = True
                        break
            if not found:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result), hits, saved

# ====================================================================
# 5. FORMATADOR DE PROMPT
# ====================================================================
def safe_str(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if val is None:
        return ""
    return str(val)

def clean_metrics(text):
    text = safe_str(text)
    return re.sub(r'\n*---\n.*$', '', text, flags=re.DOTALL).strip()

def formatar_prompt(message, history):
    prompt = "Abaixo esta uma instrucao CROM-IA.\n\n"
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role", "")
            content = clean_metrics(entry.get("content", ""))
            if role == "user":
                prompt += f"### Instruction:\n{content}\n\n### Input:\n\n\n"
            elif role == "assistant":
                prompt += f"### Response:\n{content}\n\n"
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            user_msg = safe_str(entry[0])
            bot_msg = clean_metrics(entry[1])
            prompt += f"### Instruction:\n{user_msg}\n\n### Input:\n\n\n### Response:\n{bot_msg}\n\n"
    prompt += f"### Instruction:\n{message}\n\n### Input:\n\n\n### Response:\n"
    return prompt

# ====================================================================
# 6. GERADOR DE RESPOSTA
# ====================================================================
def generate(message, history, temperature, repeat_penalty, max_tokens):
    prompt = formatar_prompt(message, history)

    stream = llm(
        prompt,
        max_tokens=int(max_tokens),
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        stop=["### Instruction:", "### Input:", "<|endoftext|>", "<|im_end|>", "</s>", "Abaixo esta", "#", "\n\n\n\n\n"],
        stream=True
    )

    resposta = ""
    t0 = time.time()
    toks = 0

    for chunk in stream:
        token = chunk["choices"][0]["text"]
        resposta += token
        toks += 1
        yield resposta

    dt = time.time() - t0
    tps = toks / dt if dt > 0 else 0

    resposta_rag, hits, saved = rag_inject(resposta)

    m = f"\n\n---\n**🧬 CROM-IA** ⚡ {tps:.1f} t/s · 📝 {toks} tokens · ⏱️ {dt:.1f}s · 🧵 {cores} threads"

    if hits > 0:
        m += f" · 🔬 RAG: {hits} hits ({saved}B expandidos)"
        final = resposta_rag
    else:
        final = resposta

    yield final + m

# ====================================================================
# 7. INTERFACE VISUAL
# ====================================================================
GITHUB = "https://github.com/MrJc01/crompressor-ia"
HF_MODEL = "https://huggingface.co/CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');
* { box-sizing: border-box; }
body, .gradio-container {
    background: #0a0f1a !important;
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9 !important;
    overflow-y: auto !important;
    height: auto !important;
    min-height: 100vh !important;
}
.gradio-container { max-width: 900px !important; margin: 0 auto !important; overflow: visible !important; }
.hdr {
    background: linear-gradient(135deg, #0f172a, #1e3a5f, #0f172a);
    border: 1px solid #334155; border-radius: 14px;
    padding: 1.5rem 1rem; text-align: center; margin-bottom: 0.5rem;
    box-shadow: 0 0 20px rgba(56,189,248,0.12);
}
.hdr h1 {
    font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem 0;
    background: linear-gradient(135deg, #38bdf8, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hdr p { color: #94a3b8; font-size: 0.85rem; margin: 0.2rem 0; }
.hdr .links { margin-top: 0.7rem; display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap; }
.hdr .links a, .hdr .links span {
    padding: 0.25rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500;
    text-decoration: none; transition: transform 0.15s;
}
.hdr .links a:hover { transform: translateY(-1px); }
.bg { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.bb { background: rgba(56,189,248,0.12); color: #38bdf8; border: 1px solid rgba(56,189,248,0.25); }
.bc { background: rgba(34,211,238,0.12); color: #22d3ee; border: 1px solid rgba(34,211,238,0.25); }
.stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.4rem; margin-bottom: 0.5rem; }
.sc {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 0.5rem; text-align: center;
}
.sv { font-size: 1rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }
.sl { font-size: 0.65rem; color: #94a3b8; margin-top: 0.1rem; }
"""

HEADER = f"""
<div class="hdr">
  <h1>🧬 Rosa — CROM-IA V3.5b</h1>
  <p>Motor de IA Organica Comprimida | 1.5B Params | Qwen2.5 Fine-Tuned 117k PT-BR</p>
  <p style="font-size:0.75rem;color:#64748b">CPU-Only | Compressao Termodinamica DNA O(1)</p>
  <div class="links">
    <a href="{GITHUB}" target="_blank" class="bg">⚙️ GitHub</a>
    <a href="{HF_MODEL}" target="_blank" class="bb">🤗 Modelo</a>
    <span class="bc">🧵 {cores} Threads</span>
    <span class="bc">🔬 {len(dna_entries)} DNA Ptrs</span>
  </div>
</div>
"""

STATS = f"""
<div class="stats">
  <div class="sc"><div class="sv">1.5B</div><div class="sl">Parametros</div></div>
  <div class="sc"><div class="sv">117k</div><div class="sl">Amostras</div></div>
  <div class="sc"><div class="sv">Q4_K_M</div><div class="sl">Quantizacao</div></div>
  <div class="sc"><div class="sv">{len(dna_entries)}</div><div class="sl">Codebook DNA</div></div>
</div>
"""

FOOTER = """
<div style="text-align:center;padding:0.4rem;color:#475569;font-size:0.7rem">
  CROM-IA © 2026<br>
  <a href="https://github.com/MrJc01/crompressor-ia" target="_blank" style="color:#38bdf8">GitHub</a> ·
  <a href="https://huggingface.co/CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic" target="_blank" style="color:#38bdf8">HuggingFace</a>
</div>
"""

with gr.Blocks(title="Rosa - CROM-IA V3.5b") as demo:
    gr.HTML(HEADER)
    gr.HTML(STATS)

    gr.ChatInterface(
        generate,
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Slider(0.0, 1.5, step=0.05, value=0.1, label="🔥 Temperatura"),
            gr.Slider(1.0, 2.0, step=0.05, value=1.15, label="🔁 Penalidade Repeticao"),
            gr.Slider(256, 4096, step=256, value=2048, label="📏 Max Tokens"),
        ],
        additional_inputs_accordion=gr.Accordion("⚙️ Configuracoes Avancadas", open=False),
    )

    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, css=CSS)
