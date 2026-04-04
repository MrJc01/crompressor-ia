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
import os
import re

# ====================================================================
# 1. DOWNLOAD DO MODELO E CODEBOOK
# ====================================================================
print("[CROM-IA] Baixando modelo GGUF V3.5b (117k amostras)...")
model_path = hf_hub_download(
    repo_id="CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic",
    filename="Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf"
)

codebook_path = hf_hub_download(
    repo_id="CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic",
    filename="logs_and_codebooks/bugged_codebook_v3_recipes.json"
)

# ====================================================================
# 2. CARREGAR CODEBOOK DNA (RAG O(1) Injector)
# ====================================================================
print("[CROM-IA] Carregando Codebook DNA V3...")
with open(codebook_path, "r", encoding="utf-8") as f:
    codebook_data = json.load(f)
dna_entries = codebook_data.get("entries", {})
total_codebook_bytes = sum(e["bytes"] for e in dna_entries.values())
print(f"[CROM-IA] {len(dna_entries)} Ponteiros DNA carregados ({total_codebook_bytes} bytes)")

# ====================================================================
# 3. INSTANCIAR LLM COM FORCA MAXIMA
# ====================================================================
cores = max(multiprocessing.cpu_count(), 2)
print(f"[CROM-IA] Alocando {cores} threads | n_batch=512 | n_ctx=4096")

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=cores,
    n_threads_batch=cores,
    n_batch=512,
    use_mmap=True,
    verbose=False
)
print("[CROM-IA] Motor de Inferencia Pronto!")

# ====================================================================
# 4. RAG INJECTOR DNA (State Machine O(1))
# ====================================================================
def rag_inject(text):
    """Intercepta ponteiros @@XX e expande via Codebook DNA."""
    hits = 0
    bytes_saved = 0
    result = []
    i = 0
    while i < len(text):
        if text[i] == '@' and i + 1 < len(text) and text[i + 1] == '@':
            found = False
            for keylen in [4, 3, 2]:
                end = i + 2 + keylen
                if end <= len(text):
                    key = text[i + 2:end]
                    if key in dna_entries:
                        expanded = dna_entries[key]["text"]
                        result.append(expanded)
                        bytes_saved += dna_entries[key]["bytes"]
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
    return "".join(result), hits, bytes_saved

# ====================================================================
# 5. FORMATADOR DE PROMPT (Alpaca Rigoroso)
# ====================================================================
def formatar_prompt(message, history):
    prompt = "Abaixo esta uma instrucao CROM-IA.\n\n"
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role", "")
            content = entry.get("content", "")
            content = re.sub(r'\n*---\n.*$', '', content, flags=re.DOTALL)
            if role == "user":
                prompt += f"### Instruction:\n{content}\n\n### Input:\n\n\n"
            elif role == "assistant":
                prompt += f"### Response:\n{content}\n\n"
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            clean = re.sub(r'\n*---\n.*$', '', str(entry[1]), flags=re.DOTALL)
            prompt += f"### Instruction:\n{entry[0]}\n\n### Input:\n\n\n### Response:\n{clean}\n\n"
    prompt += f"### Instruction:\n{message}\n\n### Input:\n\n\n### Response:\n"
    return prompt

# ====================================================================
# 6. GERADOR DE RESPOSTA (Streaming + RAG + Metricas)
# ====================================================================
def generate(message, history, temperature, repeat_penalty, max_tokens):
    prompt = formatar_prompt(message, history)
    prompt_len = len(prompt)

    stop_seqs = [
        "### Instruction:", "### Input:",
        "<|endoftext|>", "<|im_end|>", "</s>",
        "Abaixo esta", "\n\n\n\n\n"
    ]

    stream = llm(
        prompt,
        max_tokens=int(max_tokens),
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        stop=stop_seqs,
        stream=True
    )

    resposta = ""
    start_time = time.time()
    token_count = 0

    for chunk in stream:
        token = chunk["choices"][0]["text"]
        resposta += token
        token_count += 1
        yield resposta

    elapsed = time.time() - start_time
    tps = token_count / elapsed if elapsed > 0 else 0

    # Aplicar RAG Injector DNA
    resposta_rag, hits, bytes_saved = rag_inject(resposta)

    # Metricas
    metrics = f"\n\n---\n"
    metrics += f"**🧬 CROM-IA V3.5b** | "
    metrics += f"⚡ {tps:.1f} t/s | "
    metrics += f"📝 {token_count} tokens | "
    metrics += f"⏱️ {elapsed:.1f}s | "
    metrics += f"🧵 {cores} threads"
    if hits > 0:
        metrics += f" | 🔬 RAG: {hits} hits ({bytes_saved} bytes expandidos)"

    if hits > 0:
        yield resposta_rag + metrics
    else:
        yield resposta + metrics

# ====================================================================
# 7. INTERFACE VISUAL PREMIUM (Gradio)
# ====================================================================
GITHUB_URL = "https://github.com/MrJc01/crompressor-ia"
HF_MODEL_URL = "https://huggingface.co/CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic"
HF_PAPER_URL = "https://huggingface.co/spaces/CromIA/Rosa-Chat-V3.5"

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #1e293b;
    --accent-blue: #38bdf8;
    --accent-cyan: #22d3ee;
    --accent-green: #34d399;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --glow: 0 0 20px rgba(56, 189, 248, 0.15);
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Header */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    margin-bottom: 1rem;
    text-align: center;
    box-shadow: var(--glow);
    position: relative;
    overflow: hidden;
}

.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(56, 189, 248, 0.05) 0%, transparent 50%);
    animation: pulse-glow 8s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.header-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    position: relative;
}

.header-sub {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0.3rem 0;
    position: relative;
}

.header-badges {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    position: relative;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: all 0.2s ease;
}

.badge:hover { transform: translateY(-1px); }

.badge-blue {
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent-blue);
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.badge-green {
    background: rgba(52, 211, 153, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.badge-cyan {
    background: rgba(34, 211, 238, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(34, 211, 238, 0.3);
}

/* Stats bar */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 0.7rem;
    text-align: center;
}

.stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-blue);
    font-family: 'JetBrains Mono', monospace;
}

.stat-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
}

/* Chatbot */
.chatbot {
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
}
"""

header_html = f"""
<div class="header-banner">
    <div class="header-title">🧬 Rosa — CROM-IA V3.5b</div>
    <div class="header-sub">Motor de IA Organica Comprimida | 1.5B Params | Qwen2.5 Fine-Tuned (117k amostras PT-BR)</div>
    <div class="header-sub" style="font-size: 0.8rem; color: #64748b;">Rodando nativamente em CPU — Sem GPU | Compressao Termodinamica DNA O(1)</div>
    <div class="header-badges">
        <a href="{GITHUB_URL}" target="_blank" class="badge badge-green">⚙️ GitHub</a>
        <a href="{HF_MODEL_URL}" target="_blank" class="badge badge-blue">🤗 Modelo GGUF</a>
        <span class="badge badge-cyan">🧵 {cores} Threads</span>
        <span class="badge badge-cyan">🔬 {len(dna_entries)} Ponteiros DNA</span>
    </div>
</div>
"""

stats_html = f"""
<div class="stats-bar">
    <div class="stat-card">
        <div class="stat-value">1.5B</div>
        <div class="stat-label">Parametros</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">117k</div>
        <div class="stat-label">Amostras Treino</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">Q4_K_M</div>
        <div class="stat-label">Quantizacao</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{len(dna_entries)}</div>
        <div class="stat-label">Codebook DNA</div>
    </div>
</div>
"""

with gr.Blocks(css=custom_css, title="Rosa - CROM-IA V3.5b") as demo:
    gr.HTML(header_html)
    gr.HTML(stats_html)

    chatbot = gr.ChatInterface(
        generate,
        chatbot=gr.Chatbot(height=480),
        additional_inputs=[
            gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=0.1,
                      label="🔥 Temperatura (Criatividade)"),
            gr.Slider(minimum=1.0, maximum=2.0, step=0.05, value=1.15,
                      label="🔁 Penalidade de Repeticao"),
            gr.Slider(minimum=256, maximum=4096, step=256, value=2048,
                      label="📏 Maximo de Tokens"),
        ],
    )

    gr.HTML("""
    <div style="text-align:center; padding: 1rem; color: #475569; font-size: 0.8rem;">
        CROM-IA &copy; 2026 — Arquitetura Sub-Simbolica Termodinamica para Edge Devices<br>
        <a href="https://github.com/MrJc01/crompressor-ia" target="_blank" style="color: #38bdf8;">GitHub</a> ·
        <a href="https://huggingface.co/CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic" target="_blank" style="color: #38bdf8;">HuggingFace</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
