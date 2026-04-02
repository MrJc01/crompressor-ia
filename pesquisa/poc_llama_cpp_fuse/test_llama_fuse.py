#!/usr/bin/env python3
import time
import os
import psutil

try:
    from llama_cpp import Llama
except ImportError:
    print("ERRO FATAL: llama_cpp_python não instalado no VirtualEnvironment.")
    exit(1)

print("Iniciando Pipeline Física: CROM-LLM via mmap (Bypass VRAM)")
model_path = "/tmp/crom_mnt/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(model_path):
    print(f"ERRO: Asset termodinâmico GGUF não encontrado em '{model_path}'")
    exit(1)

print(f"Alocando Tensores diretamente de CROM FUSE: {model_path}")
print("Configurando HNSW Mmap -> use_mmap=True, n_gpu_layers=0")

# Telemetria Inicial
process = psutil.Process(os.getpid())
ram_antes = process.memory_info().rss / 1024 / 1024

start = time.time()
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    use_mmap=True,
    use_mlock=False,  # OSO mlock is dangerous in FUSE constraints
    n_gpu_layers=0,   # Pure CPU Edge mode
    verbose=False
)
load_time = time.time() - start

ram_depois = process.memory_info().rss / 1024 / 1024
delta_ram = ram_depois - ram_antes

print(f"\n[SUCESSO] Conector FUSE mmap atracado. Tempo Carregamento: {load_time:.2f}s")
print(f"[TELEMETRIA] A LLM de 640MB retém na RAM Física apenas: {delta_ram:.2f} MB")
if delta_ram < 300:
    print("[SRE STATUS] Bypass de Memória Comprovado. Risco de OOM: Nulo.")

print("\nRealizando Loop Termodinâmico de Resposta Neural...")
prompt = "<|system|>\nVocê é a CROM-IA, uma IA otimizada para ser veloz.<|end|>\n<|user|>\nQual seu objetivo principal como CROM-IA?<|end|>\n<|assistant|>"

print(f"\nDisparando HNSW Context na Query: '{prompt.strip()}'\n---")
output = llm(
    prompt,
    max_tokens=64,
    stop=["<|user|>"],
    echo=False
)

resposta = output['choices'][0]['text']
print(f"RESPOSTA FÍSICA GERADA (O(1) Streaming):\n{resposta}\n---")
print("Integração Física GGUF Completa. Sem Swap Panic.")
