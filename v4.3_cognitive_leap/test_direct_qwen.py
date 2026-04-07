import os
from llama_cpp import Llama

# 🧬 CROM-IA v4.3: TESTE DE DECORADAÇÃO DNA (DIRETO)
MODEL_PATH = "/home/j/Área de trabalho/crompressor-ia/models/CROM-IA_v4.3_Qwen3.5-2B.gguf"

print(f"🚀 Carregando Chassis: {os.path.basename(MODEL_PATH)}")

# Configurações de Invocação
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# Template ChatML (Soberania CROM)
prompt = "<|im_start|>system\nVocê é uma Célula CROM. Decodifique a sequência DNA CROM Base-4 para Português.<|im_end|>\n<|im_start|>user\nDecodifique: ATCGTAGC<|im_end|>\n<|im_start|>assistant\n"

print("\n--- 🧬 ENTRADA DNA: ATCGTAGC ---")
print("📡 Processando Salto Cognitivo...\n")

output = llm(
    prompt,
    max_tokens=64,
    stop=["<|im_end|>"],
    echo=False,
    temperature=0.1
)

resposta = output["choices"][0]["text"]
print(f"✅ RESPOSTA CROM: {resposta.strip()}")
print("\n--- TESTE FINALIZADO ---")
