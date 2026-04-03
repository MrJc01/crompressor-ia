#!/usr/bin/env python3
"""
CROM-IA Terminal Mode - Chat Direto e Rápido.
Modo Streaming Ativado via Chat format nativo.
"""
import os
import sys
import time

try:
    from llama_cpp import Llama
    import psutil
except ImportError:
    print("[ERRO] Bibliotecas ausentes. Execute o ./iniciar_chat_real.sh primeiro ou ative o .venv e instale llama-cpp-python.")
    sys.exit(1)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

if not os.path.exists(MODEL_PATH):
    print(f"[ERRO] Modelo nao encontrado em: {MODEL_PATH}")
    sys.exit(1)

print("\n" + "="*50)
print(" 🧠 CROM-IA TERMINAL (Neurônio Primário)")
print("="*50)
print(" Carregando pesos neurais (Zero-Copy FUSE)...")

# Core System Identity
SYSTEM_PROMPT = (
    "You are CROM-IA, an AI assistant neural node. "
    "Respond concisely to the user in Portuguese. "
    "Do not hallucinate or loop spaces. Keep your answers straight to the point."
)

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        use_mmap=False, # <-- Desativado para evitar lentidão IOWait / PageFault
        use_mlock=False,
        verbose=False,
        chat_format="chatml" # Força o formato ChatML ideal para o TinyLlama
    )
except Exception as e:
    print(f"\n[ERRO FATAL] Falha ao carregar modelo: {e}")
    sys.exit(1)

proc = psutil.Process(os.getpid())
rss_mb = proc.memory_info().rss / 1024 / 1024

print(f" [OK] Cérebro Ativado! RSS: {rss_mb:.1f} MB")
print("="*50)
print(" Digite 'sair' para encerrar.")
print("="*50 + "\n")

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        user_input = input("\n👤 Você: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['sair', 'exit', 'quit']:
            break

        conversation.append({"role": "user", "content": user_input})
        if len(conversation) > 5:
            # Mantem o system prompt, remove o oldest turn (user + assistant = 2 items)
            conversation.pop(1)
            conversation.pop(1)

        print("\n🤖 CROM: ", end="", flush=True)
        
        t0 = time.time()
        tokens = 0
        response_text = ""
        first_token_time = None

        print(f"\n[DEBUG] Iniciando inferência... Avaliando prompt ({len(conversation)} turnos)")

        # Usando a API de Chat nativa com anti-hallucination param (repeat_penalty)
        stream = llm.create_chat_completion(
            messages=conversation,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.18,
            stream=True
        )

        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
                eval_time = first_token_time - t0
                print(f"[DEBUG] Prompt avaliado em {eval_time:.2f}s. Iniciando tokens...\n🤖 CROM: ", end="")

            if "content" in chunk["choices"][0]["delta"]:
                text_chunk = chunk["choices"][0]["delta"]["content"]
                response_text += text_chunk
                
                # Print current token and latency
                token_latency = time.time() - first_token_time if tokens == 0 else time.time() - last_token_time
                last_token_time = time.time()
                
                print(f"{text_chunk} (⚡{token_latency:.2f}s)", end="", flush=True)
                tokens += 1

        elapsed = time.time() - t0
        gen_time = elapsed - eval_time if first_token_time else 0
        speed = tokens / gen_time if gen_time > 0 else 0
        
        print(f"\n\n[📊 RELATÓRIO SRE]")
        print(f" - Tempo Avaliação do Prompt: {eval_time:.2f}s")
        print(f" - Tempo Geração ({tokens} tokens): {gen_time:.2f}s")
        print(f" - Velocidade Média: {speed:.2f} tokens/seg")
        print(f" - Memória Residente (RSS): {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
        
        conversation.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\n\n[!] Operação cancelada. Digite 'sair' para encerrar ou pergunte novamente.")
    except Exception as e:
        print(f"\n\n[ERRO] {e}")

print("\nTerminais nervosos desconectados. Até logo.")
