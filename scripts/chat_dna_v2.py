#!/usr/bin/env python3
"""
🧬 CROM-IA V2: Chat DNA Terminal — Motor AVX Puro + Decoder O(1) Streaming
Usa pexpect para capturar TTY do llama-cli com decodificação em tempo real.
"""
import os
import sys
import json
import re
import time
import pexpect

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CLI = "/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

TAXA = sys.argv[1] if len(sys.argv) > 1 else "1x5"
MODO = sys.argv[2] if len(sys.argv) > 2 else "fixo"

MODELO = os.path.join(BASE_DIR, "models", f"crom-dna-{TAXA}-{MODO}.gguf")
CODEBOOK_PATH = os.path.join(BASE_DIR, "codebooks", f"codebook_{TAXA}_{MODO}.json")

for f in [LLAMA_CLI, MODELO, CODEBOOK_PATH]:
    if not os.path.exists(f):
        print(f"❌ Não encontrado: {f}")
        sys.exit(1)

with open(CODEBOOK_PATH, 'r', encoding='utf-8') as f:
    codebook = json.load(f)
entries = codebook.get("entries", {})
escape_pfx = codebook.get("escape_prefix", "@@")

def decodifica_token(token):
    """Decodifica um único token DNA → texto humano."""
    token = token.strip()
    if not token:
        return None, "vazio"
    if token.startswith(escape_pfx):
        return token[len(escape_pfx):], "escape"
    if token in entries:
        return entries[token]["text"], "codebook"
    # Ignora lixo TTY curto
    if len(token) <= 1 and not token.isalnum():
        return None, "lixo"
    return token, "desconhecido"

def stream_turno(pergunta, system_prompt):
    """Executa llama-cli em single-turn e faz streaming da decodificação."""
    cmd = (
        f'"{LLAMA_CLI}" '
        f'-m "{MODELO}" '
        f'-c 1024 -t 4 -n 128 '
        f'--temp 0.2 --repeat-penalty 1.18 '
        f'--single-turn '
        f'-sys "{system_prompt}" '
        f'-p "{pergunta}"'
    )
    
    child = pexpect.spawn('/bin/bash', ['-c', cmd], encoding='utf-8', timeout=120)
    
    # Estados: LOADING, BANNER, PROMPT_ECHO, RESPONSE, TIMING, DONE
    estado = "LOADING"
    buffer_token = ""
    timing_info = ""
    stats = {"total": 0, "codebook": 0, "escape": 0, "desconhecido": 0}
    t_primeiro = None
    
    sys.stdout.write("\033[0;32m🤖 CROM:\033[0m ")
    sys.stdout.flush()
    
    try:
        while True:
            try:
                char = child.read_nonblocking(1, timeout=60)
            except pexpect.TIMEOUT:
                break
            except pexpect.EOF:
                break
            
            # Ignora carriage returns
            if char == '\r':
                continue
            
            # Backspace: apaga último char do buffer (simula terminal real — mata spinner)
            if char == '\x08':
                if buffer_token:
                    buffer_token = buffer_token[:-1]
                continue
            
            if estado == "LOADING":
                buffer_token += char
                if buffer_token.endswith("\n> "):
                    estado = "PROMPT_ECHO"
                    buffer_token = ""
                if len(buffer_token) > 10000:
                    buffer_token = buffer_token[-100:]
                continue
            
            elif estado == "PROMPT_ECHO":
                if char == '\n':
                    estado = "RESPONSE"
                    buffer_token = ""
                continue
            
            elif estado == "RESPONSE":
                if char == '\n':
                    if buffer_token.strip():
                        stripped = buffer_token.strip()
                        # Detecta linha de timing: contém "Prompt:" ou "t/s"
                        if "Prompt:" in stripped or "t/s" in stripped:
                            timing_info = stripped
                            estado = "DONE"
                            buffer_token = ""
                            continue
                        elif stripped == "Exiting...":
                            estado = "DONE"
                            buffer_token = ""
                            continue
                        else:
                            texto, tipo = decodifica_token(stripped)
                            if texto:
                                if t_primeiro is None:
                                    t_primeiro = time.time()
                                stats["total"] += 1
                                stats[tipo] = stats.get(tipo, 0) + 1
                                sys.stdout.write(texto + " ")
                                sys.stdout.flush()
                    buffer_token = ""
                    continue
                
                elif char == ' ':
                    if buffer_token.strip():
                        stripped = buffer_token.strip()
                        # Checa timing inline (ex: "[ Prompt: 41,5 t/s ...")
                        if "Prompt:" in stripped or "t/s" in stripped:
                            timing_info = stripped
                            # Continua acumulando a linha de timing
                            buffer_token += char
                            continue
                        texto, tipo = decodifica_token(stripped)
                        if texto:
                            if t_primeiro is None:
                                t_primeiro = time.time()
                            stats["total"] += 1
                            stats[tipo] = stats.get(tipo, 0) + 1
                            sys.stdout.write(texto + " ")
                            sys.stdout.flush()
                    buffer_token = ""
                else:
                    buffer_token += char
            
            elif estado == "DONE":
                if char == '\n':
                    if buffer_token.strip():
                        if "t/s" in buffer_token:
                            timing_info = buffer_token.strip()
                    buffer_token = ""
                else:
                    buffer_token += char
    
    except Exception:
        pass
    
    # Flush final
    if buffer_token.strip() and estado == "RESPONSE":
        texto, tipo = decodifica_token(buffer_token.strip())
        if texto:
            stats["total"] += 1
            stats[tipo] = stats.get(tipo, 0) + 1
            sys.stdout.write(texto + " ")
    
    child.close()
    return stats, timing_info

# === INTERFACE ===
print("")
print("╔══════════════════════════════════════════════════════════╗")
print(f"║  🧬 CROM-IA Chat DNA (Taxa {TAXA}) — Motor AVX Puro     ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  Modelo   : crom-dna-{TAXA}-{MODO}.gguf")
print(f"║  Codebook : codebook_{TAXA}_{MODO}.json ({len(entries)} entradas)")
print("╚══════════════════════════════════════════════════════════╝")
print("")
print("[!] Digite 'sair' para encerrar.")
print("")

SYSTEM_PROMPT = (
    f"Você é um compressor CROM DNA (taxa {TAXA.replace('x', ':')})."
    f" Comprima a resposta usando códigos do codebook semântico DNA."
    f" Use prefixo {escape_pfx} para palavras sem código."
    f" Responda APENAS com códigos DNA."
)

while True:
    try:
        user_input = input("\033[0;36m👤 Você:\033[0m ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Saindo...")
            break

        t0 = time.time()
        stats, timing = stream_turno(user_input, SYSTEM_PROMPT)
        tf = time.time() - t0
        
        # Métricas na linha de baixo
        hit_rate = (stats.get("codebook", 0) / max(stats["total"], 1)) * 100
        print(f"\n\033[1;30m   [ DNA: {stats['total']} tokens | Codebook: {stats.get('codebook',0)} ({hit_rate:.0f}%) | Escape: {stats.get('escape',0)} | Tempo: {tf:.1f}s | {timing} ]\033[0m")
        print("")

    except KeyboardInterrupt:
        print("\n\nSaindo...")
        break
    except Exception as e:
        print(f"\n[ERRO] {e}")

print("Sessão encerrada.")
