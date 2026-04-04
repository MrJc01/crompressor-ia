#!/usr/bin/env python3
"""
🧬 CROM-IA V3: Hook Pipe Nativo (Native RAG Injector)
Lê do /dev/stdin char por char em tempo real sem flush buffer longo.
Detecta a sequência @@XX e injeta memória O(1) termodinâmica em VERDE.
"""

import sys
import json
import os

def carregar_codebook():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "macro_codebook_v3.json")
    if not os.path.exists(input_file):
        sys.stderr.write(f"\\n❌ Codebook não encontrado: {input_file}\\n")
        sys.exit(1)
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_pipe():
    codebook = carregar_codebook()
    entries = codebook.get("entries", {})
    
    # Cores
    COLOR_RAG = "\033[92m" # Verde (Injeção Artificial Direta)
    COLOR_RESET = "\033[0m"
    COLOR_SYSTEM = "\033[93m" # Amarelo
    
    sys.stderr.write(f"{COLOR_SYSTEM}\n[🧬 CROM-IA V3 RAG ENGINE] {len(entries)} Macros de Código Carregadas!{COLOR_RESET}\n")
    sys.stderr.write(f"{COLOR_SYSTEM}[🔒 Interceptor de Output Ativado | Aguardando LLM...]{COLOR_RESET}\n")
    
    estado_escape = False
    buffer_dna = ""
    
    try:
        while True:
            # Lê exatos 1 char direto do stream do llama.cpp
            char = sys.stdin.read(1)
            
            # EOF
            if not char:
                break
                
            # --- State Machine de Captura Rápida --- #
            if char == '@' and not estado_escape:
                # Encontrou provável marcador. Precisamos ler o próximo pra confirmar.
                proximo = sys.stdin.read(1)
                if not proximo: # EOF abrupto
                    sys.stdout.write(char)
                    break
                    
                if proximo == '@':
                    estado_escape = True
                    buffer_dna = ""
                    continue # Aguarda os proximos 2 chars de DNA
                else:
                    # Alarme falso, o modelo apenas usou a arroba numa string normal
                    sys.stdout.write(char + proximo)
                    sys.stdout.flush()
                    continue
                    
            if estado_escape:
                # O LLM piscou e já ejetou o pointer (ex 'A')
                buffer_dna += char
                if len(buffer_dna) == 2: # Ex: "AA" (Base 16 ou Base 64 chaves)
                    if buffer_dna in entries:
                        # RAG EXPLOSION INVASIVO!
                        sys.stdout.write(COLOR_RAG + entries[buffer_dna]["text"] + COLOR_RESET)
                    else:
                        # LLM alucinou um pointer quebrado.
                        sys.stdout.write(f"@@{buffer_dna}")
                        
                    sys.stdout.flush()
                    estado_escape = False # Reseta a máquina
                continue
                
            # Se não está em escape nem flagrando arroba, printa cru (0ms delay)
            sys.stdout.write(char)
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        sys.stderr.write(f"\\n{COLOR_SYSTEM}[⚠️ RAG Injector Abortado]{COLOR_RESET}\\n")

if __name__ == "__main__":
    run_pipe()
