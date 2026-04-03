#!/usr/bin/env python3
"""
🧬 CROM-IA V2: Terminal SRE Supremacia (Motor Nativo Híbrido Python/C++)
Tradução O(1) de Tokens DNA Pós-Geração.
"""
import os
import sys
import time
import json
import subprocess
import re

LLAMA_CLI = "/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

try:
    MODEL_PATH = sys.argv[1]   
    CODEBOOK_PATH = sys.argv[2]
    TAXA = sys.argv[3]
except IndexError:
    print("Faltam argumentos para o inicializador SRE.")
    sys.exit(1)

with open(CODEBOOK_PATH, 'r', encoding='utf-8') as f:
    codebook = json.load(f)
entries = codebook.get("entries", {})
escape_pfx = codebook.get("escape_prefix", "@@")

print("\n" + "="*60)
print(f" 🧠 CROM-IA TERMINAL SRE (Decoder O(1) Batch - Taxa {TAXA})")
print("="*60)

system_prompt = f"Você é um compressor CROM DNA (taxa {TAXA.replace('x', ':')}). Comprima a resposta usando códigos do codebook semântico DNA. Use prefixo {escape_pfx} para palavras sem código. Responda APENAS com códigos DNA ou escapes permitidos."

historico = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
print("\n\033[1;33m[!] TTY Protegido: Digite 'sair' ou pressione Ctrl+C para encerrar com segurança.\033[0m\n")

while True:
    try:
        user_input = input("\033[0;36m👤 Você:\033[0m ").strip()
        if not user_input: continue
        if user_input.lower() in ['sair', 'exit', 'quit']:
            sys.exit(0)

        historico += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        cmd = [
            LLAMA_CLI,
            "-m", MODEL_PATH,
            "-c", "1024",
            "-t", "4",
            "-n", "128",            
            "--temp", "0.2",
            "--repeat-penalty", "1.18",
            "-p", historico
        ]

        print("\033[0;32m🤖 CROM:\033[0m ", end="", flush=True)

        t0 = time.time()
        
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL # Mata banners sujos da LLMA CLI
        )
        
        # Como o llama C++ da build 8589 tá agarrando reverse-prompt/chat mode teimoso:
        # A gente envia o \n pra fechar prompts pendentes se houver, e sinaliza EOF pra ele matar a sub-instância.
        proc.stdin.write(b"\n/exit\n")
        proc.stdin.close()
        
        # Lê tudo sem deadlocks de pipe aberto!
        resposta_bruta = proc.stdout.read().decode('utf-8', 'ignore')
        proc.wait()
        
        # Filtro de Metadados UI vazado do Prompt que o python C++ pode ter regurgitado
        partes = resposta_bruta.split("<|im_start|>assistant\n")
        dna_puro = partes[-1].split("<|im_end|>")[0].strip() if len(partes) > 1 else resposta_bruta.replace(historico, "").strip()

        # Limpador ANSI e Lixos Llama CLI
        dna_puro = re.sub(r'>.*|\[.*\]|/exit', '', dna_puro) # Tira ">>>" ou "[ prompt ]" se vazaram via stdout da flag -p
        dna_puro = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', dna_puro) 
        
        tokens = dna_puro.strip().split()
        saida_final = []
        for t in tokens:
            if t.startswith(escape_pfx):
                saida_final.append(t[len(escape_pfx):])
            elif t in entries:
                saida_final.append(entries[t]["text"])
            else:
                saida_final.append(t)
                
        texto_humano = " ".join(saida_final).replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
        
        for char in texto_humano:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.005)
            
        tf = time.time() - t0
        historico += dna_puro + "<|im_end|>\n"
        print(f"\n\033[1;30m[ ⚡ Turno em {tf:.2f}s | Real: {len(saida_final)} words (Genêmica: {len(tokens)} tokens) ]\033[0m\n")

    except KeyboardInterrupt:
        print("\n\n[!] SRE Terminated.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERRO FATAL CROM] {e}")
        sys.exit(1)
