#!/usr/bin/env python3
import sys
import subprocess
import json
import os
import re

def carregar_codebook(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        codebook = json.load(f)
    return codebook.get("entries", {}), codebook.get("escape_prefix", "@@")

def limpa_ansi(texto):
    """Remove códigos ANSI de cor caso o llama os gere na saída."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', texto)

def run_chat_loop(llama_cli, modelo, ctx, threads, system_prompt, codebook_path):
    entries, escape = carregar_codebook(codebook_path)
    historico = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    
    print("\n[!] Loop Conversacional CROM Híbrido Ativado (Interação Blindada)")
    print("Digite 'sair' ou pressione Ctrl+C para encerrar.\n")
    
    while True:
        try:
            pergunta = input("\033[0;36m👤 Você:\033[0m ").strip()
            if not pergunta: continue
            if pergunta.lower() in ["sair", "exit", "quit", "/exit"]:
                print("Saindo...")
                break
                
            # Atualiza histórico para QWEN/TinyLLaMa ChatML Format
            historico += f"<|im_start|>user\n{pergunta}<|im_end|>\n<|im_start|>assistant\n"
            
            # Chama a Llama em modo raw/batch. O "--log-disable" blinda a tela!
            cmd = [
                llama_cli, "-m", modelo, "-c", str(ctx), "-t", str(threads),
                "-n", "128", "--temp", "0.2", "--repeat-penalty", "1.18",
                "--log-disable", "-p", historico
            ]
            
            print("\033[0;32m🤖 CROM:\033[0m ", end="", flush=True)
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # Ignora meta dados TTY do stdout/stderr
            )
            
            # Pega saida bash, ignora o proprio prompt inserido
            out, _ = proc.communicate()
            resposta_raw = out.decode("utf-8", "ignore")
            
            # A Llama cospe o prompt inteiro de novo no modo '-p'. Precisamos fatiar apenas a resposta nova!
            if "<|im_start|>assistant\n" in resposta_raw:
                # Pega só o que ele respondeu por ultimo
                partes = resposta_raw.split("<|im_start|>assistant\n")
                nova_resposta = partes[-1].split("<|im_end|>")[0].strip()
            else:
                # Fallback caso a modelagem corte as tags
                nova_resposta = resposta_raw.replace(historico, "").strip()
            
            # Decodificador em Lote Super Rápido (1x5)
            nova_resposta_limpa = limpa_ansi(nova_resposta)
            tokens = nova_resposta_limpa.split()
            saida_final = []
            
            for t in tokens:
                if t.startswith(escape):
                    saida_final.append(t[len(escape):]) # É humano com @@
                elif t in entries:
                    saida_final.append(entries[t]["text"]) # É DNA CROM
                else:
                    saida_final.append(t) # Letras soltas ou pontuação
                    
            frase = " ".join(saida_final)
            print(f"{frase}\n")
            
            # Alimenta o loop para a proxima rodada
            historico += nova_resposta + "<|im_end|>\n"
            
        except KeyboardInterrupt:
            print("\n[!] Conversa abortada.")
            break

if __name__ == "__main__":
    run_chat_loop(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
