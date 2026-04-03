#!/usr/bin/env python3
"""
CROM-DNA Laboratório: Ribossomo Matemático Sub-Simbólico
Este script provará o conceito da ingestão de entropia via Base-4 (A, T, C, G)
"""

import os
import sys
import subprocess
import time

BINARIO_NATIVO = "/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"
MODELO = "/home/j/Área de trabalho/crompressor-ia/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if not os.path.exists(BINARIO_NATIVO):
    print(f"[ERRO FATAL] Motor Nativo ausente: {BINARIO_NATIVO}")
    sys.exit(1)

# ENCODING ZERO-ENTROPY RADIX-4
DNA_MAP = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
INV_DNA_MAP = {v: k for k, v in DNA_MAP.items()}

def txt_para_dna(texto):
    dna_seq = []
    # Converte cada caractere do UTF-8 para uma série de bases
    for char in texto.encode('utf-8'):
        bits = format(char, '08b')
        # Pega a cada 2 bits e mapeia para nucleotídeo
        for i in range(0, 8, 2):
             dna_seq.append(DNA_MAP[bits[i:i+2]])
    return "".join(dna_seq)

def dna_para_txt(dna_str):
    bits = ""
    # Remove lixo da IA garantindo ler apenas nucleotídeos limpos
    limpo = "".join([c for c in dna_str.upper() if c in INV_DNA_MAP])
    
    for base in limpo:
        bits += INV_DNA_MAP[base]
        
    bytes_arr = []
    for i in range(0, len(bits), 8):
        octeto = bits[i:i+8]
        if len(octeto) == 8:
            bytes_arr.append(int(octeto, 2))
            
    try:
        return bytes(bytes_arr).decode('utf-8', errors='ignore')
    except:
        return "<Decodificação Entrópica Falhou>"

print("==================================================")
print(" 🧬 LABORATÓRIO CROM-DNA: Modo Edge Biológico")
print("==================================================")
user_txt = input("\n[1] Digite a injeção (Texto): ").strip()

if not user_txt:
    user_txt = "ola"

print("\n[2] Acionando Enzima Tradutora Radix-4...")
dna_payload = txt_para_dna(user_txt)
print(f" -> Carga de DNA Entrópica: {dna_payload}")
print(f" -> Tamanho Original: {len(user_txt)} un | Tamanho DNA: {len(dna_payload)} un")\

# A Engine LLM precisaria de um treinamento LoRA pesado para gerar cadeias compatíveis na hora. 
# Aqui estamos validando como a IA vai processar dados alienínegas/binários (Zero-Shot) na base estrutural.
prompt_sistema = "You are a biologic translator cell."
prompt_completo = f"Decode context if you can. Answer concisely. Biological data payload: {dna_payload}"

print("\n[3] Injetando carga DNA no Cérebro Nativo (Llama C++)...")

t0 = time.time()
comando = [
    BINARIO_NATIVO,
    "-m", MODELO,
    "--threads", "2",
    "-c", "1024",
    "-n", "128",            # Resposta curta para análise
    "--temp", "0.2",
    "-p", f"<|im_start|>system\n{prompt_sistema}<|im_end|>\n<|im_start|>user\n{prompt_completo}<|im_end|>\n<|im_start|>assistant\n",
    "--log-disable"
]

try:
    proc = subprocess.run(comando, capture_output=True, text=True)
    tf = time.time() - t0
    saida = proc.stdout
    # Llama-cli outputs the whole prompt along with response if not tuned perfectly, let's extract the assistant part
    if "<|im_start|>assistant\n" in saida:
        saida_ia = saida.split("<|im_start|>assistant\n")[-1].strip()
    else:
        saida_ia = saida.strip()

    print(f"\n======== RESULTADO DA INFERÊNCIA ({tf:.2f}s) ========")
    print(f"🧬 Saída Neural Bruta:\n{saida_ia}")
    print("====================================================")
    
    # Tenta decodificar caso a IA tenha tentado imitar cadeias base-4
    print("\n[4] Retranscrevendo Mutação de Volta para Humano...")
    texto_traduzido = dna_para_txt(saida_ia)
    if texto_traduzido.strip():
        print(f"🗣️  Legenda Traduzida do DNA Retornado: {texto_traduzido}")
    else:
        print("🗣️  A IA não conseguiu estruturar as bases radiciais (Necessita LoRA Fine-Tuning).")
        
except Exception as e:
    print(f"[ERRO] Llama.cpp falhou na injeção: {e}")

print("\nExperimento concluído.")
