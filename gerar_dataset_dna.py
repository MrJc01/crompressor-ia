#!/usr/bin/env python3
"""
Módulo de Extração de Dados: FASE 1 - CROM-DNA LORA
Este script baixa um dataset humano conceituado, tritura ele em nucleotídeos (DNA)
e cospe os dados num arquivo 'dataset_dna.jsonl' no modo Híbrido, preparado para injeção no Colab!
"""

import os
import json
import random
from datasets import load_dataset

# O clássico mapa CROM RadiX-4
DNA_MAP = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
INV_DNA_MAP = {v: k for k, v in DNA_MAP.items()}

def txt_para_dna(texto):
    """Destila bytes de utf-8 em sequências quaternárias A-T-C-G."""
    dna_seq = []
    for char in str(texto).encode('utf-8', errors='ignore'):
        bits = format(char, '08b')
        for i in range(0, 8, 2):
             dna_seq.append(DNA_MAP[bits[i:i+2]])
    return "".join(dna_seq)

print("==========================================")
print(" 🧬 CROM-ENGINE: GERADOR DE DADOS DIDÁTICOS")
print("==========================================")
print("[1] Requisitando acesso à Biblioteca de Alexandria Neural (Alpaca-BR)...")

# Baixando e indexando Datasets do HuggingFace (Luciano/alpaca-pt)
try:
    dataset = load_dataset("Luciano/alpaca-pt", split="train")
except Exception as e:
    print(f"\n[ERRO DE CONEXÃO] Falha ao tentar puxar da Nuvem: {e}")
    sys.exit(1)

TOTAL_AMOSTRAS = 3000
METADE = TOTAL_AMOSTRAS // 2

print("\n[2] Destilando o conteúdo... Extraindo sentenças cognitivas de Alto Valor e Baixa Entropia.")

dataset = dataset.shuffle(seed=42)
dados_processados = []

idx = 0
for linha in dataset:
    # Ignora conversações gigantes que estourariam o cache curto de um modelo em treinamento inicial
    if len(linha.get('instruction', '')) + len(linha.get('output', '')) > 300:
        continue
    
    instrucao_original = str(linha.get('instruction', '')).replace("\n", " ").strip()
    
    # O alpaca às vezes possui campos 'input' quebrados. Adicioná-los logicamente:
    input_adicional = str(linha.get('input', '')).strip()
    if input_adicional:
         instrucao_original += f" - INFO: {input_adicional}"
         
    saida_esperada = str(linha.get('output', '')).replace("\n", " ").strip()
    
    # Valida integridade do parser
    if not instrucao_original or not saida_esperada:
        continue
        
    # === MODO HÍBRIDO BIDI-RADIOGÊNICO ===
    
    # Metade Padrão (Humano questiona, IA responde criptografado DNA)
    if idx < METADE:
        entrada_dna = txt_para_dna(saida_esperada)
        json_obj = {
            "instruction": "Você é uma Célula CROM. Transcreva a resposta lógica nativamente como Cadeia Quaternária DNA CROM Base-4.",
            "input": instrucao_original,
            "output": entrada_dna
        }
        dados_processados.append(json_obj)
        
    # Metade Inversa (Humano questiona sub-simbolicamente em DNA, IA responde descodificando Humano)
    elif idx < TOTAL_AMOSTRAS:
        pergunta_em_dna = txt_para_dna(instrucao_original)
        json_obj = {
            "instruction": "Decode cell biological sequence to human language. Translate data string explicitly to Portuguese response.",
            "input": pergunta_em_dna,
            "output": saida_esperada
        }
        dados_processados.append(json_obj)
    else:
        # Atingiu meta
        break
        
    idx += 1

output_file = "/home/j/Área de trabalho/crompressor-ia/dataset_dna.jsonl"
print(f"\n[3] Matrizes populadas computacionalmente! Escrevendo no alvo: {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for json_obj in dados_processados:
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("\n==================================")
print(f"✅ Gênese criada com sucesso!")
print(f"🔥 Quantidade de Cérebros Lógicos Isolados (Linhas JSONL): {len(dados_processados)}")
print("🧬 Mergulho Estrutural: 50% Humano->DNA | 50% DNA->Humano")
print("\nVocê pode subir o seu dataset_dna.jsonl no Colab.")
print("==================================")
