#!/usr/bin/env python3
"""
Módulo de Extração de Dados: FASE 1 - CROM-DNA LORA
Gera dataset_dna.jsonl no modo Híbrido Trifásico:
  33% Humano → DNA (codificação)
  33% DNA → Humano (decodificação)
  33% Contexto + DNA (instrução mista)

Preparado para upload direto ao Google Colab + Unsloth LoRA.
"""

import os
import sys
import json
import math
import argparse
from collections import Counter

# ============================
# CROM Radix-4 DNA Engine
# ============================
DNA_MAP = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
INV_DNA_MAP = {v: k for k, v in DNA_MAP.items()}

def txt_para_dna(texto):
    """Destila bytes UTF-8 em sequências quaternárias A-T-C-G."""
    dna_seq = []
    for char in str(texto).encode('utf-8', errors='ignore'):
        bits = format(char, '08b')
        for i in range(0, 8, 2):
            dna_seq.append(DNA_MAP[bits[i:i+2]])
    return "".join(dna_seq)

def calcular_entropia(texto):
    """Calcula entropia de Shannon H para uma string."""
    if not texto:
        return 0.0
    contagem = Counter(texto.encode('utf-8', errors='ignore'))
    total = sum(contagem.values())
    entropia = 0.0
    for count in contagem.values():
        if count > 0:
            p = count / total
            entropia -= p * math.log2(p)
    return entropia

def main():
    parser = argparse.ArgumentParser(description="Gerador de Dataset DNA CROM para LoRA")
    parser.add_argument("--amostras", type=int, default=10000, help="Total de amostras (default: 10000)")
    parser.add_argument("--output", type=str, default=None, help="Caminho do arquivo de saída")
    parser.add_argument("--max-entropia", type=float, default=7.0, help="Limiar máximo de entropia Shannon (default: 7.0)")
    parser.add_argument("--max-chars", type=int, default=300, help="Tamanho máximo instrução+output (default: 300)")
    args = parser.parse_args()

    if args.output is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(base_dir, "dataset_dna.jsonl")

    print("=" * 55)
    print(" 🧬 CROM-ENGINE: GERADOR DE DADOS TRIFÁSICO DNA")
    print("=" * 55)
    print(f" Amostras alvo  : {args.amostras}")
    print(f" Entropia máx   : {args.max_entropia}")
    print(f" Chars máx      : {args.max_chars}")
    print(f" Saída          : {args.output}")
    print("=" * 55)

    print("\n[1] Baixando dataset Alpaca-PT do HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-portuguese", split="train")
    except Exception as e:
        print(f"\n[ERRO] Falha ao baixar dataset: {e}")
        sys.exit(1)

    TERCO = args.amostras // 3

    print(f"\n[2] Processando {args.amostras} amostras em 3 fases...")
    print(f"    Fase A: {TERCO} Humano → DNA")
    print(f"    Fase B: {TERCO} DNA → Humano")
    print(f"    Fase C: {args.amostras - 2*TERCO} Contexto Misto")

    dataset = dataset.shuffle(seed=42)
    dados = []
    stats = {"descartados_entropia": 0, "descartados_tamanho": 0, "descartados_vazio": 0}

    idx = 0
    for linha in dataset:
        if len(dados) >= args.amostras:
            break

        convs = linha.get('conversations', [])
        if not convs or len(convs) < 2:
            stats["descartados_vazio"] += 1
            continue

        instrucao = ""
        saida = ""
        
        # Procura a primeira interação human/gpt
        for c in convs:
            if c.get("from") == "human" and not instrucao:
                instrucao = str(c.get("value", "")).replace("\n", " ").strip()
            elif c.get("from") == "gpt" and not saida:
                saida = str(c.get("value", "")).replace("\n", " ").strip()

        # Validação: campos vazios
        if not instrucao or not saida:
            stats["descartados_vazio"] += 1
            continue

        # Validação: tamanho
        if len(instrucao) + len(saida) > args.max_chars:
            stats["descartados_tamanho"] += 1
            continue

        # Validação: entropia Shannon
        h_instrucao = calcular_entropia(instrucao)
        h_saida = calcular_entropia(saida)
        if h_instrucao > args.max_entropia or h_saida > args.max_entropia:
            stats["descartados_entropia"] += 1
            continue

        fase = len(dados)

        # FASE A: Humano pergunta, IA responde em DNA
        if fase < TERCO:
            dna_saida = txt_para_dna(saida)
            obj = {
                "instruction": "Você é uma Célula CROM. Transcreva a resposta como Cadeia Quaternária DNA CROM Base-4 (usar apenas A, T, C, G).",
                "input": instrucao,
                "output": dna_saida
            }

        # FASE B: Humano pergunta em DNA, IA decodifica para humano
        elif fase < 2 * TERCO:
            dna_pergunta = txt_para_dna(instrucao)
            obj = {
                "instruction": "Decodifique a sequência biológica DNA CROM Base-4 para linguagem humana em Português.",
                "input": dna_pergunta,
                "output": saida
            }

        # FASE C: Contexto misto (instrução humana + snippet DNA na entrada)
        else:
            dna_snippet = txt_para_dna(saida[:50])
            obj = {
                "instruction": "Analise o contexto abaixo. A primeira parte é linguagem humana. A segunda é uma amostra DNA CROM Radix-4. Responda o que for pedido em Português claro.",
                "input": f"Contexto: {instrucao}\nAmostra DNA: {dna_snippet}",
                "output": saida
            }

        dados.append(obj)
        idx += 1

    # Escrever JSONL
    print(f"\n[3] Escrevendo {len(dados)} amostras em {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for obj in dados:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Estatísticas
    tamanhos = [len(json.dumps(d, ensure_ascii=False)) for d in dados]
    entropias = [calcular_entropia(d["output"]) for d in dados]
    arquivo_mb = os.path.getsize(args.output) / 1024 / 1024

    if len(dados) > 0:
        linha_media = sum(tamanhos) / len(tamanhos)
        entropia_media = sum(entropias) / len(entropias)
    else:
        linha_media = 0
        entropia_media = 0

    print("\n" + "=" * 55)
    print(" ✅ DATASET GERADO COM SUCESSO!")
    print("=" * 55)
    print(f" 📊 Amostras geradas      : {len(dados)}")
    print(f" 📁 Arquivo               : {args.output}")
    print(f" 💾 Tamanho               : {arquivo_mb:.1f} MB")
    print(f" 📏 Linha média           : {linha_media:.0f} chars")
    print(f" 🧬 Entropia média output : {entropia_media:.2f}")
    print(f" ❌ Descartados entropia  : {stats['descartados_entropia']}")
    print(f" ❌ Descartados tamanho   : {stats['descartados_tamanho']}")
    print(f" ❌ Descartados vazios    : {stats['descartados_vazio']}")
    print("=" * 55)
    print("\n🚀 Próximo passo: Suba dataset_dna.jsonl no Google Colab")
    print("   e execute o script de treinamento LoRA!")

if __name__ == "__main__":
    main()
