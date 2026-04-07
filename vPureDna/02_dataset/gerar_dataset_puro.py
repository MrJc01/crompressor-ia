#!/usr/bin/env python3
"""
🧬 vPureDna — Gerador de Dataset DNA (⌬)

Pega Alpaca-PT, comprime frases longas nas respostas usando ⌬IDs,
e gera dataset para treinar um modelo que APRENDE a usar ⌬.

O modelo usa tokenizer normal. ⌬ é parte do texto de treino.
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_encoder'))
from tokenizer_dna import DNACompressor

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def carregar_corpus():
    """Carrega Alpaca-PT ou fallback local."""
    print("  📥 Carregando corpus...")
    
    # Tentar fallback local primeiro (mais rápido)
    fallback = os.path.join(REPO_DIR, "dataset_dna_v43_test.jsonl")
    if os.path.exists(fallback):
        with open(fallback) as f:
            data = [json.loads(l) for l in f]
        print(f"  ✅ {len(data)} amostras do dataset local")
        return data, "local"
    
    # Tentar HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("FreedomIntelligence/alpaca-gpt4-portuguese", split="train")
        print(f"  ✅ {len(ds)} amostras do HuggingFace")
        return ds, "hf"
    except Exception as e:
        print(f"  ❌ Nenhuma fonte disponível: {e}")
        sys.exit(1)


def extrair_texto(item, source):
    """Extrai instrução e resposta."""
    if source == "hf" and 'conversations' in item:
        instrucao = saida = ""
        for c in item['conversations']:
            if c.get("from") == "human" and not instrucao:
                instrucao = str(c.get("value", "")).strip()
            elif c.get("from") == "gpt" and not saida:
                saida = str(c.get("value", "")).strip()
        return instrucao, saida
    
    instrucao = str(item.get('instruction', '') or '')
    inp = str(item.get('input', '') or '')
    if inp:
        instrucao = f"{instrucao} {inp}"
    saida = str(item.get('output', '') or '')
    return instrucao.strip(), saida.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="🧬 vPureDna Dataset Generator (⌬)")
    parser.add_argument("--max-amostras", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "dataset_vpuredna.jsonl")

    random.seed(args.seed)

    print("=" * 60)
    print(" 🧬 vPureDna — GERADOR DE DATASET DNA (⌬)")
    print("=" * 60)

    # 1. Compressor
    print("\n[1] Inicializando DNA Compressor (⌬)...")
    comp = DNACompressor()
    print(f"  ⌬W: {comp.total_w} | ⌬F: {comp.total_f}")

    # 2. Corpus
    print("\n[2] Carregando corpus...")
    dataset, source = carregar_corpus()

    # 3. Gerar dataset trifásico
    print(f"\n[3] Gerando {args.max_amostras} amostras trifásicas...")

    TERCO = args.max_amostras // 3
    dados = []
    stats = {"total": 0, "descartados": 0, "avg_ratio": []}

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if stats["total"] >= args.max_amostras:
            break

        item = dataset[idx]
        instrucao, saida = extrair_texto(item, source)

        if not instrucao or not saida:
            stats["descartados"] += 1
            continue
        if len(instrucao) + len(saida) > args.max_chars:
            stats["descartados"] += 1
            continue

        fase = stats["total"]

        # Comprimir a resposta
        resposta_dna = comp.compress(saida)
        s = comp.stats(saida)
        stats["avg_ratio"].append(s["compression_ratio"])

        # FASE A (33%): Pergunta normal → Resposta com ⌬
        # Ensina o modelo a EMITIR ⌬ nas respostas
        if fase < TERCO:
            obj = {
                "instruction": "Responda de forma comprimida usando marcadores ⌬ para frases conhecidas.",
                "input": instrucao,
                "output": resposta_dna,
            }

        # FASE B (33%): Texto com ⌬ → Expandir para humano
        # Ensina o modelo a ENTENDER ⌬
        elif fase < 2 * TERCO:
            obj = {
                "instruction": "Expanda os marcadores ⌬ para texto humano completo em Português.",
                "input": resposta_dna,
                "output": saida.lower(),
            }

        # FASE C (33%): Pergunta normal → Resposta normal (baseline)
        # Mantém capacidade de responder sem ⌬
        else:
            obj = {
                "instruction": "Responda em Português claro.",
                "input": instrucao,
                "output": saida,
            }

        dados.append(obj)
        stats["total"] += 1

        if stats["total"] % 200 == 0:
            print(f"    ... {stats['total']}/{args.max_amostras}")

    # 4. Salvar
    print(f"\n[4] Salvando {len(dados)} amostras...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for obj in dados:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    avg_ratio = sum(stats["avg_ratio"]) / max(len(stats["avg_ratio"]), 1)
    file_mb = os.path.getsize(args.output) / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f" ✅ DATASET vPureDna GERADO!")
    print(f"{'=' * 60}")
    print(f"  Arquivo:     {args.output}")
    print(f"  Tamanho:     {file_mb:.1f} MB")
    print(f"  Amostras:    {len(dados)}")
    print(f"  Descartados: {stats['descartados']}")
    print(f"  Compressão:  {avg_ratio:.1f}x média")
    print(f"  Fases:       A={TERCO} (⌬ output) | B={TERCO} (expand) | C={len(dados)-2*TERCO} (normal)")
    print(f"{'=' * 60}")

    # Amostra
    print(f"\n📋 Amostras:")
    for i, fase in [(0, "A (comprimir)"), (TERCO, "B (expandir)"), (2*TERCO, "C (normal)")]:
        if i < len(dados):
            d = dados[i]
            print(f"\n  [{fase}]")
            print(f"  Inst:   {d['instruction'][:60]}...")
            print(f"  Input:  {d['input'][:60]}...")
            print(f"  Output: {d['output'][:60]}...")


if __name__ == "__main__":
    main()
