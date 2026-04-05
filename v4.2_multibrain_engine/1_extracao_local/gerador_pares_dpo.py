#!/usr/bin/env python3
"""
CROM-IA V4.2 — Gerador de Pares DPO (Direct Preference Optimization)
=====================================================================
Gera pares {prompt, chosen, rejected} automaticamente:
  - chosen  = resposta com DNA aplicado (25%) — preferida
  - rejected = resposta original sem DNA — rejeitada
O modelo aprende a PREFERIR usar DNA quando oportuno.
"""

import json
import os
import sys
import random
import re
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def carregar_codebook(path_codebook):
    """Carrega codebook DNA."""
    if path_codebook and os.path.exists(path_codebook):
        with open(path_codebook, "r") as f:
            data = json.load(f)
        if "codebook" in data and isinstance(data["codebook"], dict):
            return data["codebook"]
        return data
    return {}


def aplicar_mutacao_dna(texto, codebook, taxa=0.25):
    """Aplica DNA apenas a uma fração das palavras-chave encontradas."""
    resultado = texto
    matches_encontrados = 0
    matches_mutados = 0

    for palavra, token_dna in codebook.items():
        ocorrencias = len(re.findall(re.escape(palavra), resultado, re.IGNORECASE))
        if ocorrencias > 0:
            matches_encontrados += ocorrencias
            # Aplicar DNA a cada ocorrência com probabilidade
            def substituir_com_prob(match):
                nonlocal matches_mutados
                if random.random() < taxa:
                    matches_mutados += 1
                    return token_dna
                return match.group(0)

            pattern = re.compile(re.escape(palavra), re.IGNORECASE)
            resultado = pattern.sub(substituir_com_prob, resultado)

    return resultado, matches_mutados


def extrair_instrucao_output(entry):
    """Extrai instrução e output de múltiplos formatos."""
    # Formato ChatML
    if "text" in entry and "<|im_start|>" in entry.get("text", ""):
        text = entry["text"]
        user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
        asst_match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
        instruction = user_match.group(1).strip() if user_match else ""
        output = asst_match.group(1).strip() if asst_match else ""
        return instruction, output

    # Formato instrução/output
    instruction = (entry.get('instruction', '') or
                  entry.get('input', '') or
                  entry.get('prompt', '') or '')
    output = (entry.get('output', '') or
             entry.get('response', '') or
             entry.get('completion', '') or '')

    return instruction.strip(), output.strip()


def gerar_pares_dpo(path_input, path_output, codebook, max_pares=5000,
                     taxa_dna=0.25, min_dna_tokens=3, min_output_chars=100):
    """
    Pipeline de geração de pares DPO.
    chosen = output com DNA
    rejected = output original
    """
    print(f"\n{'='*60}")
    print(f"🧬 CROM-IA V4.2 — Gerador de Pares DPO")
    print(f"   Input: {os.path.basename(path_input)}")
    print(f"   Taxa DNA: {taxa_dna*100:.0f}%")
    print(f"   Max pares: {max_pares}")
    print(f"{'='*60}\n")

    total_lidos = 0
    total_pares = 0
    total_rejeitados_curtos = 0
    total_rejeitados_sem_dna = 0

    with open(path_input, "r", encoding="utf-8") as fin, \
         open(path_output, "w", encoding="utf-8") as fout:

        for line in fin:
            if total_pares >= max_pares:
                break

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            total_lidos += 1
            instruction, output = extrair_instrucao_output(entry)

            if not instruction or not output:
                continue

            # Filtro: output mínimo
            if len(output) < min_output_chars:
                total_rejeitados_curtos += 1
                continue

            # Gerar chosen (com DNA)
            chosen, num_dna = aplicar_mutacao_dna(output, codebook, taxa=taxa_dna)

            # Filtro: precisa ter DNA suficiente
            if num_dna < min_dna_tokens:
                total_rejeitados_sem_dna += 1
                continue

            # chosen e rejected devem ser DIFERENTES
            if chosen == output:
                total_rejeitados_sem_dna += 1
                continue

            # Formato DPO
            par = {
                "prompt": instruction,
                "chosen": chosen,
                "rejected": output,
            }

            fout.write(json.dumps(par, ensure_ascii=False) + "\n")
            total_pares += 1

    print(f"📊 RELATÓRIO DPO:")
    print(f"   Amostras lidas: {total_lidos}")
    print(f"   Pares gerados: {total_pares}")
    print(f"   Rejeitados (curtos): {total_rejeitados_curtos}")
    print(f"   Rejeitados (sem DNA suficiente): {total_rejeitados_sem_dna}")
    print(f"   Taxa de conversão: {total_pares/max(total_lidos,1)*100:.1f}%")
    print(f"\n   Salvo em: {path_output}")

    return total_pares


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CROM-IA V4.2 — Gerador de Pares DPO")
    parser.add_argument('--input', required=True, help='Dataset de entrada (.jsonl)')
    parser.add_argument('--codebook', required=True, help='Codebook DNA (.json)')
    parser.add_argument('--output', default=None, help='Saída (.jsonl)')
    parser.add_argument('--max_pares', type=int, default=5000, help='Máximo de pares (default: 5000)')
    parser.add_argument('--taxa_dna', type=float, default=0.25, help='Taxa de DNA (default: 0.25)')
    parser.add_argument('--min_dna', type=int, default=3, help='Mínimo de tokens DNA por par (default: 3)')

    args = parser.parse_args()

    if not args.output:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(SCRIPT_DIR, "datasets_hibridos",
                                   f"dataset_DPO_{base}.jsonl")

    codebook = carregar_codebook(args.codebook)
    if not codebook:
        print("❌ Codebook vazio ou não encontrado!")
        sys.exit(1)

    print(f"📖 Codebook: {len(codebook)} tokens DNA carregados")
    gerar_pares_dpo(args.input, args.output, codebook,
                     max_pares=args.max_pares, taxa_dna=args.taxa_dna,
                     min_dna_tokens=args.min_dna)
