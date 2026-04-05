#!/usr/bin/env python3
"""
CROM-IA V4.2 — Gerador de Codebook Hierárquico (Data-Driven)
=============================================================
Filosofia Crompressor: comprimir o que é REALMENTE repetido nos dados.
DNA em 3 NÍVEIS hierárquicos:
  W (Word)      = palavras isoladas frequentes     → @@GWA
  F (Phrase)    = frases de 2-8 palavras repetidas  → @@GFA
  P (Paragraph) = blocos de 8-20 palavras           → @@GPA

Economia REAL: score = frequência × (len(texto) - len(token_dna))
Um parágrafo inteiro que se repete 50x vale MUITO mais que uma palavra.
"""

import json
import re
import sys
import os
from collections import Counter
import math


def tokenizar(texto):
    """Tokeniza texto preservando estrutura (palavras + pontuação + espaços)."""
    return re.findall(r'\w+|\s+|[^\w\s]', texto)


def extrair_ngrams(texto, min_n, max_n):
    """Extrai n-grams via sliding window (core do Crompressor)."""
    tokens = tokenizar(texto)
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            chunk = "".join(tokens[i:i + n])
            chunk_limpo = chunk.strip()
            if len(chunk_limpo) >= 4:
                ngrams.append(chunk_limpo)
    return ngrams


def analisar_dataset(path_dataset, max_linhas=50000):
    """
    Fase 1 do Crompressor: Mineração de frequência REAL.
    3 níveis hierárquicos:
      - Palavras: tokens individuais
      - Frases: 2-8 tokens combinados
      - Blocos: 8-20 tokens (parágrafos repetidos)
    """
    print(f"🔬 Analisando frequência real do dataset: {path_dataset}")

    cnt_palavras = Counter()
    cnt_frases = Counter()
    cnt_blocos = Counter()

    linhas = 0
    with open(path_dataset, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Suporta múltiplos formatos
                texto = (data.get("output", "") or
                        data.get("text", "") or
                        data.get("response", "") or "")

                # Se for ChatML, extrair o conteúdo do assistant
                if "<|im_start|>assistant" in texto:
                    match = re.search(r'<\|im_start\|>assistant\n(.*?)(<\|im_end\|>|$)',
                                      texto, re.DOTALL)
                    if match:
                        texto = match.group(1)
            except json.JSONDecodeError:
                texto = line.strip()

            if not texto:
                continue

            # NÍVEL 1 — Palavras individuais (frequência bruta)
            palavras = re.findall(r'\b\w{4,}\b', texto.lower())
            cnt_palavras.update(palavras)

            # NÍVEL 2 — Frases (3-8 palavras)
            # Frases que se repetem entre múltiplas amostras = alto valor
            for bloco in texto.split("\n"):
                bloco = bloco.strip()
                if len(bloco) > 20:
                    cnt_frases.update(extrair_ngrams(bloco, 3, 8))

            # NÍVEL 3 — Blocos/Parágrafos (8-20 palavras)
            # Parágrafos inteiros repetidos = MÁXIMA economia
            for bloco in texto.split("\n"):
                bloco = bloco.strip()
                if len(bloco) > 60:
                    cnt_blocos.update(extrair_ngrams(bloco, 8, 16))

                # Também indexar o parágrafo inteiro se não for muito longo
                if 60 < len(bloco) < 300:
                    cnt_blocos[bloco] += 1

            linhas += 1
            if linhas >= max_linhas:
                break

    print(f"   Linhas analisadas: {linhas}")
    print(f"   Palavras únicas: {len(cnt_palavras)}")
    print(f"   Frases únicas: {len(cnt_frases)}")
    print(f"   Blocos únicos: {len(cnt_blocos)}")

    return cnt_palavras, cnt_frases, cnt_blocos


def ranquear_por_economia(contador, min_freq=5):
    """
    Fase 2: Ranquear por ECONOMIA REAL de bytes.
    Score = frequência × (len(texto) - len(token_dna))
    Um parágrafo de 200 chars × 50 vezes = 10.000 bytes economizados!
    """
    TOKEN_DNA_SIZE = 6  # @@XXYY = 6 bytes médio
    candidatos = []

    for texto, freq in contador.items():
        if freq < min_freq:
            continue
        economia_por_hit = len(texto) - TOKEN_DNA_SIZE
        if economia_por_hit <= 0:
            continue
        score = freq * economia_por_hit
        candidatos.append({
            "texto": texto,
            "freq": freq,
            "tamanho": len(texto),
            "economia_por_hit": economia_por_hit,
            "score_total": score,
        })

    candidatos.sort(key=lambda x: x["score_total"], reverse=True)
    return candidatos


def gerar_hash_radix4(idx):
    """Gera sufixo DNA usando base Radix-4 (A, T, C, G)."""
    radix = ['A', 'T', 'C', 'G']
    if idx < 4:
        return radix[idx]
    elif idx < 16:
        return radix[idx // 4] + radix[idx % 4]
    elif idx < 64:
        return radix[(idx // 16) % 4] + radix[(idx // 4) % 4] + radix[idx % 4]
    else:
        return (radix[(idx // 64) % 4] + radix[(idx // 16) % 4] +
                radix[(idx // 4) % 4] + radix[idx % 4])


def gerar_codebook(path_dataset, path_saida, sigla_dominio="G", max_tokens=200,
                   min_freq=5, distribuicao=None):
    """
    Pipeline completo Crompressor → Codebook DNA Hierárquico.
    1. Minera frequência real
    2. Ranqueia por economia de bytes
    3. Gera codebook em 3 níveis (W, F, P) com hash Radix-4
    """
    if distribuicao is None:
        # Distribuição padrão: 40% palavras, 40% frases, 20% blocos
        distribuicao = (0.4, 0.4, 0.2)

    print(f"\n{'='*60}")
    print(f"🧬 CROM-IA V4.2 — Gerador de Codebook Hierárquico")
    print(f"   Domínio: {sigla_dominio}")
    print(f"   Dataset: {os.path.basename(path_dataset)}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Distribuição: W={distribuicao[0]*100:.0f}% F={distribuicao[1]*100:.0f}% P={distribuicao[2]*100:.0f}%")
    print(f"{'='*60}\n")

    # Fase 1: Minerar
    cnt_palavras, cnt_frases, cnt_blocos = analisar_dataset(path_dataset)

    # Fase 2: Ranquear por economia
    rank_palavras = ranquear_por_economia(cnt_palavras, min_freq)
    rank_frases = ranquear_por_economia(cnt_frases, min_freq)
    rank_blocos = ranquear_por_economia(cnt_blocos, min_freq=3)

    # Fase 3: Distribuir tokens por hierarquia
    n_palavras = int(max_tokens * distribuicao[0])
    n_frases = int(max_tokens * distribuicao[1])
    n_blocos = max_tokens - n_palavras - n_frases

    codebook = {}
    stats = {"economia_total_estimada": 0, "por_nivel": {}}
    idx = 0

    # W (Words)
    economia_w = 0
    for entry in rank_palavras[:n_palavras]:
        sufixo = gerar_hash_radix4(idx)
        chave = f"@@{sigla_dominio}W{sufixo}"
        codebook[entry["texto"]] = chave
        economia_w += entry["score_total"]
        idx += 1
    stats["por_nivel"]["W_palavras"] = {"tokens": min(len(rank_palavras), n_palavras),
                                         "economia": economia_w}

    # F (Phrases)
    economia_f = 0
    idx_f = 0
    for entry in rank_frases[:n_frases]:
        if entry["texto"] not in codebook:
            sufixo = gerar_hash_radix4(idx_f)
            chave = f"@@{sigla_dominio}F{sufixo}"
            codebook[entry["texto"]] = chave
            economia_f += entry["score_total"]
            idx_f += 1
    stats["por_nivel"]["F_frases"] = {"tokens": idx_f, "economia": economia_f}

    # P (Paragraphs/Blocks)
    economia_p = 0
    idx_p = 0
    for entry in rank_blocos[:n_blocos]:
        if entry["texto"] not in codebook:
            sufixo = gerar_hash_radix4(idx_p)
            chave = f"@@{sigla_dominio}P{sufixo}"
            codebook[entry["texto"]] = chave
            economia_p += entry["score_total"]
            idx_p += 1
    stats["por_nivel"]["P_blocos"] = {"tokens": idx_p, "economia": economia_p}

    stats["economia_total_estimada"] = economia_w + economia_f + economia_p

    # Salvar codebook
    payload = {
        "version": "4.2",
        "domain": sigla_dominio,
        "method": "hierarchical_frequency_x_size",
        "total_tokens": len(codebook),
        "niveis": {
            "W": "palavras isoladas",
            "F": "frases (2-8 palavras)",
            "P": "parágrafos/blocos (8-20 palavras)"
        },
        "economia_bytes_estimada": stats["economia_total_estimada"],
        "stats": stats,
        "codebook": codebook,
    }

    with open(path_saida, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Relatório
    print(f"\n📊 RELATÓRIO DO CODEBOOK HIERÁRQUICO:")
    print(f"   Tokens gerados: {len(codebook)}")
    print(f"   ├── W (palavras): {stats['por_nivel']['W_palavras']['tokens']} "
          f"({stats['por_nivel']['W_palavras']['economia']:,} bytes)")
    print(f"   ├── F (frases):   {stats['por_nivel']['F_frases']['tokens']} "
          f"({stats['por_nivel']['F_frases']['economia']:,} bytes)")
    print(f"   └── P (blocos):   {stats['por_nivel']['P_blocos']['tokens']} "
          f"({stats['por_nivel']['P_blocos']['economia']:,} bytes)")
    print(f"   Economia total: {stats['economia_total_estimada']:,} bytes")

    print(f"\n   Top 5 maior economia:")
    top5 = sorted(
        [(k, v) for k, v in codebook.items()],
        key=lambda x: len(x[0]),
        reverse=True
    )[:5]
    for texto, token in top5:
        nivel = "BLOCO" if len(texto) > 50 else "FRASE" if len(texto) > 15 else "WORD"
        print(f"     [{nivel}] '{texto[:60]}' → {token}")

    print(f"\n   Salvo em: {path_saida}")
    return codebook


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("CROM-IA V4.2 — Gerador de Codebook Hierárquico")
        print(f"Uso: python3 {sys.argv[0]} <sigla> <dataset.jsonl> <saida.json> [max_tokens]")
        print(f"Exemplo: python3 {sys.argv[0]} P python_15k.jsonl codebook_python.json 200")
        print(f"\nSignas: P=Python, M=Medicina, G=Geral, C=Conversa")
        sys.exit(1)

    sigla = sys.argv[1]
    dataset = sys.argv[2]
    saida = sys.argv[3]
    max_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    gerar_codebook(dataset, saida, sigla, max_tok)
