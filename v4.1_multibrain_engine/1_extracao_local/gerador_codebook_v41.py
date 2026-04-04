#!/usr/bin/env python3
"""
CROM-IA V4.1 — Gerador de Codebook Inteligente (Data-Driven)
Filosofia Crompressor: comprimir o que é REALMENTE repetido nos dados.

Em vez de codebook hardcoded, este script:
1. Analisa o dataset REAL por frequência de n-grams
2. Ranqueia por (frequência × tamanho) = economia real de bytes
3. Gera codebook ordenado pelos tokens que MAIS economizam espaço
4. Usa hash Radix-4 (DNA: A, T, C, G) para chaves compactas
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
            if len(chunk_limpo) >= 4:  # Mínimo 4 chars para valer a pena
                ngrams.append(chunk_limpo)
    return ngrams


def calcular_entropia(texto):
    """Entropia de Shannon — mede a 'informação' do texto."""
    if not texto:
        return 0
    freq = Counter(texto)
    total = len(texto)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def analisar_dataset(path_dataset, max_linhas=50000):
    """
    Fase 1 do Crompressor: Mineração de frequência REAL.
    Analisa o dataset inteiro e conta o que mais se repete.
    """
    print(f"🔬 Analisando frequência real do dataset: {path_dataset}")

    # Contadores por nível hierárquico
    cnt_palavras = Counter()      # Words (1 token)
    cnt_frases = Counter()        # Phrases (2-8 tokens)
    cnt_blocos = Counter()        # Blocks (8-20 tokens)

    linhas = 0
    with open(path_dataset, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                texto = data.get("output", data.get("text", data.get("response", "")))
            except json.JSONDecodeError:
                texto = line.strip()

            if not texto:
                continue

            # Palavras individuais (frequência bruta)
            palavras = re.findall(r'\b\w{4,}\b', texto.lower())
            cnt_palavras.update(palavras)

            # N-grams de frases (4-8 palavras)
            for bloco in texto.split("\n"):
                bloco = bloco.strip()
                if len(bloco) > 20:
                    cnt_frases.update(extrair_ngrams(bloco, 3, 8))
                if len(bloco) > 60:
                    cnt_blocos.update(extrair_ngrams(bloco, 8, 16))

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
    Fase 2 do Crompressor: Ranquear por ECONOMIA REAL de bytes.
    Score = frequência × (len(texto) - len(token_dna))
    Um token de 20 chars que aparece 100 vezes vale MAIS que
    um token de 5 chars que aparece 200 vezes.
    """
    TOKEN_DNA_SIZE = 5  # @@XXX = 5 bytes
    candidatos = []

    for texto, freq in contador.items():
        if freq < min_freq:
            continue
        economia_por_hit = len(texto) - TOKEN_DNA_SIZE
        if economia_por_hit <= 0:
            continue  # Não vale comprimir se o token DNA é maior
        score = freq * economia_por_hit  # Bytes totais economizados
        candidatos.append({
            "texto": texto,
            "freq": freq,
            "tamanho": len(texto),
            "economia_por_hit": economia_por_hit,
            "score_total": score,
        })

    # Ordenar pelo que MAIS economiza bytes no total
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
        return radix[(idx // 64) % 4] + radix[(idx // 16) % 4] + radix[(idx // 4) % 4] + radix[idx % 4]


def gerar_codebook(path_dataset, path_saida, sigla_dominio="G", max_tokens=200, min_freq=5):
    """
    Pipeline completo Crompressor → Codebook DNA.
    1. Minera frequência real
    2. Ranqueia por economia de bytes
    3. Gera codebook com hash Radix-4
    """
    print(f"\n{'='*60}")
    print(f"🧬 CROM-IA V4.1 — Gerador de Codebook Inteligente")
    print(f"   Domínio: {sigla_dominio} | Dataset: {os.path.basename(path_dataset)}")
    print(f"{'='*60}\n")

    # Fase 1: Minerar
    cnt_palavras, cnt_frases, cnt_blocos = analisar_dataset(path_dataset)

    # Fase 2: Ranquear por economia
    rank_palavras = ranquear_por_economia(cnt_palavras, min_freq)
    rank_frases = ranquear_por_economia(cnt_frases, min_freq)
    rank_blocos = ranquear_por_economia(cnt_blocos, min_freq=3)

    # Fase 3: Distribuir tokens por hierarquia
    # 40% palavras, 40% frases, 20% blocos
    n_palavras = int(max_tokens * 0.4)
    n_frases = int(max_tokens * 0.4)
    n_blocos = max_tokens - n_palavras - n_frases

    codebook = {}
    stats = {"economia_total_estimada": 0}
    idx = 0

    # Palavras (W)
    for entry in rank_palavras[:n_palavras]:
        sufixo = gerar_hash_radix4(idx)
        chave = f"@@{sigla_dominio}W{sufixo}"
        codebook[entry["texto"]] = chave
        stats["economia_total_estimada"] += entry["score_total"]
        idx += 1

    # Frases (F)
    idx_f = 0
    for entry in rank_frases[:n_frases]:
        # Evitar sobreposição com palavras já no codebook
        if entry["texto"] not in codebook:
            sufixo = gerar_hash_radix4(idx_f)
            chave = f"@@{sigla_dominio}F{sufixo}"
            codebook[entry["texto"]] = chave
            stats["economia_total_estimada"] += entry["score_total"]
            idx_f += 1

    # Blocos (P)
    idx_p = 0
    for entry in rank_blocos[:n_blocos]:
        if entry["texto"] not in codebook:
            sufixo = gerar_hash_radix4(idx_p)
            chave = f"@@{sigla_dominio}P{sufixo}"
            codebook[entry["texto"]] = chave
            stats["economia_total_estimada"] += entry["score_total"]
            idx_p += 1

    # Salvar codebook
    payload = {
        "version": "4.1",
        "domain": sigla_dominio,
        "method": "frequency_x_size_ranked",
        "total_tokens": len(codebook),
        "economia_bytes_estimada": stats["economia_total_estimada"],
        "codebook": codebook,
    }

    with open(path_saida, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Relatório
    print(f"\n📊 RELATÓRIO DO CODEBOOK:")
    print(f"   Tokens gerados: {len(codebook)}")
    print(f"   Economia estimada: {stats['economia_total_estimada']:,} bytes")
    print(f"   Top 5 maior economia:")
    top5 = sorted(
        [(k, v) for k, v in codebook.items()],
        key=lambda x: len(x[0]),
        reverse=True
    )[:5]
    for texto, token in top5:
        print(f"     '{texto[:50]}...' → {token}")
    print(f"\n   Salvo em: {path_saida}")

    return codebook


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python3 gerador_codebook_v41.py <sigla_dominio> <dataset.jsonl> <saida_codebook.json> [max_tokens]")
        print("Exemplo: python3 gerador_codebook_v41.py P dataset_python.jsonl codebook_python.json 200")
        sys.exit(1)

    sigla = sys.argv[1]
    dataset = sys.argv[2]
    saida = sys.argv[3]
    max_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    gerar_codebook(dataset, saida, sigla, max_tok)
