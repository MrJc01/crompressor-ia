#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 CROM-IA V2: Gerador de Codebooks Hierárquicos DNA      ║
║                                                              ║
║  Gera codebooks semânticos para compressão via Codebook-DNA  ║
║  Modos: --fixo (estático) e --dinamico (expansível)          ║
║  Taxas: 1:3, 1:5, 1:10, 1:20                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import re
import math
import argparse
import unicodedata
from collections import Counter
from itertools import product

# ============================
# Constantes de Configuração
# ============================
DNA_ALPHABET = ['A', 'T', 'C', 'G']
ESCAPE_PREFIX = "@@"

TAXA_CONFIG = {
    "1x3": {
        "nome": "Bigramas/Trigramas",
        "n_gramas": [2, 3],
        "max_entradas": 5000,
        "descricao": "Fragmentos de 2-3 palavras",
    },
    "1x5": {
        "nome": "Trigramas Expandidos",
        "n_gramas": [2, 3],
        "max_entradas": 15000,
        "descricao": "Dicionário expandido de 2-3 palavras",
    },
    "1x10": {
        "nome": "Frases Completas",
        "n_gramas": [5, 6, 7, 8, 9, 10],
        "max_entradas": 50000,
        "descricao": "Frases de 5-10 palavras",
    },
    "1x20": {
        "nome": "Parágrafos",
        "n_gramas": [10, 12, 15, 18, 20],
        "max_entradas": 100000,
        "descricao": "Blocos de 10-20 palavras",
    },
}


def normalizar_texto(texto):
    """Remove pontuação excessiva, normaliza espaços."""
    texto = str(texto).strip()
    texto = re.sub(r'\s+', ' ', texto)
    # Manter pontuação básica mas normalizar
    texto = texto.lower()
    return texto


def tokenizar(texto):
    """Tokeniza texto em palavras simples."""
    palavras = re.findall(r'[\w]+|[.,!?;:]', texto.lower())
    return palavras


def extrair_ngramas(palavras, ns):
    """Extrai n-gramas de uma lista de palavras."""
    ngramas = []
    for n in ns:
        for i in range(len(palavras) - n + 1):
            fragmento = ' '.join(palavras[i:i+n])
            # Filtrar fragmentos que são só pontuação
            if any(c.isalpha() for c in fragmento):
                ngramas.append(fragmento)
    return ngramas


def gerar_codigos_dna(quantidade):
    """
    Gera códigos DNA únicos com propriedade Huffman-like:
    códigos mais curtos são atribuídos primeiro (= mais frequentes).
    
    Reserva:
    - Prefixo '@@' para escape literal
    - Códigos começando com 2+ caracteres
    """
    codigos = []
    tamanho = 2  # Mínimo 2 chars para evitar ambiguidade
    
    while len(codigos) < quantidade:
        for combo in product(DNA_ALPHABET, repeat=tamanho):
            codigo = ''.join(combo)
            codigos.append(codigo)
            if len(codigos) >= quantidade:
                break
        tamanho += 1
    
    return codigos[:quantidade]


def calcular_entropia(texto):
    """Calcula entropia de Shannon H."""
    if not texto:
        return 0.0
    contagem = Counter(texto)
    total = sum(contagem.values())
    entropia = 0.0
    for count in contagem.values():
        if count > 0:
            p = count / total
            entropia -= p * math.log2(p)
    return entropia


def carregar_corpus():
    """Carrega corpus Alpaca-PT do HuggingFace."""
    print("\n[1/4] 📥 Carregando corpus Alpaca-PT do HuggingFace...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "FreedomIntelligence/alpaca-gpt4-portuguese",
            split="train"
        )
        print(f"      ✅ {len(dataset)} exemplos carregados")
        return dataset
    except ImportError:
        print("      ❌ Biblioteca 'datasets' não encontrada!")
        print("      Instale com: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"      ❌ Erro ao carregar: {e}")
        sys.exit(1)


def extrair_respostas(dataset):
    """Extrai todas as respostas do corpus."""
    respostas = []
    for item in dataset:
        convs = item.get('conversations', [])
        for c in convs:
            if c.get('from') == 'gpt':
                texto = str(c.get('value', '')).strip()
                if texto and len(texto) > 10:
                    respostas.append(texto)
    print(f"      ✅ {len(respostas)} respostas extraídas")
    return respostas


def gerar_codebook_para_taxa(respostas, taxa_key, modo):
    """
    Gera um codebook para uma taxa específica.
    
    Args:
        respostas: lista de respostas do corpus
        taxa_key: "1x3", "1x5", "1x10", "1x20"
        modo: "fixo" ou "dinamico"
    
    Returns:
        dict: codebook completo com metadados
    """
    config = TAXA_CONFIG[taxa_key]
    max_entradas = config["max_entradas"]
    ns = config["n_gramas"]
    
    print(f"\n{'─' * 60}")
    print(f"  📊 Gerando codebook {taxa_key} ({modo})")
    print(f"     N-gramas: {ns}")
    print(f"     Meta: {max_entradas} entradas")
    print(f"{'─' * 60}")
    
    # ── Fase 1: Incluir palavras unitárias mais frequentes ──
    print("  [a] Contando palavras unitárias...")
    contagem_palavras = Counter()
    for resp in respostas:
        palavras = tokenizar(resp)
        contagem_palavras.update(palavras)
    
    # Top 200 palavras (Camada 1 - baseline)
    top_palavras = [(p, f) for p, f in contagem_palavras.most_common(200)
                    if len(p) > 1 and p.isalpha()]
    
    # ── Fase 2: Extrair n-gramas ──
    print("  [b] Extraindo n-gramas do corpus...")
    contagem_ngramas = Counter()
    total = len(respostas)
    
    for i, resp in enumerate(respostas):
        if i % 5000 == 0:
            print(f"      Processando {i}/{total}...")
        palavras = tokenizar(resp)
        ngramas = extrair_ngramas(palavras, ns)
        contagem_ngramas.update(ngramas)
    
    # Filtrar n-gramas com frequência mínima
    freq_minima = 3 if taxa_key in ("1x3", "1x5") else 2
    ngramas_filtrados = [
        (ng, f) for ng, f in contagem_ngramas.most_common()
        if f >= freq_minima and len(ng) > 3
    ]
    
    print(f"      ✅ {len(ngramas_filtrados)} n-gramas únicos (freq ≥ {freq_minima})")
    
    # ── Fase 3: Selecionar top entradas ──
    # Combinar palavras unitárias + n-gramas
    todas_entradas = []
    
    # Adicionar palavras unitárias primeiro (alta prioridade)
    for palavra, freq in top_palavras[:200]:
        todas_entradas.append({
            "text": palavra,
            "freq": freq,
            "category": "word",
            "n": 1,
        })
    
    # Adicionar n-gramas por frequência
    for ngrama, freq in ngramas_filtrados:
        if len(todas_entradas) >= max_entradas:
            break
        todas_entradas.append({
            "text": ngrama,
            "freq": freq,
            "category": f"{len(ngrama.split())}-gram",
            "n": len(ngrama.split()),
        })
    
    entradas_reais = len(todas_entradas)
    print(f"      ✅ {entradas_reais} entradas selecionadas (meta: {max_entradas})")
    
    if entradas_reais < max_entradas:
        print(f"      ⚠️  Corpus insuficiente para {max_entradas}. Usando {entradas_reais}.")
    
    # ── Fase 4: Atribuir códigos DNA (Huffman-like) ──
    print("  [c] Gerando códigos DNA (Huffman-like)...")
    codigos = gerar_codigos_dna(entradas_reais)
    
    entries = {}
    reverse_map = {}  # text → code (para compressão rápida)
    
    for i, entrada in enumerate(todas_entradas):
        code = codigos[i]
        entries[code] = {
            "text": entrada["text"],
            "freq": entrada["freq"],
            "category": entrada["category"],
            "n": entrada["n"],
        }
        reverse_map[entrada["text"]] = code
    
    # ── Fase 5: Calcular estatísticas ──
    total_freq = sum(e["freq"] for e in entries.values())
    palavras_por_codigo = sum(e["n"] * e["freq"] for e in entries.values()) / max(total_freq, 1)
    code_lengths = [len(c) for c in entries.keys()]
    avg_code_len = sum(code_lengths) / max(len(code_lengths), 1)
    
    # Calcular cobertura estimada do corpus
    palavras_cobertas = sum(e["freq"] for e in entries.values())
    total_palavras_corpus = sum(contagem_palavras.values())
    cobertura_pct = (palavras_cobertas / max(total_palavras_corpus, 1)) * 100
    
    # ── Fase 6: Montar codebook final ──
    codebook = {
        "version": "2.0",
        "taxa_alvo": taxa_key.replace("x", ":"),
        "modo": modo,
        "dynamic": modo == "dinamico",
        "expansion_threshold": 10 if modo == "dinamico" else None,
        "descricao": config["descricao"],
        "n_gramas_usados": ns,
        "escape_prefix": ESCAPE_PREFIX,
        "entries": entries,
        "reverse_map": reverse_map,
        "stats": {
            "total_entries": entradas_reais,
            "meta_entries": max_entradas,
            "avg_compression_ratio": round(palavras_por_codigo, 2),
            "avg_code_length": round(avg_code_len, 2),
            "min_code_length": min(code_lengths) if code_lengths else 0,
            "max_code_length": max(code_lengths) if code_lengths else 0,
            "coverage_pct": round(cobertura_pct, 1),
            "corpus_size": len(respostas),
            "total_freq": total_freq,
            "freq_minima": freq_minima,
            "entropia_media_codebook": round(
                sum(calcular_entropia(e["text"]) for e in list(entries.values())[:100]) / 100, 2
            ),
        },
    }
    
    # Info de categorias
    cat_counts = Counter(e["category"] for e in entries.values())
    codebook["stats"]["categorias"] = dict(cat_counts.most_common())
    
    print(f"      ✅ Codebook {taxa_key}_{modo} gerado!")
    print(f"         Entradas     : {entradas_reais}")
    print(f"         Compressão   : 1:{palavras_por_codigo:.1f} média")
    print(f"         Code length  : {min(code_lengths)}-{max(code_lengths)} chars")
    print(f"         Cobertura    : {cobertura_pct:.1f}%")
    
    return codebook


def salvar_codebook(codebook, caminho):
    """Salva codebook como JSON."""
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)
    
    tamanho_mb = os.path.getsize(caminho) / 1024 / 1024
    print(f"      💾 Salvo: {caminho} ({tamanho_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="🧬 CROM-IA V2: Gerador de Codebooks Hierárquicos DNA"
    )
    parser.add_argument(
        "--taxas", nargs="+",
        default=["1x3", "1x5", "1x10", "1x20"],
        choices=["1x3", "1x5", "1x10", "1x20"],
        help="Taxas de compressão a gerar (default: todas)"
    )
    parser.add_argument(
        "--modos", nargs="+",
        default=["fixo", "dinamico"],
        choices=["fixo", "dinamico"],
        help="Modos a gerar (default: ambos)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Diretório de saída (default: ./codebooks/)"
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__))
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" 🧬 CROM-IA V2: GERADOR DE CODEBOOKS HIERÁRQUICOS DNA")
    print("=" * 60)
    print(f" Taxas   : {args.taxas}")
    print(f" Modos   : {args.modos}")
    print(f" Saída   : {args.output_dir}")
    print("=" * 60)
    
    # ── Carregar corpus ──
    dataset = carregar_corpus()
    respostas = extrair_respostas(dataset)
    
    # ── Gerar codebooks ──
    print(f"\n[2/4] 🏗️  Gerando {len(args.taxas) * len(args.modos)} codebooks...")
    
    gerados = []
    for taxa in args.taxas:
        for modo in args.modos:
            codebook = gerar_codebook_para_taxa(respostas, taxa, modo)
            
            nome_arquivo = f"codebook_{taxa}_{modo}.json"
            caminho = os.path.join(args.output_dir, nome_arquivo)
            salvar_codebook(codebook, caminho)
            gerados.append((taxa, modo, caminho, codebook["stats"]))
    
    # ── Relatório Final ──
    print("\n" + "=" * 60)
    print(" ✅ RELATÓRIO FINAL — CODEBOOKS GERADOS")
    print("=" * 60)
    print(f"{'Taxa':<8} {'Modo':<10} {'Entradas':<10} {'Compressão':<12} {'Cobertura':<10}")
    print("─" * 60)
    
    for taxa, modo, caminho, stats in gerados:
        print(
            f"{taxa:<8} {modo:<10} "
            f"{stats['total_entries']:<10} "
            f"1:{stats['avg_compression_ratio']:<11} "
            f"{stats['coverage_pct']:.1f}%"
        )
    
    print("─" * 60)
    print(f" Total: {len(gerados)} codebooks gerados em {args.output_dir}")
    print("=" * 60)
    print("\n🚀 Próximo passo: Execute gerar_dataset_comprimido.py")


if __name__ == "__main__":
    main()
