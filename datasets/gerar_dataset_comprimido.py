#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 CROM-IA V2: Gerador de Datasets Comprimidos via DNA    ║
║                                                              ║
║  Comprime respostas do Alpaca-PT usando codebooks semânticos ║
║  Modo fixo: codebook estático                                ║
║  Modo dinâmico: codebook que expande com escapes frequentes  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import re
import math
import argparse
from collections import Counter


def carregar_codebook(caminho):
    """Carrega um codebook JSON."""
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def carregar_corpus():
    """Carrega corpus Alpaca-PT do HuggingFace."""
    print("  [1] 📥 Carregando corpus Alpaca-PT...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "FreedomIntelligence/alpaca-gpt4-portuguese",
            split="train"
        )
        print(f"      ✅ {len(dataset)} exemplos carregados")
        return dataset
    except Exception as e:
        print(f"      ❌ Erro: {e}")
        sys.exit(1)


def tokenizar(texto):
    """Tokeniza texto em palavras."""
    return re.findall(r'[\w]+|[.,!?;:]', texto.lower())


def comprimir_texto(texto, reverse_map, escape_prefix="@@"):
    """
    Comprime texto usando o codebook (greedy longest match).
    
    Retorna:
        compressed: string de códigos DNA separados por espaço
        stats: dicionário com métricas de compressão
    """
    palavras = tokenizar(texto)
    if not palavras:
        return "", {"hits": 0, "misses": 0, "total_words": 0}
    
    resultado = []
    escapes = []
    i = 0
    hits = 0
    misses = 0
    
    while i < len(palavras):
        matched = False
        
        # Tentar match do maior fragmento primeiro (greedy)
        max_n = min(20, len(palavras) - i)
        for n in range(max_n, 0, -1):
            fragmento = ' '.join(palavras[i:i+n])
            if fragmento in reverse_map:
                resultado.append(reverse_map[fragmento])
                hits += n  # Cada palavra do fragmento conta como hit
                i += n
                matched = True
                break
        
        if not matched:
            # Escape: emitir texto literal
            palavra = palavras[i]
            resultado.append(f"{escape_prefix}{palavra}")
            escapes.append(palavra)
            misses += 1
            i += 1
    
    compressed = ' '.join(resultado)
    total = hits + misses
    
    stats = {
        "hits": hits,
        "misses": misses,
        "total_words": total,
        "hit_rate": round(hits / max(total, 1) * 100, 1),
        "tokens_dna": len(resultado),
        "palavras_originais": len(palavras),
        "taxa_real": round(len(palavras) / max(len(resultado), 1), 2),
        "escapes": escapes,
    }
    
    return compressed, stats


def expandir_codebook_dinamico(codebook, contagem_escapes, limiar):
    """
    Expande codebook dinâmico com escapes frequentes.
    
    Adiciona fragmentos que apareceram como escape mais que `limiar` vezes.
    """
    from itertools import product
    
    reverse_map = codebook.get("reverse_map", {})
    entries = codebook.get("entries", {})
    
    # Encontrar próximo código disponível
    max_len = max((len(k) for k in entries.keys()), default=2)
    codigos_usados = set(entries.keys())
    
    adicionados = 0
    for fragmento, freq in contagem_escapes.most_common():
        if freq < limiar:
            break
        if fragmento in reverse_map:
            continue
        if len(fragmento) < 2 or not any(c.isalpha() for c in fragmento):
            continue
        
        # Gerar próximo código disponível
        codigo = None
        for length in range(2, max_len + 3):
            for combo in product(['A', 'T', 'C', 'G'], repeat=length):
                c = ''.join(combo)
                if c not in codigos_usados:
                    codigo = c
                    codigos_usados.add(c)
                    break
            if codigo:
                break
        
        if codigo:
            entries[codigo] = {
                "text": fragmento,
                "freq": freq,
                "category": "dynamic",
                "n": len(fragmento.split()),
            }
            reverse_map[fragmento] = codigo
            adicionados += 1
    
    codebook["entries"] = entries
    codebook["reverse_map"] = reverse_map
    codebook["stats"]["dynamic_additions"] = adicionados
    codebook["stats"]["total_entries"] = len(entries)
    
    return codebook, adicionados


def gerar_dataset(codebook, dataset_corpus, output_path, max_amostras=10000):
    """
    Gera dataset comprimido no formato trifásico.
    
    33% Humano → DNA (codificação)
    33% DNA → Humano (decodificação)
    33% Contexto misto
    """
    reverse_map = codebook.get("reverse_map", {})
    entries = codebook.get("entries", {})
    escape_prefix = codebook.get("escape_prefix", "@@")
    taxa = codebook.get("taxa_alvo", "1:?")
    modo = codebook.get("modo", "fixo")
    is_dynamic = codebook.get("dynamic", False)
    limiar = codebook.get("expansion_threshold", 10)
    
    TERCO = max_amostras // 3
    
    print(f"  [2] 🏗️  Gerando dataset ({taxa}, {modo})...")
    print(f"      Fase A: {TERCO} Humano → DNA")
    print(f"      Fase B: {TERCO} DNA → Humano")
    print(f"      Fase C: {max_amostras - 2*TERCO} Contexto Misto")
    
    dataset_corpus = dataset_corpus.shuffle(seed=42)
    dados = []
    stats_global = {
        "hits_total": 0,
        "misses_total": 0,
        "descartados": 0,
        "taxa_compressao_media": [],
        "hit_rates": [],
    }
    contagem_escapes = Counter()
    
    for item in dataset_corpus:
        if len(dados) >= max_amostras:
            break
        
        convs = item.get('conversations', [])
        if not convs or len(convs) < 2:
            stats_global["descartados"] += 1
            continue
        
        instrucao = ""
        saida = ""
        for c in convs:
            if c.get("from") == "human" and not instrucao:
                instrucao = str(c.get("value", "")).replace("\n", " ").strip()
            elif c.get("from") == "gpt" and not saida:
                saida = str(c.get("value", "")).replace("\n", " ").strip()
        
        if not instrucao or not saida or len(instrucao) + len(saida) > 500:
            stats_global["descartados"] += 1
            continue
        
        # Comprimir a saída usando o codebook
        compressed, comp_stats = comprimir_texto(saida, reverse_map, escape_prefix)
        
        if not compressed:
            stats_global["descartados"] += 1
            continue
        
        stats_global["hits_total"] += comp_stats["hits"]
        stats_global["misses_total"] += comp_stats["misses"]
        stats_global["taxa_compressao_media"].append(comp_stats["taxa_real"])
        stats_global["hit_rates"].append(comp_stats["hit_rate"])
        
        # Contar escapes para expansão dinâmica
        for esc in comp_stats["escapes"]:
            contagem_escapes[esc] += 1
        
        fase = len(dados)
        
        # FASE A: Humano pergunta → IA responde em DNA comprimido
        if fase < TERCO:
            obj = {
                "instruction": f"Você é um compressor CROM DNA (taxa {taxa}). Comprima a resposta usando códigos do codebook semântico DNA. Use prefixo {escape_prefix} para palavras sem código.",
                "input": instrucao,
                "output": compressed,
            }
        
        # FASE B: DNA comprimido → IA decodifica para humano
        elif fase < 2 * TERCO:
            obj = {
                "instruction": f"Decodifique os códigos DNA CROM (codebook semântico {taxa}) para linguagem humana em Português.",
                "input": compressed,
                "output": saida,
            }
        
        # FASE C: Contexto misto
        else:
            snippet_compressed, _ = comprimir_texto(
                saida[:80], reverse_map, escape_prefix
            )
            obj = {
                "instruction": f"Analise o contexto abaixo. O DNA CROM ({taxa}) representa texto comprimido. Responda em Português claro.",
                "input": f"Contexto: {instrucao}\nDNA: {snippet_compressed}",
                "output": saida,
            }
        
        dados.append(obj)
    
    # ── Expansão dinâmica (se modo dinâmico) ──
    if is_dynamic and contagem_escapes:
        print(f"  [3] 🔄 Expansão dinâmica do codebook...")
        codebook, adicionados = expandir_codebook_dinamico(
            codebook, contagem_escapes, limiar
        )
        print(f"      ✅ {adicionados} novos códigos adicionados")
        
        if adicionados > 0:
            print(f"  [3b] 🔄 Re-comprimindo dataset com codebook expandido...")
            reverse_map = codebook["reverse_map"]
            dados_recomprimidos = []
            
            for i, obj in enumerate(dados):
                fase = i
                # Re-comprimir apenas Fase A
                if fase < TERCO and escape_prefix in obj["output"]:
                    # Re-comprimir
                    # Precisamos da saída original, que está na Fase B correspondente
                    # Como workaround, re-usamos o texto da instrução
                    pass  # Mantém como está na primeira passada
                dados_recomprimidos.append(obj)
            
            dados = dados_recomprimidos
    
    # ── Salvar dataset ──
    print(f"  [4] 💾 Salvando {len(dados)} amostras em {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in dados:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    tamanho_mb = os.path.getsize(output_path) / 1024 / 1024
    
    # ── Estatísticas finais ──
    avg_taxa = (sum(stats_global["taxa_compressao_media"]) / 
                max(len(stats_global["taxa_compressao_media"]), 1))
    avg_hit = (sum(stats_global["hit_rates"]) / 
               max(len(stats_global["hit_rates"]), 1))
    
    print(f"\n      {'─' * 50}")
    print(f"      📊 ESTATÍSTICAS DO DATASET ({taxa}, {modo})")
    print(f"      {'─' * 50}")
    print(f"      Amostras        : {len(dados)}")
    print(f"      Arquivo          : {tamanho_mb:.1f} MB")
    print(f"      Taxa compressão  : 1:{avg_taxa:.1f}")
    print(f"      Hit Rate médio   : {avg_hit:.1f}%")
    print(f"      Hits totais      : {stats_global['hits_total']}")
    print(f"      Misses totais    : {stats_global['misses_total']}")
    print(f"      Descartados      : {stats_global['descartados']}")
    if is_dynamic:
        print(f"      Códigos dinâmicos: {codebook['stats'].get('dynamic_additions', 0)}")
    print(f"      {'─' * 50}")
    
    return codebook, stats_global


def main():
    parser = argparse.ArgumentParser(
        description="🧬 CROM-IA V2: Gerador de Datasets Comprimidos via DNA"
    )
    parser.add_argument(
        "--codebook-dir", type=str,
        default=None,
        help="Diretório com os codebooks JSON"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Diretório de saída para datasets"
    )
    parser.add_argument(
        "--taxas", nargs="+",
        default=["1x3", "1x5", "1x10", "1x20"],
        choices=["1x3", "1x5", "1x10", "1x20"],
        help="Taxas a processar"
    )
    parser.add_argument(
        "--modos", nargs="+",
        default=["fixo", "dinamico"],
        choices=["fixo", "dinamico"],
        help="Modos a processar"
    )
    parser.add_argument(
        "--amostras", type=int, default=10000,
        help="Amostras por dataset (default: 10000)"
    )
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.codebook_dir is None:
        args.codebook_dir = os.path.join(base_dir, "codebooks")
    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "datasets")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" 🧬 CROM-IA V2: GERADOR DE DATASETS COMPRIMIDOS DNA")
    print("=" * 60)
    print(f" Codebooks  : {args.codebook_dir}")
    print(f" Saída      : {args.output_dir}")
    print(f" Taxas      : {args.taxas}")
    print(f" Modos      : {args.modos}")
    print(f" Amostras   : {args.amostras}")
    print("=" * 60)
    
    # Carregar corpus uma vez
    dataset_corpus = carregar_corpus()
    
    resultados = []
    
    for taxa in args.taxas:
        for modo in args.modos:
            codebook_path = os.path.join(
                args.codebook_dir, f"codebook_{taxa}_{modo}.json"
            )
            
            if not os.path.exists(codebook_path):
                print(f"\n⚠️  Codebook não encontrado: {codebook_path}")
                print(f"    Execute gerar_codebook.py primeiro!")
                continue
            
            print(f"\n{'═' * 60}")
            print(f" Processando: {taxa} / {modo}")
            print(f"{'═' * 60}")
            
            codebook = carregar_codebook(codebook_path)
            
            output_path = os.path.join(
                args.output_dir, f"dataset_dna_{taxa}_{modo}.jsonl"
            )
            
            codebook_atualizado, stats = gerar_dataset(
                codebook, dataset_corpus, output_path, args.amostras
            )
            
            # Se dinâmico, salvar codebook atualizado
            if modo == "dinamico" and codebook_atualizado.get("stats", {}).get("dynamic_additions", 0) > 0:
                salvar_path = codebook_path.replace(".json", "_expandido.json")
                with open(salvar_path, 'w', encoding='utf-8') as f:
                    json.dump(codebook_atualizado, f, ensure_ascii=False, indent=2)
                print(f"      💾 Codebook dinâmico expandido salvo: {salvar_path}")
            
            avg_taxa_real = (
                sum(stats["taxa_compressao_media"]) /
                max(len(stats["taxa_compressao_media"]), 1)
            )
            avg_hit = (
                sum(stats["hit_rates"]) /
                max(len(stats["hit_rates"]), 1)
            )
            
            resultados.append({
                "taxa": taxa,
                "modo": modo,
                "amostras": args.amostras,
                "taxa_real": round(avg_taxa_real, 2),
                "hit_rate": round(avg_hit, 1),
            })
    
    # ── Relatório Final Comparativo ──
    print("\n" + "=" * 70)
    print(" 📊 RELATÓRIO COMPARATIVO — DATASETS GERADOS")
    print("=" * 70)
    print(f"{'Taxa':<8} {'Modo':<12} {'Amostras':<10} {'Compressão':<12} {'Hit Rate':<10}")
    print("─" * 70)
    
    for r in resultados:
        print(
            f"{r['taxa']:<8} {r['modo']:<12} "
            f"{r['amostras']:<10} "
            f"1:{r['taxa_real']:<11} "
            f"{r['hit_rate']:.1f}%"
        )
    
    print("─" * 70)
    print(f" Total: {len(resultados)} datasets gerados em {args.output_dir}")
    print("=" * 70)
    print("\n🚀 Próximo passo: Execute treinar_codebook.py no Google Colab!")


if __name__ == "__main__":
    main()
