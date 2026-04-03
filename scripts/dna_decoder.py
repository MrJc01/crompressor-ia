#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 CROM-IA V2: Decoder Streaming DNA → Texto Humano       ║
║                                                              ║
║  Recebe tokens DNA do llama-cli via stdin (pipe)             ║
║  Decodifica usando codebook semântico (hashmap O(1))         ║
║  Emite texto humano em tempo real via stdout                 ║
║  Emite métricas SRE no stderr ao final                       ║
╚══════════════════════════════════════════════════════════════╝

Uso:
  llama-cli -m model.gguf -p "..." | python3 dna_decoder.py --codebook codebook_1x3_fixo.json
"""

import os
import sys
import json
import time
import argparse


def carregar_codebook(caminho):
    """Carrega codebook e prepara hashmap de lookup."""
    with open(caminho, 'r', encoding='utf-8') as f:
        codebook = json.load(f)
    
    entries = codebook.get("entries", {})
    escape = codebook.get("escape_prefix", "@@")
    taxa = codebook.get("taxa_alvo", "?")
    modo = codebook.get("modo", "?")
    
    sys.stderr.write(f"[CROM DNA Decoder]\n")
    sys.stderr.write(f"  Codebook : {os.path.basename(caminho)}\n")
    sys.stderr.write(f"  Taxa     : {taxa}\n")
    sys.stderr.write(f"  Modo     : {modo}\n")
    sys.stderr.write(f"  Entradas : {len(entries)}\n")
    sys.stderr.write(f"  Escape   : {escape}\n")
    sys.stderr.write(f"  ─────────────────────────\n")
    sys.stderr.flush()
    
    return entries, escape, taxa, modo


def decoder_streaming(entries, escape_prefix, max_buffer=10):
    """
    Decodifica stream de tokens DNA do stdin.
    
    Protocolo:
    - Tokens são separados por espaços ou newlines
    - Tokens que começam com o escape_prefix são texto literal
    - Tokens que existem no codebook são expandidos
    - Tokens desconhecidos são marcados como [?token]
    """
    stats = {
        "tokens_recebidos": 0,
        "tokens_validos": 0,
        "tokens_escape": 0,
        "tokens_alucinados": 0,
        "palavras_emitidas": 0,
        "t_inicio": time.time(),
        "t_primeiro_token": None,
    }
    
    buffer = ""
    
    for char in sys.stdin:
        for c in char:
            if c in (' ', '\n', '\t'):
                if buffer:
                    _processar_token(buffer, entries, escape_prefix, stats)
                    buffer = ""
            else:
                buffer += c
                
                # Safety: se buffer ficou muito grande, é alucinação
                if len(buffer) > 20:
                    stats["tokens_alucinados"] += 1
                    stats["tokens_recebidos"] += 1
                    sys.stdout.write(f"[?{buffer}] ")
                    sys.stdout.flush()
                    buffer = ""
    
    # Flush final
    if buffer:
        _processar_token(buffer, entries, escape_prefix, stats)
    
    return stats


def _processar_token(token, entries, escape_prefix, stats):
    """Processa um token individual."""
    stats["tokens_recebidos"] += 1
    
    if stats["t_primeiro_token"] is None:
        stats["t_primeiro_token"] = time.time()
    
    # Modo escape: texto literal
    if token.startswith(escape_prefix):
        texto = token[len(escape_prefix):]
        sys.stdout.write(texto + " ")
        sys.stdout.flush()
        stats["tokens_escape"] += 1
        stats["palavras_emitidas"] += 1
        return
    
    # Lookup no codebook
    if token in entries:
        texto = entries[token]["text"]
        n_palavras = entries[token].get("n", len(texto.split()))
        sys.stdout.write(texto + " ")
        sys.stdout.flush()
        stats["tokens_validos"] += 1
        stats["palavras_emitidas"] += n_palavras
        return
    
    # Token desconhecido (alucinação)
    sys.stdout.write(f"[?{token}] ")
    sys.stdout.flush()
    stats["tokens_alucinados"] += 1


def emitir_metricas(stats, taxa, modo, codebook_name):
    """Emite métricas SRE no stderr."""
    t_total = time.time() - stats["t_inicio"]
    t_primeiro = (stats["t_primeiro_token"] - stats["t_inicio"]
                  if stats["t_primeiro_token"] else 0)
    
    total = stats["tokens_recebidos"]
    validos = stats["tokens_validos"]
    escapes = stats["tokens_escape"]
    alucinados = stats["tokens_alucinados"]
    palavras = stats["palavras_emitidas"]
    
    hit_rate = (validos / max(total, 1)) * 100
    alucinacao = (alucinados / max(total, 1)) * 100
    taxa_real = palavras / max(total, 1)
    palavras_por_s = palavras / max(t_total, 0.001)
    
    sys.stderr.write(f"\n")
    sys.stderr.write(f"{'═' * 55}\n")
    sys.stderr.write(f"  [CROM DNA Decoder] RELATÓRIO FINAL\n")
    sys.stderr.write(f"{'═' * 55}\n")
    sys.stderr.write(f"  Codebook          : {codebook_name}\n")
    sys.stderr.write(f"  Taxa alvo         : {taxa}\n")
    sys.stderr.write(f"  Modo              : {modo}\n")
    sys.stderr.write(f"  ─────────────────────────\n")
    sys.stderr.write(f"  Tokens DNA in     : {total}\n")
    sys.stderr.write(f"  Tokens válidos    : {validos} ({hit_rate:.1f}%)\n")
    sys.stderr.write(f"  Tokens escape     : {escapes}\n")
    sys.stderr.write(f"  Tokens alucinados : {alucinados} ({alucinacao:.1f}%)\n")
    sys.stderr.write(f"  ─────────────────────────\n")
    sys.stderr.write(f"  Palavras emitidas : {palavras}\n")
    sys.stderr.write(f"  Taxa real         : 1:{taxa_real:.1f}\n")
    sys.stderr.write(f"  Velocidade        : {palavras_por_s:.1f} palavras/s\n")
    sys.stderr.write(f"  Latência 1º token : {t_primeiro*1000:.0f} ms\n")
    sys.stderr.write(f"  Tempo total       : {t_total:.2f} s\n")
    sys.stderr.write(f"{'═' * 55}\n")
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="🧬 CROM-IA V2: DNA Decoder Streaming"
    )
    parser.add_argument(
        "--codebook", type=str, required=True,
        help="Caminho para o codebook JSON"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suprime métricas no stderr"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.codebook):
        sys.stderr.write(f"❌ Codebook não encontrado: {args.codebook}\n")
        sys.exit(1)
    
    entries, escape, taxa, modo = carregar_codebook(args.codebook)
    
    try:
        stats = decoder_streaming(entries, escape)
    except KeyboardInterrupt:
        stats = {"tokens_recebidos": 0, "tokens_validos": 0,
                 "tokens_escape": 0, "tokens_alucinados": 0,
                 "palavras_emitidas": 0, "t_inicio": time.time(),
                 "t_primeiro_token": None}
    
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    if not args.quiet:
        emitir_metricas(stats, taxa, modo, os.path.basename(args.codebook))


if __name__ == "__main__":
    main()
