#!/usr/bin/env python3
"""
🧬 CROM-IA V3: Extrator Termodinâmico de Macro-Conceitos (Stream O(1))
Varre um Dataset gigante em busca de sentenças e blocos completos (>20 chars)
preservando a indentação do Python (Macro-Tokens RAG Dimensional).
"""

import sys
import re
import json
import ujson
from collections import Counter
import os

def iterar_corpus(caminho):
    print(f"📥 Lendo Jsonl Local via Yield: {caminho}")
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            for line in f:
                linha = line.strip()
                if not linha: continue
                obj = ujson.loads(linha)
                yield obj['content']
    except Exception as e:
        print(f"❌ Erro ao ler corpus local: {e}")
        sys.exit(1)

def extrair_linhas_python(texto):
    """
    Divide o texto preservando a formatação essencial de código.
    Divide apenas por novas linhas reais e elimina linhas muito pequenas ou irrelevantes.
    """
    linhas_limpas = []
    # Quebra por \n exato
    blocos = texto.split('\n')
    for b in blocos:
        # rstrip() pois o trailing space nao importa, 
        # mas o leading space (indentação) é vital no Python.
        b_sanitizado = b.rstrip()
        
        # Ignora linhas muito curtas (macros que nao economizam nada)
        if len(b_sanitizado) >= 20:
            linhas_limpas.append(b_sanitizado)
            
    return linhas_limpas

def analisar_massivamente(iterador_respostas):
    print("🔍 Extraindo e mapeando blocos frasais massivos...")
    contador = Counter()
    
    total_lidas = 0
    for resp in iterador_respostas:
        if total_lidas % 2000 == 0:
            print(f"   Processando documento {total_lidas}...")
            
        sentencas = extrair_linhas_python(resp)
        contador.update(sentencas)
        total_lidas += 1
        
    print(f"✅ Varredura finalizada. Documentos processados: {total_lidas}")
    
    # Filtra apenas o que repete muito (Pelo menos 3x, ideal > 5x)
    # Aumentando filtro para focar em blocos realmente pesados e boilerplates absolutos
    blocos_frequentes = {b: freq for b, freq in contador.items() if freq >= 3}
    
    # Ordena por (tamanho_do_bloco * frequencia) -> Economia real em bytes na VFS
    blocos_ranqueados = sorted(
        blocos_frequentes.items(),
        key=lambda x: len(x[0]) * x[1],
        reverse=True
    )
    
    return blocos_ranqueados

def salvar_blocos(blocos_ranqueados, output_file="blocos_extraidos_v3.json"):
    print(f"💾 Salvando {len(blocos_ranqueados)} macros no disco...")
    
    dados = []
    for bloco, freq in blocos_ranqueados:
        dados.append({
            "texto": bloco,
            "freq": freq,
            "bytes_salvos": len(bloco) * freq
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        ujson.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"✅ Arquivo {output_file} gerado com sucesso!")
    
    # Showcase dos top 10
    print("\n🏆 Top 10 Macros (Mais Economia de Bytes):")
    for b in dados[:10]:
        print(f"   - Ocorrências: {b['freq']} | Economia: {b['bytes_salvos']} bytes | Len: {len(b['texto'])}")
        print(f"   >>> {b['texto']}")
        print("   ---")

if __name__ == "__main__":
    caminho_dataset = "data/raw_corpus/python_extremado_corpus.jsonl"
    if not os.path.exists(caminho_dataset):
        print(f"Erro: Dataset não encontrado em {caminho_dataset}.")
        print("Execute primeiro a Task 1 (downloader_dataset_bruto.py)")
        sys.exit(1)
        
    respostas = iterar_corpus(caminho_dataset)
    blocos = analisar_massivamente(respostas)
    salvar_blocos(blocos)
