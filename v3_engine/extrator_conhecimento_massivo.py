#!/usr/bin/env python3
"""
🧬 CROM-IA V3: Extrator Termodinâmico de Macro-Conceitos
Varre um Dataset gigante (Alpaca-GPT4 PT) em busca de sentenças e blocos completos (>40 chars)
que se repetem frequentemente, caracterizando-os como 'Macros'.
"""

import sys
import re
import json
from collections import Counter

def carregar_corpus():
    print("📥 Baixando/Carregando corpus Alpaca-PT...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-portuguese", split="train")
        respostas = []
        for item in dataset:
            for c in item.get('conversations', []):
                if c.get('from') == 'gpt':
                    texto = str(c.get('value', '')).strip()
                    respostas.append(texto)
        print(f"✅ {len(respostas)} respostas extraídas do dataset.")
        return respostas
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        sys.exit(1)

def extrair_sentencas(texto):
    """
    Divide o texto em sentenças. Usa heurística simples com pontuação.
    """
    # Quebra por . ! ? \n
    blocos = re.split(r'[\.\!\?\n]+', texto)
    sentencas_limpas = []
    for b in blocos:
        b = b.strip()
        # Sentenças úteis para RAG Dimensional costumam ter acima de 30-40 caracteres
        if len(b) > 40 and len(b.split()) > 5:
            # Substitui excesso de espaço
            b = re.sub(r'\s+', ' ', b)
            sentencas_limpas.append(b)
    return sentencas_limpas

def analisar_massivamente(respostas):
    print("🔍 Extraindo e mapeando blocos frasais massivos...")
    contador = Counter()
    
    total = len(respostas)
    for i, resp in enumerate(respostas):
        if i % 10000 == 0:
            print(f"   Processando {i}/{total}...")
        sentencas = extrair_sentencas(resp)
        contador.update(sentencas)
    
    # Filtra apenas o que repete muito (Pelo menos 3x, ideal > 5x)
    blocos_frequentes = {b: freq for b, freq in contador.items() if freq >= 3}
    
    # Ordena por (tamanho_do_bloco * frequencia) -> Economia real em bytes
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
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"✅ Arquivo {output_file} gerado com sucesso!")
    
    # Showcase dos top 5
    print("\n🏆 Top 5 Macros (Mais Economia de Bytes):")
    for b in dados[:5]:
        print(f"   - Ocorrências: {b['freq']} | Economia: {b['bytes_salvos']} bytes")
        print(f"   - [{b['texto']}]")
        print("   ---")

if __name__ == "__main__":
    respostas = carregar_corpus()
    blocos = analisar_massivamente(respostas)
    salvar_blocos(blocos)
