#!/usr/bin/env python3
"""
CROM-IA V3.6: Extrator de N-Grams Shannon (Codebook Generator)

Uso:
  python3 gerador_codebook.py <caminho_dataset.jsonl>

Este script busca padrões gigantescos e massivos no seu Dataset 
(focado no bloco de respostas 'output'), e ejetará automaticamente 
um dicionário "macro_codebook_v4.json" apenas com matemática densa.
"""

import sys
import json
from collections import Counter
import re

def extract_ngrams(text, min_len=10, max_len=60):
    # Regex brutal para separar quebras logicas e tokens de codigo
    # Adaptado para python
    tokens = re.split(r'(\n|\s{4})', text)
    ngrams = []
    
    # Simula N-Grams de palavras + whitespace em sliding window
    for window_size in range(3, 15): 
        for i in range(len(tokens) - window_size + 1):
            block = "".join(tokens[i:i+window_size])
            if min_len <= len(block) <= max_len:
                ngrams.append(block)
    return ngrams

def build_codebook(dataset_path, output_path="macro_codebook_v4.json"):
    print("🔬 [CROM-IA] Mapeando Entropia e Gerando N-Grams...")
    counter = Counter()
    total_samples = 0
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # O CROM so atua sobre o output da IA!
                target_text = data.get("output", "")
                if target_text:
                    ngrams = extract_ngrams(target_text)
                    counter.update(ngrams)
            except: pass
            total_samples += 1
            if total_samples % 1000 == 0:
                print(f"Processando amostra {total_samples}...")

    print(f"✅ Analise concluida! Filtrando o Ouro Sintatico...")
    
    # Criterio da V3.6: Minimo 50 chars ganhos vs 10 perdidos. Exige Altissima freq.
    MIN_FREQ = 150
    dict_entries = {}
    
    # Gerador sequencial de chaves Radix-4 (AA, AT, AC, AG, TA...)
    radix = ['A', 'T', 'C', 'G']
    def get_radix_key(index):
        if index < 16: return radix[index // 4] + radix[index % 4]
        # Aqui a logica avanca pra 3 blocos (AAA) ou mais
        # No script oficial teremos um conversor Base-4 matematico puro.
        return radix[(index // 16) % 4] + radix[(index // 4) % 4] + radix[index % 4]
    
    idx = 0
    for block, freq in counter.most_common(10000): # Top 10K Ngrams brutos
        if freq < MIN_FREQ: break
        
        # Matematica: Se tem espacos em branco ou quebras e tamanho solido
        if len(block) >= 12 and block.strip() != "":
            key = get_radix_key(idx)
            dict_entries[key] = {
                "text": block,
                "freq": freq,
                "bytes": len(block)
            }
            idx += 1
            if idx > 200: break # Limite fisico do FUSE O(1) de memoria rapida
            
    # Salva
    final_payload = {
        "version": "4.0",
        "description": "Codebook Shannon Numerico V3.6",
        "escape_prefix": "@@",
        "entries": dict_entries
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
        
    print(f"💿 CROM-IA V3.6 Codebook salvo em {output_path} com {len(dict_entries)} entradas!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Falta o caminho do jsonl!")
        sys.exit(1)
    build_codebook(sys.argv[1])
