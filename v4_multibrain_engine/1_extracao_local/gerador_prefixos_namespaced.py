#!/usr/bin/env python3
"""
CROM-IA V4.0: Extrator de Conhecimento Dimensional Namespaced
Uso:
  python3 gerador_prefixos_namespaced.py <domínio> <arquivo.jsonl>
Exemplo:
  python3 gerador_prefixos_namespaced.py P dataset_python.jsonl
Onde "P" representa o Domínio (Python). "J" para Jurídico, etc.
Gera chaves: @@PW... (Palavras), @@PF... (Frases), @@PP... (Parágrafos)
"""

import sys
import json
import re
from collections import Counter

def get_ngrams(texto, min_n, max_n):
    tokens = re.findall(r'\w+|\s+|[^\w\s]', texto)
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            chunk = "".join(tokens[i:i+n])
            if len(chunk.strip()) > 4: 
                ngrams.append(chunk)
    return ngrams

def extrator_hierarquico(caminho_dataset, sigla_dominio="P"):
    print(f"🔬 [CROM-IA V4.0] Iniciando Extração Massiva para o Domínio '{sigla_dominio}'")
    
    cnt_words = Counter()
    cnt_phrases = Counter()
    cnt_paragraphs = Counter()
    
    lines_processed = 0
    with open(caminho_dataset, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                target = data.get("output", "")
            except:
                target = line.strip() 
                
            if target:
                blocos_brutos = target.split("\n")
                for bloco in blocos_brutos:
                    if len(bloco.strip()) < 3: continue
                    
                    if len(bloco) > 60:
                        cnt_paragraphs.update(get_ngrams(bloco, 15, 25))
                        cnt_phrases.update(get_ngrams(bloco, 6, 12))
                    if len(bloco) > 20:
                        cnt_phrases.update(get_ngrams(bloco, 4, 8))
                    
                    tokens_words = re.findall(r'\b\w{6,}\b', bloco)
                    cnt_words.update(tokens_words)
            
            lines_processed += 1
            if lines_processed > 30000: break # limite de RAM para amostra perfeita
            
    # Construtor da base Radix-4 para o suffix do ponteiro
    radix = ['A', 'T', 'C', 'G']
    def hash_sufixo(idx):
        if idx < 16: return radix[idx // 4] + radix[idx % 4]
        return radix[(idx // 16) % 4] + radix[(idx // 4) % 4] + radix[idx % 4]
        
    def export_codebook(contador, identifier, hierarquia_letra):
        entries = {}
        idx = 0
        for text, freq in contador.most_common(2000): # Top 2k
            if freq < 8: break # Reduzido threshold para permitir multi-LoRAs super especificados
            if len(text.strip()) >= 5: # Filtro de poeira
                sufixo = hash_sufixo(idx)
                chave = f"{identifier}{hierarquia_letra}{sufixo}" # Ex: @@PWAT
                entries[chave] = {"text": text, "freq": freq, "bytes": len(text)}
                idx += 1
                if idx >= 180: break # Limite fisico do Llama dictionary de cada Cérebro 
                
        output_file = f"codebook_{identifier}_{hierarquia_letra}_V4.json"
        
        payload = {
            "version": "4.0",
            "domain": identifier,
            "hierarchy": hierarquia_letra,
            "escape_prefix": "@@",
            "entries": entries
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"📦 Sucesso: {output_file} gerado com {len(entries)} ponteiros.")
        
    print(f"\n--- Exportando Plug-ins Cerebrais (Zero-Retreino) ---")
    export_codebook(cnt_words, sigla_dominio, "W") # Word
    export_codebook(cnt_phrases, sigla_dominio, "F") # Frase
    export_codebook(cnt_paragraphs, sigla_dominio, "P") # Paragrafo
    
    print("\n✅ V4.0 Finalizado! Codebooks separados metricamente prontos para Nuvem.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Faltam argumentos! Uso: script.py <SiglaDominio> <arquivo.jsonl>")
        sys.exit(1)
    extrator_hierarquico(sys.argv[2], sys.argv[1])
