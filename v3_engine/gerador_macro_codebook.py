#!/usr/bin/env python3
"""
🧬 CROM-IA V3: Gerador do RAG Dimensional
Transforma as macros extraídas em um Dicionário V3 utilizando
"Memory Pointers" Base-4 (ATCG) de forma ultra-compacta.
"""

import json
import os
from itertools import product

DNA_ALPHABET = ['A', 'T', 'C', 'G']
ESCAPE_PREFIX = "@@"

def gerar_ponteiros_dna(quantidade):
    """
    Na V3, tentamos usar o menor número de caracteres possível
    Cadeias de 2 letras dão 16 chaves (4^2).
    Cadeias de 3 letras dão 64 chaves (4^3).
    Cadeias de 4 letras dão 256 chaves.
    """
    codigos = []
    tamanho = 2
    
    while len(codigos) < quantidade:
        for combo in product(DNA_ALPHABET, repeat=tamanho):
            codigos.append(''.join(combo))
            if len(codigos) >= quantidade:
                break
        tamanho += 1
        
    return codigos[:quantidade]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "blocos_extraidos_v3.json")
    
    if not os.path.exists(input_file):
        print(f"❌ Arquivo {input_file} não encontrado!")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        blocos = json.load(f)
        
    print(f"📥 {len(blocos)} macros carregadas.")
    
    # Gerando os "Ponteiros de Memória" (Tamanho Válido = Entradas)
    ponteiros = gerar_ponteiros_dna(len(blocos))
    
    macro_codebook = {
        "version": "3.0",
        "description": "Dicionário RAG Dimensional V3",
        "escape_prefix": ESCAPE_PREFIX,
        "entries": {},
        "stats": {
            "total_macros": len(blocos),
            "total_bytes_economizados_corpus": sum(b["bytes_salvos"] for b in blocos)
        }
    }
    
    for i, macro in enumerate(blocos):
        pointer = ponteiros[i]
        macro_codebook["entries"][pointer] = {
            "text": macro["texto"],
            "freq": macro["freq"],
            "bytes": len(macro["texto"]),
            "compression_ratio": round(len(macro["texto"]) / len(pointer), 2)
        }
        
    output_file = os.path.join(base_dir, "macro_codebook_v3.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(macro_codebook, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Macro-Codebook V3 Gerado com Sucesso: {output_file}")
    
    # Print showcase
    print("\n🔍 Demonstração (Mapeamento Pointer -> RAM):")
    for pointer, data in list(macro_codebook["entries"].items())[:5]:
        print(f"   {pointer:<4} ➔ 🚀 Expansão de {data['bytes']} bytes (Compressão 1:{data['compression_ratio']})")
        print(f"          \"{data['text']}\"")
        print("          ---")

if __name__ == "__main__":
    main()
