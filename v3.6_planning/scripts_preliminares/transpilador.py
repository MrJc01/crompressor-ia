#!/usr/bin/env python3
"""
CROM-IA V3.6: Transpilador Híbrido Text-DNA (Dataset Mutator)

Uso:
  python3 transpilador.py <input_dataset.jsonl> <codebook.json> <output_mutante.jsonl>

Ele pega o seu Dataset normal (Alpaca, ShareGPT), intercepta as respostas 
"output" da IA, injecta os marcadores `@@XX` com base no tamanho das matrizes,
e re-salva um Dataset que forçará o SFT do Qwen2.5 a "aprender a comprimir" por osmose.
"""

import sys
import json

def load_codebook(cb_path):
    with open(cb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    prefix = data.get("escape_prefix", "@@")
    
    # IMPORTANTE: Ordenar pelas macros MAIS LONGAS primeiro.
    # Ex: Para não substituir "def " quando deveria ser "def __init__(self):"
    
    entries = []
    for key, val in data["entries"].items():
        entries.append((f"{prefix}{key}", val["text"]))
        
    entries.sort(key=lambda x: len(x[1]), reverse=True)
    return prefix, entries

def transpile_dataset(input_path, cb_path, output_path):
    prefix, entries = load_codebook(cb_path)
    
    total = 0
    modificados = 0
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
             
        for line in fin:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                original_output = data.get("output", "")
                
                if original_output:
                    new_output = original_output
                    hit = False
                    
                    # Roda o Swap DNA reverso
                    for ptr, text_block in entries:
                        if text_block in new_output:
                            new_output = new_output.replace(text_block, ptr)
                            hit = True
                    
                    data["output"] = new_output
                    if hit: modificados += 1
                
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                total += 1
                
            except Exception as e:
                print(f"Erro em linha: {e}")
                
    print(f"\n🧬 [Transpilador O(1)] Concluído!")
    print(f"Dataset salvo em: {output_path}")
    print(f"Total Amostras: {total} | Amostras Mutadas: {modificados}")
    print(f"O Dataset agora está pronto para Finetune LoRA/Unsloth.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: transpilador.py <entrada.jsonl> <codebook.json> <saida.jsonl>")
        sys.exit(1)
        
    transpile_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
