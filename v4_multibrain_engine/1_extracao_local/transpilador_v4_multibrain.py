#!/usr/bin/env python3
"""
CROM-IA V4.0: Transpilador Híbrido Multi-Brain
Uso:
  python3 transpilador_v4_multibrain.py <input.jsonl> <codebook_W.json> <codebook_F.json> <codebook_P.json> <saída_híbrida.jsonl>
  
Ele carregará todos os Cérebros de um domínio simultaneamente e fará Swap 50/50!
"""

import sys
import json
import random

def carregar_codebook(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefix = data.get("escape_prefix", "@@")
        
        entries = []
        for key, ptr_data in data["entries"].items():
            # A nova key do v4 ja contem a hierarquia PW, PF (Ex: PWA, PFA)
            ptr = f"{prefix}{key}" 
            text_block = ptr_data["text"]
            entries.append((ptr, text_block))
        return entries
    except Exception as e:
        print(f"⚠️ Erro ao carregar {caminho}: {e}")
        return []

def orquestrador_transpile(input_file, cbs_paths, output_file):
    print("📡 [CROM-IA V4] Carregando a Matrix Multi-Brain no Transpilador...")
    entries_globais = []
    
    for cb_path in cbs_paths:
        entries_globais.extend(carregar_codebook(cb_path))
        
    # Ordenar por tamanho do texto em blocos maiores primeiro
    # Evita que o P (Palavra) destrua regras da Frase (F)
    entries_globais.sort(key=lambda x: len(x[1]), reverse=True)
    print(f"🧬 Codebooks Engatados! Total Ponteiros Disponíveis: {len(entries_globais)}")
    
    lines_processed = 0
    modificados = 0
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
             
        for line in fin:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                outp = data.get("output", "")
                
                if outp:
                    # Rota de Decisão MLOps (50/50)
                    if random.random() < 0.50:
                        teve_mutacao = False
                        
                        for ptr, text_block in entries_globais:
                            if text_block in outp:
                                outp = outp.replace(text_block, ptr)
                                teve_mutacao = True
                                
                        data["output"] = outp
                        if teve_mutacao: 
                            modificados += 1
                            
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                lines_processed += 1
            except: pass
            
    print(f"\n✅ Concluído! Transpilação Híbrida concluída.")
    print(f"Total avaliado: {lines_processed} | Sofreu Mutação (50% Chance Base): {modificados}")
    print(f"O Arquivo '{output_file}' está pronto e otimizado para upload no Colab V4.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Uso incorreto. Exemplo: script.py original.jsonl W.json F.json P.json output.jsonl")
        sys.exit(1)
        
    in_file = sys.argv[1]
    ponts = sys.argv[2:-1] # Todos os arquivos intermediários
    out_file = sys.argv[-1]
    
    orquestrador_transpile(in_file, ponts, out_file)
