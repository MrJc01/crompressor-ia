#!/usr/bin/env python3
"""
CROM-IA V4.0: Minerador de Micro-Cérebros Base (Data Harvester)
Baixa sub-datasets fortíssimos em streaming para evitar sobrecarga local, 
formatando-os para o exigido pelo Transpilador DNA.
"""

import os
import json
import traceback
try:
    from datasets import load_dataset
except ImportError:
    print("⚠️ Faltando biblioteca. Rode: pip install datasets")
    import sys; sys.exit(1)

def baixar_cerebro(hf_name, output_file, name="Geral", max_samples=8000):
    print(f"\n📥 Acionando Antena para Cérebro {name} ({hf_name})...")
    try:
        # Usa streaming para nao baixar gigabtyes para RAM
        dataset = load_dataset(hf_name, split="train", streaming=True)
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                if count >= max_samples: break
                
                # Harmonizaçao para Formato Alpaca
                # Tenta multiplas chaves comuns dos datasets do HuggingFace
                instr = item.get("instruction", "") or item.get("question", "") or item.get("text", "")
                inp = item.get("input", "")
                outp = item.get("output", "") or item.get("answer", "") or item.get("response", "")
                
                # Exige respostas com alguma densidade para justificar entropia
                if outp and len(outp) > 15:
                    linha = {"instruction": instr, "input": inp, "output": outp}
                    f.write(json.dumps(linha, ensure_ascii=False) + "\n")
                    count += 1
        
        print(f"✅ Disco Frio: {output_file} capturou {count} sinapses brutas.")
    except Exception as e:
        print(f"❌ Falha de Antena {hf_name}. Erro: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    dir_bruto = "datasets_brutos"
    os.makedirs(dir_bruto, exist_ok=True)
    
    # 1. Cérebro de Computação (Código Python Exato)
    baixar_cerebro("iamtarun/python_code_instructions_18k_alpaca", 
                   f"{dir_bruto}/P_python.jsonl", "Linguagem Python", 8000)
                   
    # 2. Cérebro Clínico/Biomédico (Medicina em Flashcards)
    baixar_cerebro("medalpaca/medical_meadow_medical_flashcards", 
                   f"{dir_bruto}/M_medicina.jsonl", "Biomedicina", 8000)
                   
    # 3. Cérebro Analítico PT-BR (GPT-4 Distilled Português)
    baixar_cerebro("FreedomIntelligence/alpaca-gpt4-portuguese", 
                   f"{dir_bruto}/G_geral_ptbr.jsonl", "Geral/Conversação PT-BR", 8000)
    
    print("\n🚀 DOWNLOADS CONCLUÍDOS. Bases brutas prontas para Transpilação DNA.")
