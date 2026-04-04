#!/usr/bin/env python3
"""
🧬 CROM-IA V3.5: Dataset Mixer (100k Real + V3 Code)
Combina dados reais de chat do HuggingFace com código Python V3 comprimido.
"""
import json
import os
import sys
import random

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw_corpus")
    codebook_file = os.path.join(base_dir, "macro_codebook_v3.json")
    output_dataset = os.path.join(base_dir, "data", "dataset_v3_lora.jsonl")
    
    # Fontes de dados reais
    chat_files = [
        os.path.join(raw_dir, "chat_alpaca_pt.jsonl"),
        os.path.join(raw_dir, "chat_alpaca_gpt4_pt.jsonl"),
        os.path.join(raw_dir, "rosa_identidade.jsonl"),
    ]
    code_file = os.path.join(raw_dir, "python_extremado_corpus.jsonl")
    
    # 1. Carregar Codebook V3
    if not os.path.exists(codebook_file):
        print(f"❌ Falta {codebook_file}")
        sys.exit(1)
    with open(codebook_file, 'r', encoding='utf-8') as f:
        codebook = json.load(f)
    prefix = codebook["escape_prefix"]
    macros_ordenadas = sorted(
        codebook["entries"].items(),
        key=lambda x: len(x[1]["text"]),
        reverse=True
    )

    dados_finais = []

    # 2. Carregar TODOS os datasets de chat reais
    print("⚙️ Carregando datasets de chat reais...")
    for cf in chat_files:
        if not os.path.exists(cf):
            print(f"   ⚠️ Arquivo não encontrado: {cf}")
            continue
        with open(cf, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                obj = json.loads(line)
                # Padronizar formato
                dados_finais.append({
                    "instruction": obj.get("instruction", "Responda ao usuário."),
                    "input": obj.get("input", ""),
                    "output": obj.get("output", "")
                })
                count += 1
            print(f"   ✅ {count} amostras de {os.path.basename(cf)}")

    chat_total = len(dados_finais)
    print(f"📊 Total Chat Real: {chat_total}")

    # 3. Carregar e Comprimir CÓDIGO V3 (limite ~18k)
    print("⚙️ Processando amostras de Código V3...")
    if os.path.exists(code_file):
        with open(code_file, 'r', encoding='utf-8') as f:
            code_count = 0
            for line in f:
                obj = json.loads(line)
                texto_comprimido = obj['content']
                for pointer, meta in macros_ordenadas:
                    macro_str = meta["text"]
                    pointer_str = f"{prefix}{pointer}"
                    texto_comprimido = texto_comprimido.replace(macro_str, pointer_str)
                dados_finais.append({
                    "instruction": obj.get("instruction", "Escreva o código Python solicitado."),
                    "input": obj.get("prompt", ""),
                    "output": texto_comprimido
                })
                code_count += 1
        print(f"   ✅ {code_count} amostras de Código V3")
    else:
        print(f"   ⚠️ Código V3 não encontrado: {code_file}")

    # 4. SHUFFLE TOTAL
    print("🔀 Embaralhando todo o dataset...")
    random.shuffle(dados_finais)

    # 5. Gravar
    with open(output_dataset, 'w', encoding='utf-8') as f:
        for d in dados_finais:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"✅ DATASET FINAL: {len(dados_finais)} amostras")
    print(f"   Chat Real: {chat_total}")
    print(f"   Código V3: {len(dados_finais) - chat_total}")
    print(f"   Arquivo: {output_dataset}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
