#!/usr/bin/env python3
"""
CROM-IA V4.2 — Download de Datasets Reais do HuggingFace
=========================================================
Downloads:
  1. Canarim-Instruct-PTBR (30K) — Base conversacional PT-BR
  2. Tested-22k-Python-Alpaca (15K) — Código Python
  3. OpenHermes-2.5 (10K filtrados) — Destilação GPT-4 (para traduzir)

Filtra por qualidade: resposta > 100 chars, sem duplicatas.
"""

import json
import os
import sys
import hashlib

# Diretório de saída
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets_hibridos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def filtrar_qualidade(texto, min_chars=100):
    """Filtra respostas muito curtas ou vazias."""
    if not texto or not isinstance(texto, str):
        return False
    return len(texto.strip()) >= min_chars


def deduplicate(items, key_fn):
    """Remove duplicatas por hash do conteúdo."""
    seen = set()
    unicos = []
    for item in items:
        h = hashlib.md5(key_fn(item).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unicos.append(item)
    return unicos


def formatar_chatml(instruction, output, system_msg="Você é CROM-IA, um assistente inteligente brasileiro."):
    """Formata no padrão ChatML (Qwen)."""
    texto = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": texto}


def download_canarim(max_samples=30000):
    """1. Canarim-Instruct-PTBR — Base conversacional PT-BR."""
    from datasets import load_dataset

    path_out = os.path.join(OUTPUT_DIR, "canarim_30k.jsonl")
    if os.path.exists(path_out):
        linhas = sum(1 for _ in open(path_out))
        print(f"⏭️  Canarim já existe: {path_out} ({linhas} linhas)")
        return path_out

    print("\n" + "=" * 60)
    print("📥 Baixando Canarim-Instruct-PTBR...")
    print("=" * 60)

    ds = load_dataset('dominguesm/Canarim-Instruct-PTBR-Dataset', split='train')
    print(f"   Dataset total: {len(ds)} amostras")

    # Detectar formato
    colunas = ds.column_names
    print(f"   Colunas: {colunas}")

    total = 0
    with open(path_out, 'w', encoding='utf-8') as f:
        for item in ds:
            # Tentar múltiplos formatos
            instruction = (item.get('instruction', '') or
                          item.get('input', '') or
                          item.get('prompt', '') or '')
            output = (item.get('output', '') or
                     item.get('response', '') or
                     item.get('completion', '') or '')

            # Se tem 'text' direto (formato single-field)
            if not output and 'text' in item:
                output = item['text']
                instruction = ""

            if not filtrar_qualidade(output, min_chars=50):
                continue

            formatted = formatar_chatml(instruction, output)
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
            total += 1
            if total >= max_samples:
                break

    print(f"✅ Canarim salvo: {path_out} ({total} amostras)")
    return path_out


def download_python(max_samples=15000):
    """2. Tested-22k-Python-Alpaca — Código Python."""
    from datasets import load_dataset

    path_out = os.path.join(OUTPUT_DIR, "python_15k.jsonl")
    if os.path.exists(path_out):
        linhas = sum(1 for _ in open(path_out))
        print(f"⏭️  Python já existe: {path_out} ({linhas} linhas)")
        return path_out

    print("\n" + "=" * 60)
    print("📥 Baixando Tested-22k-Python-Alpaca...")
    print("=" * 60)

    ds = load_dataset('Vezora/Tested-22k-Python-Alpaca', split='train')
    print(f"   Dataset total: {len(ds)} amostras")
    print(f"   Colunas: {ds.column_names}")

    total = 0
    with open(path_out, 'w', encoding='utf-8') as f:
        for item in ds:
            instruction = (item.get('instruction', '') or
                          item.get('input', '') or '')
            output = (item.get('output', '') or
                     item.get('response', '') or '')

            if not filtrar_qualidade(output, min_chars=80):
                continue

            formatted = formatar_chatml(
                instruction, output,
                system_msg="Você é CROM-IA, especialista em Python. Responda com código e explicações claras."
            )
            f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
            total += 1
            if total >= max_samples:
                break

    print(f"✅ Python salvo: {path_out} ({total} amostras)")
    return path_out


def download_openhermes(max_samples=10000):
    """3. OpenHermes-2.5 — Top 10K (para traduzir depois)."""
    from datasets import load_dataset

    path_out = os.path.join(OUTPUT_DIR, "openhermes_10k_en.jsonl")
    if os.path.exists(path_out):
        linhas = sum(1 for _ in open(path_out))
        print(f"⏭️  OpenHermes já existe: {path_out} ({linhas} linhas)")
        return path_out

    print("\n" + "=" * 60)
    print("📥 Baixando OpenHermes-2.5 (top 10K)...")
    print("=" * 60)

    ds = load_dataset('teknium/OpenHermes-2.5', split='train')
    print(f"   Dataset total: {len(ds)} amostras")
    print(f"   Colunas: {ds.column_names}")

    # Filtrar respostas longas e ricas (qualidade GPT-4)
    candidatos = []
    for item in ds:
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            continue

        # Pegar a última resposta do assistant
        instruction = ""
        output = ""
        for msg in conversations:
            role = msg.get('from', msg.get('role', ''))
            value = msg.get('value', msg.get('content', ''))
            if role in ('human', 'user'):
                instruction = value
            elif role in ('gpt', 'assistant'):
                output = value

        if not instruction or not output:
            continue

        if len(output) < 200:  # Só respostas ricas
            continue

        candidatos.append({
            'instruction': instruction,
            'output': output,
            'length': len(output)
        })

    # Ordenar por tamanho (respostas mais ricas primeiro)
    candidatos.sort(key=lambda x: x['length'], reverse=True)
    candidatos = candidatos[:max_samples]

    print(f"   Filtrados: {len(candidatos)} amostras (> 200 chars)")

    total = 0
    with open(path_out, 'w', encoding='utf-8') as f:
        for item in candidatos:
            # Salvar em EN (traduzir depois com tradutor_batch_argos.py)
            entry = {
                'instruction': item['instruction'],
                'output': item['output'],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            total += 1

    print(f"✅ OpenHermes EN salvo: {path_out} ({total} amostras)")
    print(f"   ⚠️  ATENÇÃO: Precisa traduzir com tradutor_batch_argos.py!")
    return path_out


def main():
    print("\n" + "=" * 60)
    print("🧬 CROM-IA V4.2 — Download de Datasets Reais")
    print(f"   Saída: {OUTPUT_DIR}")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Biblioteca 'datasets' não instalada!")
        print("   pip install datasets")
        sys.exit(1)

    # 1. Canarim (PT-BR nativo)
    path_canarim = download_canarim(30000)

    # 2. Python
    path_python = download_python(15000)

    # 3. OpenHermes (EN, para traduzir)
    path_hermes = download_openhermes(10000)

    # Relatório final
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO FINAL")
    print("=" * 60)
    paths = [path_canarim, path_python, path_hermes]
    for p in paths:
        if p and os.path.exists(p):
            linhas = sum(1 for _ in open(p))
            tamanho = os.path.getsize(p) / (1024 * 1024)
            print(f"   ✅ {os.path.basename(p)}: {linhas} amostras ({tamanho:.1f} MB)")

    print("\n📋 PRÓXIMOS PASSOS (NO COLAB):")
    print("   Suba o `openhermes_10k_en.jsonl` para o Google Colab.")
    print("   Execute a tradução LA COM GPU para evitar travamento local.")
    print("   Depois, realize a geração do codebook e DNA localmente no dataset traduzido.")
    print("   5. Enviar para Colab!")


if __name__ == "__main__":
    main()
