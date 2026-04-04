#!/usr/bin/env python3
"""
CROM-IA V4.1 — Adicionar Cérebro Rápido
=========================================
Uso: python3 adicionar_cerebro.py "Nome_Cerebro" /caminho/pasta_ou_arquivo

Aceita:
  - Pasta com .txt, .md, .py, .json, .jsonl
  - Arquivo individual .jsonl (já formatado)
  - Arquivo .txt ou .md (auto-converte para Q&A)

Pipeline automático:
  1. Coleta e converte tudo para JSONL
  2. Gera codebook inteligente (frequência real)
  3. Transpila com DNA 75%
  4. Gera arquivo pronto para treino no Colab
"""

import os
import sys
import json
import re
import glob

# Importar ferramentas locais
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "1_extracao_local"))

from gerador_codebook_v41 import gerar_codebook
from transpilador_v41 import transpilar_dataset_v41, carregar_codebook


def texto_para_qa(texto, titulo="documento"):
    """Converte texto livre em pares instruçao/resposta."""
    pares = []
    secoes = re.split(r'\n(?=#{1,3}\s)', texto)

    for secao in secoes:
        secao = secao.strip()
        if len(secao) < 50:
            continue

        # Extrair título da seção
        match = re.match(r'^#{1,3}\s+(.+)', secao)
        titulo_secao = match.group(1) if match else titulo

        # Limitar tamanho
        conteudo = secao[:2000]

        # Gerar variações de perguntas
        perguntas = [
            f"Explique sobre {titulo_secao}.",
            f"O que é {titulo_secao}?",
            f"Descreva {titulo_secao} em detalhes.",
        ]

        for p in perguntas:
            pares.append({"instruction": p, "output": conteudo})

    # Se não encontrou seções, tratar como bloco único
    if not pares and len(texto) > 50:
        blocos = [texto[i:i+1500] for i in range(0, len(texto), 1500)]
        for i, bloco in enumerate(blocos):
            pares.append({
                "instruction": f"Explique o conteúdo do {titulo} (parte {i+1}).",
                "output": bloco
            })

    return pares


def coletar_arquivos(caminho):
    """Coleta todos os arquivos suportados de um caminho."""
    extensoes = [".txt", ".md", ".py", ".json", ".jsonl", ".csv", ".log"]
    arquivos = []

    if os.path.isfile(caminho):
        arquivos.append(caminho)
    elif os.path.isdir(caminho):
        for ext in extensoes:
            arquivos.extend(glob.glob(os.path.join(caminho, f"**/*{ext}"), recursive=True))
    else:
        print(f"❌ Caminho não encontrado: {caminho}")
        sys.exit(1)

    # Filtrar arquivos muito pequenos
    arquivos = [a for a in arquivos if os.path.getsize(a) > 100]
    return sorted(arquivos)


def processar_arquivo(path):
    """Converte um arquivo individual em pares Q&A."""
    nome = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            conteudo = f.read()
    except Exception as e:
        print(f"   ⚠️ Erro lendo {nome}: {e}")
        return []

    if ext == ".jsonl":
        # Já é JSONL — verificar se tem instruction/output
        pares = []
        for line in conteudo.split("\n"):
            try:
                data = json.loads(line.strip())
                if "instruction" in data or "output" in data or "text" in data:
                    pares.append(data)
            except:
                pass
        return pares

    elif ext in [".txt", ".md"]:
        return texto_para_qa(conteudo, nome)

    elif ext == ".py":
        # Código Python → Q&A sobre funções
        pares = []
        funcoes = re.findall(r'(def \w+\(.*?\):.*?)(?=\ndef |\nclass |\Z)', conteudo, re.DOTALL)
        for func in funcoes:
            if len(func) > 50:
                nome_func = re.match(r'def (\w+)', func)
                if nome_func:
                    pares.append({
                        "instruction": f"Mostre como implementar a função {nome_func.group(1)} em Python.",
                        "output": func[:1500]
                    })
        if not pares:
            pares = texto_para_qa(conteudo, f"código {nome}")
        return pares

    else:
        return texto_para_qa(conteudo, nome)


def adicionar_cerebro(nome_cerebro, caminho_entrada):
    """Pipeline completo: coleta → codebook → transpila → pronto!"""
    sigla = nome_cerebro[0].upper()
    base_dir = os.path.join(SCRIPT_DIR, "1_extracao_local")
    datasets_dir = os.path.join(base_dir, "datasets_hibridos")
    codebooks_dir = os.path.join(base_dir, "codebooks")
    colab_dir = os.path.join(SCRIPT_DIR, "2_treinamento_nuvem", "arquivos_para_colab")

    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(codebooks_dir, exist_ok=True)
    os.makedirs(colab_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🧠 CROM-IA V4.1 — Adicionando Cérebro: {nome_cerebro}")
    print(f"   Fonte: {caminho_entrada}")
    print(f"{'='*60}")

    # Fase 1: Coletar e converter
    print(f"\n📂 [1/4] Coletando arquivos...")
    arquivos = coletar_arquivos(caminho_entrada)
    print(f"   Encontrados: {len(arquivos)} arquivos")

    raw_path = os.path.join(datasets_dir, f"dataset_{nome_cerebro}_raw.jsonl")
    total_pares = 0

    with open(raw_path, "w", encoding="utf-8") as fout:
        for arq in arquivos:
            pares = processar_arquivo(arq)
            for par in pares:
                fout.write(json.dumps(par, ensure_ascii=False) + "\n")
                total_pares += 1
            if pares:
                print(f"   ✅ {os.path.basename(arq)} → {len(pares)} pares")

    print(f"\n   Total de pares Q&A: {total_pares}")

    if total_pares < 10:
        print("❌ Poucos dados! Precisa de pelo menos 10 pares para treinar.")
        return

    # Fase 2: Gerar codebook inteligente
    print(f"\n🔬 [2/4] Gerando codebook data-driven...")
    codebook_path = os.path.join(codebooks_dir, f"codebook_{nome_cerebro}_v41.json")
    gerar_codebook(raw_path, codebook_path, sigla, max_tokens=150)

    # Fase 3: Transpilar com DNA 75%
    print(f"\n🧬 [3/4] Transpilando com mutação DNA 75%...")
    smart_path = os.path.join(datasets_dir, f"dataset_{nome_cerebro}_DNA75_smart.jsonl")
    codebook = carregar_codebook(codebook_path)
    transpilar_dataset_v41(raw_path, smart_path, codebook, 0.75)

    # Fase 4: Copiar para pasta do Colab
    print(f"\n📦 [4/4] Preparando para Colab...")
    colab_file = os.path.join(colab_dir, f"dataset_{nome_cerebro}_DNA75_smart.jsonl")
    import shutil
    shutil.copy2(smart_path, colab_file)

    print(f"\n{'='*60}")
    print(f"✅ CÉREBRO '{nome_cerebro}' PRONTO!")
    print(f"   Dataset: {colab_file}")
    print(f"   Pares: {total_pares} | Taxa DNA: 75%")
    print(f"   Codebook: {codebook_path}")
    print(f"\n   PRÓXIMO PASSO:")
    print(f"   1. Faça upload de '{os.path.basename(colab_file)}' no Colab")
    print(f"   2. Adicione ao array de cérebros no código de treino:")
    print(f'   cerebros.append(("{nome_cerebro}", "{os.path.basename(colab_file)}"))')
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("=" * 60)
        print("🧠 CROM-IA V4.1 — Adicionar Cérebro Rápido")
        print("=" * 60)
        print("\nUso:")
        print("  python3 adicionar_cerebro.py <Nome_Cerebro> <caminho>")
        print("\nExemplos:")
        print("  python3 adicionar_cerebro.py Direito /pasta/com/leis/")
        print("  python3 adicionar_cerebro.py Receitas /home/j/receitas.txt")
        print("  python3 adicionar_cerebro.py Financas dados_financeiros.jsonl")
        sys.exit(0)

    nome = sys.argv[1]
    caminho = sys.argv[2]
    adicionar_cerebro(nome, caminho)
