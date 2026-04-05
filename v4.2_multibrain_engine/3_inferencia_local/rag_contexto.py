#!/usr/bin/env python3
"""
CROM-IA V4.2 — Motor RAG-Lite (sem GPU)
Lê arquivos/pastas, chunka, indexa por keywords, injeta contexto no prompt.
Projetado para rodar no i5-3320M sem embeddings.
"""

import os
import sys
import re
import json
from collections import Counter
import math

# Extensões suportadas e seus tipos
EXTENSOES_SUPORTADAS = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
    '.sh': 'shell', '.bash': 'shell',
    '.md': 'markdown', '.txt': 'text', '.rst': 'text',
    '.json': 'json', '.jsonl': 'jsonl',
    '.html': 'html', '.htm': 'html',
    '.css': 'css', '.scss': 'css',
    '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
    '.cfg': 'config', '.ini': 'config', '.env': 'config',
    '.log': 'log',
    '.xml': 'xml', '.csv': 'csv',
    '.java': 'java', '.c': 'c', '.cpp': 'cpp', '.h': 'c',
    '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
    '.sql': 'sql', '.r': 'r', '.R': 'r',
}

MAX_CHARS_POR_ARQUIVO = 3000
MAX_CONTEXTO_TOTAL = 6000  # ~1500 tokens
MAX_LINHAS_LOG = 50
MAX_LINHAS_JSONL = 20


def ler_arquivo(caminho):
    """Lê um arquivo respeitando limites por tipo."""
    ext = os.path.splitext(caminho)[1].lower()
    tipo = EXTENSOES_SUPORTADAS.get(ext, 'text')

    try:
        with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
            if tipo == 'log':
                linhas = f.readlines()
                conteudo = ''.join(linhas[-MAX_LINHAS_LOG:])
            elif tipo == 'jsonl':
                linhas = []
                for i, line in enumerate(f):
                    if i >= MAX_LINHAS_JSONL:
                        break
                    linhas.append(line)
                conteudo = ''.join(linhas)
            elif tipo == 'json':
                conteudo = f.read(MAX_CHARS_POR_ARQUIVO)
            elif tipo == 'html':
                raw = f.read(MAX_CHARS_POR_ARQUIVO * 2)
                conteudo = re.sub(r'<[^>]+>', '', raw)[:MAX_CHARS_POR_ARQUIVO]
            else:
                conteudo = f.read(MAX_CHARS_POR_ARQUIVO)
    except Exception as e:
        return None, f"Erro ao ler {caminho}: {e}"

    if len(conteudo) > MAX_CHARS_POR_ARQUIVO:
        conteudo = conteudo[:MAX_CHARS_POR_ARQUIVO] + "\n... [truncado]"

    num_linhas = conteudo.count('\n') + 1
    return {
        'nome': os.path.basename(caminho),
        'caminho': caminho,
        'tipo': tipo,
        'linhas': num_linhas,
        'conteudo': conteudo,
        'tamanho': len(conteudo),
    }, None


def listar_arquivos(caminhos_arquivos=None, caminhos_pastas=None):
    """Lista todos os arquivos a serem processados."""
    arquivos = []

    if caminhos_arquivos:
        for arq in caminhos_arquivos:
            if os.path.isfile(arq):
                ext = os.path.splitext(arq)[1].lower()
                if ext in EXTENSOES_SUPORTADAS:
                    arquivos.append(arq)
                else:
                    print(f"⚠️  Extensão não suportada: {arq}", file=sys.stderr)
            else:
                print(f"⚠️  Arquivo não encontrado: {arq}", file=sys.stderr)

    if caminhos_pastas:
        for pasta in caminhos_pastas:
            if os.path.isdir(pasta):
                for raiz, dirs, files in os.walk(pasta):
                    # Ignorar diretórios ocultos e comuns
                    dirs[:] = [d for d in dirs if not d.startswith('.')
                               and d not in ('node_modules', '__pycache__',
                                             'venv', '.git', 'dist', 'build')]
                    for nome in sorted(files):
                        ext = os.path.splitext(nome)[1].lower()
                        if ext in EXTENSOES_SUPORTADAS:
                            arquivos.append(os.path.join(raiz, nome))
            else:
                print(f"⚠️  Pasta não encontrada: {pasta}", file=sys.stderr)

    return arquivos


def chunkar(texto, tamanho_chunk=500):
    """Divide texto em chunks preservando limites lógicos."""
    chunks = []
    linhas = texto.split('\n')
    chunk_atual = []
    chars_atual = 0

    for linha in linhas:
        # Se adicionar esta linha excede o limite, fecha o chunk
        if chars_atual + len(linha) + 1 > tamanho_chunk and chunk_atual:
            chunks.append('\n'.join(chunk_atual))
            chunk_atual = []
            chars_atual = 0

        chunk_atual.append(linha)
        chars_atual += len(linha) + 1

    if chunk_atual:
        chunks.append('\n'.join(chunk_atual))

    return chunks


def extrair_keywords(texto):
    """Extrai keywords com peso por frequência (TF simplificado)."""
    # Palavras com 3+ chars, lowercase
    palavras = re.findall(r'\b\w{3,}\b', texto.lower())

    # Stopwords PT-BR + EN comuns
    stopwords = {
        'que', 'para', 'com', 'uma', 'por', 'não', 'mais', 'como', 'dos',
        'das', 'nos', 'nas', 'são', 'tem', 'seu', 'sua', 'isso', 'esta',
        'esse', 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
        'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have',
        'from', 'this', 'that', 'with', 'they', 'been', 'will', 'each',
        'def', 'self', 'none', 'true', 'false', 'return', 'import', 'class',
    }

    palavras_filtradas = [p for p in palavras if p not in stopwords]
    return Counter(palavras_filtradas)


def buscar_chunks_relevantes(query, chunks_indexados, top_k=3):
    """Busca chunks mais relevantes para a query usando keyword matching."""
    query_keywords = extrair_keywords(query)
    if not query_keywords:
        # Sem keywords úteis, retorna os primeiros chunks
        return chunks_indexados[:top_k]

    scores = []
    for i, (chunk, keywords) in enumerate(chunks_indexados):
        # Score = soma das frequências de keywords em comum
        score = sum(
            query_keywords[kw] * keywords[kw]
            for kw in query_keywords
            if kw in keywords
        )
        scores.append((score, i, chunk))

    scores.sort(reverse=True)
    return [chunk for _, _, chunk in scores[:top_k]]


def processar_para_contexto(caminhos_arquivos=None, caminhos_pastas=None):
    """
    Pipeline completo: ler → chunkar → indexar → formatar contexto.
    Retorna string pronta para injetar no system prompt.
    """
    todos_arquivos = listar_arquivos(caminhos_arquivos, caminhos_pastas)

    if not todos_arquivos:
        return "", []

    print(f"📂 Processando {len(todos_arquivos)} arquivo(s)...", file=sys.stderr)

    # Ler todos os arquivos
    docs = []
    for caminho in todos_arquivos:
        doc, erro = ler_arquivo(caminho)
        if doc:
            docs.append(doc)
            print(f"   ✅ {doc['nome']} ({doc['tipo']}, {doc['linhas']} linhas)", file=sys.stderr)
        elif erro:
            print(f"   ❌ {erro}", file=sys.stderr)

    if not docs:
        return "", []

    # Montar contexto respeitando limite total
    contexto_partes = []
    chars_total = 0

    # Primeira passada: resumo estrutural (sempre inclui)
    resumo = "ESTRUTURA DOS ARQUIVOS:\n"
    for doc in docs:
        resumo += f"  📄 {doc['nome']} ({doc['tipo']}, {doc['linhas']} linhas)\n"
    contexto_partes.append(resumo)
    chars_total += len(resumo)

    # Segunda passada: conteúdo dos arquivos (respeitando limite)
    for doc in docs:
        espaco_restante = MAX_CONTEXTO_TOTAL - chars_total
        if espaco_restante <= 200:
            break

        # Header do arquivo
        header = f"\n{'─'*40}\n📄 {doc['nome']} ({doc['tipo']}):\n"

        # Conteúdo (truncar se necessário)
        conteudo = doc['conteudo']
        max_conteudo = min(len(conteudo), espaco_restante - len(header) - 50)
        if max_conteudo <= 0:
            break

        if max_conteudo < len(conteudo):
            conteudo = conteudo[:max_conteudo] + "\n... [truncado]"

        # Wrap em code block se for código
        if doc['tipo'] in ('python', 'javascript', 'typescript', 'shell',
                            'java', 'c', 'cpp', 'rust', 'go', 'ruby',
                            'php', 'sql', 'css', 'html', 'yaml', 'json'):
            bloco = f"{header}```{doc['tipo']}\n{conteudo}\n```\n"
        else:
            bloco = f"{header}{conteudo}\n"

        contexto_partes.append(bloco)
        chars_total += len(bloco)

    contexto_final = ''.join(contexto_partes)

    # Indexar chunks para busca futura (se implementarmos busca interativa)
    chunks_indexados = []
    for doc in docs:
        for chunk in chunkar(doc['conteudo']):
            keywords = extrair_keywords(chunk)
            chunks_indexados.append((chunk, keywords))

    print(f"\n📊 Contexto: {chars_total} chars ({chars_total//4} tokens est.)",
          file=sys.stderr)
    print(f"   Chunks indexados: {len(chunks_indexados)}", file=sys.stderr)

    return contexto_final, chunks_indexados


def formatar_system_prompt(contexto=""):
    """Formata o system prompt completo com contexto injetado."""
    base = "Você é CROM-IA, assistente brasileiro inteligente com compressão DNA ativa. Responda sempre em português."

    if contexto:
        return f"""{base}

CONTEXTO — Arquivos carregados para análise:
{contexto}

Use o contexto acima para responder perguntas. Se a pergunta não for sobre os arquivos, responda normalmente."""
    else:
        return base


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CROM-IA V4.2 — RAG Contexto")
    parser.add_argument('--arquivo', action='append', help='Arquivo para processar')
    parser.add_argument('--pasta', action='append', help='Pasta para processar')
    parser.add_argument('--query', help='Query para buscar chunks relevantes')
    parser.add_argument('--prompt-only', action='store_true',
                        help='Outputar apenas o system prompt (para uso no chat.sh)')

    args = parser.parse_args()

    contexto, chunks = processar_para_contexto(args.arquivo, args.pasta)

    if args.query and chunks:
        print("\n🔍 Chunks mais relevantes para:", args.query, file=sys.stderr)
        relevantes = buscar_chunks_relevantes(args.query, chunks)
        for i, chunk in enumerate(relevantes, 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk)
    elif args.prompt_only:
        # Output limpo do prompt para captura pelo bash
        print(formatar_system_prompt(contexto))
    else:
        print(formatar_system_prompt(contexto))
