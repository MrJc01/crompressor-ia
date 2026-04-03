import os
import json
import re
import math
from collections import defaultdict, Counter

DIR_DOCS = "/home/j/Área de trabalho/crompressor-ia/docs"
ARQUIVO_INDEX = "/home/j/Área de trabalho/crompressor-ia/rag/cromdb_rag_index.json"
CHUNK_SIZE = 500
OVERLAP = 100

def tokenizar(texto):
    """Tokenização simples: minúsculas e apenas alfanuméricos."""
    return re.findall(r'\b\w+\b', texto.lower())

def ler_documentos():
    textos = []
    # Inclui o README principal
    readme = "/home/j/Área de trabalho/crompressor-ia/README.md"
    if os.path.exists(readme):
        with open(readme, 'r', encoding='utf-8') as f:
            textos.append(("README.md", f.read()))
            
    # Inclui docs/
    if os.path.exists(DIR_DOCS):
        for raiz, _, arquivos in os.walk(DIR_DOCS):
            for arq in arquivos:
                if arq.endswith(".md") or arq.endswith(".txt"):
                    caminho = os.path.join(raiz, arq)
                    with open(caminho, 'r', encoding='utf-8') as f:
                        textos.append((arq, f.read()))
    return textos

def gerar_chunks(textos):
    chunks = []
    chunk_id = 0
    for titulo, conteudo in textos:
        """Divide o documento em pedaços (Chunks) para BM25/TF-IDF"""
        pos = 0
        while pos < len(conteudo):
            pedaço = conteudo[pos:pos+CHUNK_SIZE]
            chunks.append({
                "id": chunk_id,
                "fonte": titulo,
                "texto": pedaço
            })
            chunk_id += 1
            pos += (CHUNK_SIZE - OVERLAP)
    return chunks

def indexar_bm25(chunks):
    print(f"Indexando {len(chunks)} fragmentos de conhecimento...")
    
    # Calcular N (total de documentos/chunks)
    N = len(chunks)
    
    # Document frequency: quantos chunks contém o termo X
    df = Counter()
    
    # Frequência de termo em cada chunk
    tf = []
    lista_comprimentos = []
    
    for c in chunks:
        tokens = tokenizar(c["texto"])
        freq_termo = Counter(tokens)
        tf.append(freq_termo)
        for termo in freq_termo.keys():
            df[termo] += 1
        lista_comprimentos.append(len(tokens))
        
    avgdl = sum(lista_comprimentos) / N if N > 0 else 1
    
    # Salvar estrutura
    index_db = {
        "N": N,
        "avgdl": avgdl,
        "df": dict(df),
        "chunks": chunks,
        "tf": [dict(t) for t in tf],
        "comprimentos": lista_comprimentos
    }
    
    with open(ARQUIVO_INDEX, 'w', encoding='utf-8') as f:
        json.dump(index_db, f, ensure_ascii=False)
        
    print(f"✅ Indexação concluída e salva em {ARQUIVO_INDEX} (Motor BM25 Nativo)")

if __name__ == "__main__":
    docs = ler_documentos()
    if not docs:
        print("Nenhum documento encontrado.")
    else:
        ck = gerar_chunks(docs)
        indexar_bm25(ck)
