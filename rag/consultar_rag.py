import json
import re
import math
import sys

ARQUIVO_INDEX = "/home/j/Área de trabalho/crompressor-ia/rag/cromdb_rag_index.json"

# Hiperparâmetros do BM25
k1 = 1.5
b = 0.75

def tokenizar(texto):
    return re.findall(r'\b\w+\b', texto.lower())

def carregar_index():
    try:
        with open(ARQUIVO_INDEX, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        print("[ERRO] Arquivo de index não encontrado. Execute o indexar_conhecimento.py primeiro.", file=sys.stderr)
        sys.exit(1)

def buscar_bm25(query, index_db, top_k=3):
    N = index_db["N"]
    avgdl = index_db["avgdl"]
    df = index_db["df"]
    chunks = index_db["chunks"]
    tfs = index_db["tf"]
    comprimentos = index_db["comprimentos"]
    
    tokens_query = tokenizar(query)
    scores = [0.0] * N
    
    for termo in tokens_query:
        if termo not in df:
            continue
        
        # IDF formula padrão do BM25
        idf = math.log(1 + (N - df[termo] + 0.5) / (df[termo] + 0.5))
        
        for i in range(N):
            freq = tfs[i].get(termo, 0)
            if freq == 0:
                continue
                
            dl = comprimentos[i]
            # Conta BM25
            numerador = freq * (k1 + 1)
            denominador = freq + k1 * (1 - b + b * (dl / avgdl))
            scores[i] += idf * (numerador / denominador)
            
    # Rankear
    ranked_indices = sorted(range(N), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked_indices[:top_k] if scores[i] > 0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 consultar_rag.py 'sua pergunta'")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    db = carregar_index()
    resultados = buscar_bm25(query, db, top_k=2)
    
    # O output stdout puramente montará o prompt, para ser lido pelo bash/frontend
    
    if not resultados:
        print(f"Zero informações contextuais encontradas para a query: {query}")
        sys.exit(0)
        
    prompt_context = "CONTEXTO DE CONHECIMENTO RETORNADO DO CROMDB:\n\n"
    for r in resultados:
        prompt_context += f"--- DA FONTE ({r['fonte']}) ---\n"
        prompt_context += rf"{r['texto']}" + "\n\n"
        
    prompt_context += f"Com base no contexto acima, responda à pergunta do usuário: {query}"
    
    print(prompt_context)
