#!/bin/bash
# CROM-IA RAG Inferência via Terminal (One-Shot Q&A)

MODELO="/home/j/Área de trabalho/crompressor-ia/models/qwen2.5-crom-dna.gguf"
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "[ERRO] Motor nativo não encontrado!"
    exit 1
fi

echo "=================================================="
echo " 🧠 CROM-IA: MODO RAG (Retrieval-Augmented Generation)"
echo "=================================================="
echo " Digite sua pergunta sobre a documentação/projeto."
echo " Digite 'sair' para encerrar."
echo "=================================================="

while true; do
    echo -e "\n> Pergunta: \c"
    read -r PERGUNTA

    if [[ "$PERGUNTA" == "sair" || "$PERGUNTA" == "exit" || -z "$PERGUNTA" ]]; then
        echo "Saindo do modo RAG."
        break
    fi

    # 1. Buscando contexto no cromdb_rag_index (BM25 O(1))
    echo " 🔍 [RAG] Minerando super-tokens BM25..."
    PROMPT_COM_CONTEXTO=$(source pesquisa/.venv/bin/activate && python3 rag/consultar_rag.py "$PERGUNTA")

    echo " ⚡ [RAG] Injetando contexto no Llama e gerando resposta..."
    echo "--------------------------------------------------"

    # 2. Invocando llama-cli passando o promt enriquecido pelo Python no arg -p
    "$BINARIO_NATIVO" \
        -m "$MODELO" \
        --threads 2 \
        -c 1024 \
        -n 256 \
        --temp 0.1 \
        --repeat_penalty 1.15 \
        --log-disable \
        -p "$PROMPT_COM_CONTEXTO"
        
    echo ""
    echo "--------------------------------------------------"
done
