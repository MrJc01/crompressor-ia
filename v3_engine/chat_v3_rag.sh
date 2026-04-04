#!/bin/bash
# ==============================================================================
# 🧬 CROM-IA V3: Shell Interceptador O(1) de Latência
# Liga o Llama.cpp ao Python Injector desabilitando buffers TTY.
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELO="/home/j/Área de trabalho/crompressor-ia/models/Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf"
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$MODELO" ]; then
    echo "❌ Erro: Modelo GGUF da Fase 3/Colab não encontrado em $MODELO"
    exit 1
fi

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "❌ Erro: Binário base do LLaMA CLI não encontrado em $BINARIO_NATIVO"
    exit 1
fi

echo "============================================================"
echo " 🟢 STARTING CROM-RAG ENGINE (V3) - INJECTION PIPELINE"
echo "============================================================"
echo ""
echo "Digite seu prompt normal. O script chamará o Qwen V3 que"
echo "foi refinado via LoRA e devolverá Pointers @@DNA. A UI"
echo "vai expandir com as macros da sua base de dados!"
echo ""
read -p "Prompt > " USER_PROMPT

# Parâmetros de contexto otimizados para Causal Instruction V3
# Esvaziamos todos os Buffers do System (stdbuf -o0) para que 
# o pipe envie char por char para o python imediatamente.
stdbuf -o0 -i0 "$BINARIO_NATIVO" \
    -m "$MODELO" \
    --threads 4 \
    -c 2048 \
    -n 1024 \
    --temp 0.1 \
    --repeat_penalty 1.15 \
    --repeat_last_n 64 \
    --color false \
    --prompt "Abaixo está uma instrução CROM-IA.

### Instruction:
$USER_PROMPT

### Input:


### Response:
" 2>/dev/null | python3 -u "$DIR/rag_injector_native.py"

echo ""
echo "============================================================"
echo " ⏹️ FIM DA TRANSMISSÃO."
echo "============================================================"
