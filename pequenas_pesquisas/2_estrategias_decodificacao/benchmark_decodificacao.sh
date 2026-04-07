#!/bin/bash

# Módulo 2: Benchmark de Decodificação (Pequenas Pesquisas)
# Objetivo: Testar a estabilidade do modelo 0.6B sob diferentes temperaturas.

BINARY="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODEL="/home/j/Área de trabalho/crompressor-ia/v4.2_multibrain_engine/3_inferencia_local/micro_cerebros/qwen3-0.6b.Q4_K_M.gguf"
OUTPUT_DIR="/home/j/Área de trabalho/crompressor-ia/pequenas_pesquisas/2_estrategias_decodificacao/logs"

mkdir -p "$OUTPUT_DIR"

PROMPT="Escreva um haicai curto sobre inteligência artificial em português."

echo "--- Iniciando Benchmark de Temperatura ---"

# Use quotes around variables to handle spaces in paths
for TEMP in 0.1 0.4 0.7 1.0; do
    echo "Testando Temp: $TEMP..."
    # Corrected invocation with quotes
    "$BINARY" -m "$MODEL" -p "$PROMPT" --temp "$TEMP" -n 64 > "$OUTPUT_DIR/temp_$TEMP.txt" 2>&1
    echo "Concluído. Log salvo em logs/temp_$TEMP.txt"
done

echo "--- Benchmark Finalizado ---"
