#!/bin/bash

# Módulo 6: Teste de Gramática Estrutural (GBNF)
# Objetivo: Forçar o modelo a seguir uma estrutura de 3 linhas.

BINARY="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODEL="/home/j/Área de trabalho/crompressor-ia/v4.2_multibrain_engine/3_inferencia_local/micro_cerebros/qwen3-0.6b.Q4_K_M.gguf"
GRAMMAR="/home/j/Área de trabalho/crompressor-ia/pequenas_pesquisas/6_gramatica_estrutural_gbnf/haicai.gbnf"

PROMPT="Escreva um haicai sobre o processamento de dados."

echo "--- Testando COM Gramática (GBNF) ---"
"$BINARY" -m "$MODEL" -p "$PROMPT" --grammar-file "$GRAMMAR" -n 128 --temp 0.1

echo -e "\n\n--- Teste Finalizado ---"
