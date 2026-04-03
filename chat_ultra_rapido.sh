#!/bin/bash

# Este script utiliza o motor puramente nativo C++ (Compilado para AVX no seu Humble PC)
# Ele anula completamente o Python e o pip das jogadas, indo direto ao Kernel.

MODELO="/home/j/Área de trabalho/crompressor-ia/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "[ERRO] Motor nativo não encontrado!"
    exit 1
fi

echo "=================================================="
echo " ⚡ CROM-IA TERMINAL (Motor C++ Nativo AVX)"
echo "=================================================="
echo " Esse modo opera 10x mais rápido que o Python"
echo " (Carregando Pesos Dinamicamente...)"
echo "=================================================="

# Parâmetros agressivos para travar o contexto do TinyLlama e dar velocidade
# -c 1024 : Contexto reduzido na RAM para evaluação instantânea
# -n 256  : Gera até 256 tokens per block
# -t 4    : Usa seus 4 threads principais
# --temp 0.2 --repeat_penalty 1.18 : Evita loops de repetição

"$BINARIO_NATIVO" \
    -m "$MODELO" \
    --threads 2 \
    -c 1024 \
    -n 256 \
    --temp 0.2 \
    --repeat_penalty 1.18 \
    -cnv \
    -p "You are CROM-IA, an AI assistant neural node. Respond concisely to the user in Portuguese. Do not hallucinate or loop spaces. Keep your answers straight to the point."

echo ""
echo "[SISTEMA] Motor C++ Desativado."
