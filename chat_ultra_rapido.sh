#!/bin/bash

# Este script utiliza o motor puramente nativo C++ (Compilado para AVX no seu Humble PC)
# Ele anula completamente o Python e o pip das jogadas, indo direto ao Kernel.

MODELO="/home/j/Área de trabalho/crompressor-ia/models/qwen2.5-crom-dna.gguf"
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "[ERRO] Motor nativo não encontrado!"
    exit 1
fi

echo "=================================================="
echo " ⚡ CROM-IA TERMINAL (Motor Sub-Simbólico DNA)"
echo "=================================================="
echo " 🧬 ATENÇÃO: As métricas de (t/s) geradas a seguir"
echo " representam a taxa de Transpilação de Base-4 (DNA Blocks/s)."
echo " Esse modo contorna a RAM engolida, puxando pesos via FUSE."
echo "=================================================="

USE_FUSE=false
for arg in "$@"; do
    if [ "$arg" == "--fuse" ]; then
        USE_FUSE=true
    fi
done

if [ "$USE_FUSE" = true ]; then
    echo "[!] MODO FUSE ATIVADO: Montando arquivo sob demanda virtual"
    bash "/home/j/Área de trabalho/crompressor-ia/scripts/montar_fuse_modelo.sh"
    MODELO="/home/j/Área de trabalho/crompressor-ia/mnt_crom/qwen2.5-crom-dna.gguf"
fi

echo "=================================================="

# Parâmetros agressivos para travar o contexto do TinyLlama e dar velocidade
# -c 1024 : Contexto reduzido na RAM para evaluação instantânea
# -n 256  : Gera até 256 tokens per block
# -t 4    : Usa seus 4 threads principais
# --temp 0.2 --repeat_penalty 1.18 : Evita loops de repetição

"$BINARIO_NATIVO" \
    -m "$MODELO" \
    --threads 4 \
    -c 512 \
    -b 256 \
    -n 512 \
    --temp 0.1 \
    --repeat_penalty 1.15 \
    -cnv \
    -p "You are CROM-IA, an AI assistant neural node. Respond concisely to the user in Portuguese. Do not hallucinate or loop spaces. Keep your answers straight to the point."

echo ""
echo "[SISTEMA] Motor C++ Desativado."
