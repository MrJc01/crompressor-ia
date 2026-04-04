#!/usr/bin/env bash
# ==============================================================================
# CROM-IA V4.1: Inferência Multi-Cérebros (Qwen3-0.6B + DNA Reforçado)
# Cada GGUF é um modelo completo com LoRA DNA fundido internamente.
# ==============================================================================

LLAMA_CLI="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELS_DIR="/home/j/Área de trabalho/crompressor-ia/models"

# Filtrar apenas modelos V4.1 (379MB, Qwen3-0.6B)
V41_MODELS=()
V41_NAMES=()
for f in "$MODELS_DIR"/*.gguf; do
    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
    # V4.1 = ~379MB (397934592 bytes), V4.0 = ~941MB
    if [ "$size" -lt 500000000 ]; then
        V41_MODELS+=("$f")
        V41_NAMES+=("$(basename "$f" .gguf)")
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     🧠 CROM-IA V4.1 — Multi-Brain Engine (Qwen3)       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  DNA Reforçado: 75% | Rank: 64 | Steps: 2000            ║"
echo "║                                                          ║"

i=1
for nome in "${V41_NAMES[@]}"; do
    echo "║  [$i] $nome"
    i=$((i + 1))
done

echo "║                                                          ║"
echo "║  [0] Sair                                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
read -p "🎯 Escolha (número): " escolha

if [ "$escolha" = "0" ]; then
    echo "👋 Até logo!"
    exit 0
fi

idx=$((escolha - 1))
if [ -z "${V41_MODELS[$idx]}" ]; then
    echo "❌ Opção inválida!"
    exit 1
fi

MODELO="${V41_MODELS[$idx]}"
NOME="${V41_NAMES[$idx]}"

echo ""
echo "🚀 Ativando Cérebro V4.1: $NOME"
echo "📦 Arquivo: $MODELO ($(du -h "$MODELO" | cut -f1))"
echo "⏳ Carregando via mmap..."
echo ""

"$LLAMA_CLI" \
    -m "$MODELO" \
    -c 1024 \
    -n 512 \
    --threads 4 \
    --batch-size 512 \
    --ubatch-size 256 \
    --mlock \
    --temp 0.7 \
    --repeat-penalty 1.15 \
    --conversation \
    --prompt "Você é o CROM-IA, um assistente inteligente brasileiro com Micro-Cérebro $NOME ativo."
