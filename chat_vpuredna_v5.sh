#!/bin/bash
# ==============================================================================
# 🧬 vPureDna v5.1 — LAUNCHER (Chat DNA + Server HTTP)
# ==============================================================================
# Uso:
#   ./chat_vpuredna_v5.sh          ← Q4_K_M (1.1GB, recomendado)
#   ./chat_vpuredna_v5.sh --f16    ← Full precision (3.3GB)
#   ./chat_vpuredna_v5.sh --raw    ← Sem compressão DNA
# ==============================================================================

DIR_BASE="$(dirname "$(realpath "$0")")"
INFERENCE_PY="$DIR_BASE/vPureDna/06_inference/chat_vpuredna_v5.py"

# Cores
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}   🧬 ${GREEN}vPureDna v5.1${NC} — Inicializador de Elite                 ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

# 1. Verificar modelo
echo -e "\n${YELLOW}[1/4] Verificando modelo GGUF...${NC}"
Q4_MODEL="$DIR_BASE/models/vpuredna_v5/vpuredna_v5_Q4KM.gguf"
F16_MODEL="$DIR_BASE/models/vpuredna_v5/vpuredna_v5.gguf"

if [ -f "$Q4_MODEL" ]; then
    SIZE_Q4=$(du -h "$Q4_MODEL" | cut -f1)
    echo -e "   ${GREEN}✅${NC} Q4_K_M: $SIZE_Q4"
fi

if [ -f "$F16_MODEL" ]; then
    SIZE_F16=$(du -h "$F16_MODEL" | cut -f1)
    echo -e "   ${GREEN}✅${NC} F16:    $SIZE_F16"
fi

if [ ! -f "$Q4_MODEL" ] && [ ! -f "$F16_MODEL" ]; then
    echo -e "   ${RED}❌ Nenhum modelo encontrado!${NC}"
    echo -e "   Coloque vpuredna_v5_Q4KM.gguf em models/vpuredna_v5/"
    exit 1
fi

# 2. Verificar llama-server
echo -e "${YELLOW}[2/4] Verificando engine llama-cli...${NC}"
LLAMA_CLI="$DIR_BASE/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
if [ -f "$LLAMA_CLI" ]; then
    echo -e "   ${GREEN}✅${NC} llama-cli encontrado"
else
    echo -e "   ${RED}❌ llama-cli não encontrado!${NC}"
    echo -e "   Compile: cd pesquisa/poc_llama_cpp_fuse/llama.cpp && cmake -B build && cmake --build build -j"
    exit 1
fi

# 3. Verificar Python e codebook
echo -e "${YELLOW}[3/4] Verificando DNA Compressor...${NC}"
CODEBOOK="$DIR_BASE/codebooks/codebook_1x5_dinamico_expandido.json"
if [ -f "$CODEBOOK" ]; then
    ENTRIES=$(python3 -c "import json; d=json.load(open('$CODEBOOK')); print(len(d.get('entries',{})))" 2>/dev/null)
    echo -e "   ${GREEN}✅${NC} Codebook: $ENTRIES entries"
else
    echo -e "   ${RED}❌ Codebook não encontrado!${NC}"
    exit 1
fi

# 4. Limpar processos anteriores
echo -e "${YELLOW}[4/4] Preparando ambiente...${NC}"
pkill -f "llama-server.*vpuredna" 2>/dev/null
sleep 0.3
echo -e "   ${GREEN}✅${NC} Ambiente limpo"

# Detectar threads
THREADS=$(nproc 2>/dev/null || echo 2)
if [ "$THREADS" -gt 4 ]; then THREADS=4; fi

# Parsear argumentos
MODEL_FLAG="Q4"
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --f16)    MODEL_FLAG="F16" ;;
        --raw)    EXTRA_ARGS="$EXTRA_ARGS --raw" ;;
    esac
done

echo -e "\n${GREEN}🚀 LANÇANDO vPureDna v5.1 (model=$MODEL_FLAG, threads=$THREADS)...${NC}\n"
sleep 0.3

# Cleanup ao sair
cleanup() {
    echo -e "\n${YELLOW}Limpando processos...${NC}"
    pkill -f "llama-cli.*vpuredna" 2>/dev/null
}
trap cleanup EXIT

python3 "$INFERENCE_PY" \
    --model "$MODEL_FLAG" \
    --threads "$THREADS" \
    --ctx 2048 \
    --temp 0.3 \
    $EXTRA_ARGS

