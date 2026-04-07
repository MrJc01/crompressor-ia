#!/bin/bash
# ==============================================================================
# ⌬  CROM-IA v4.3: MATRIZ DE BENCHMARK SRE
# ==============================================================================
# Este script mede a performance real do motor nativo v4.3 (llama-cli).
# ==============================================================================

DIR_BASE="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
LLAMA_CLI="$DIR_BASE/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELO_VFS="$DIR_BASE/models/CROM-IA_v4.3_Qwen3.5-2B.gguf"
LOG_BENCH="$DIR_BASE/v4.3_cognitive_leap/BENCHMARK_RESULTS.log"

# Cores
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🧬 CROM-IA v4.3: BENCHMARK DE PERFORMANCE             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

echo -e "Data: $(date)" | tee "$LOG_BENCH"
echo -e "Hardware: i5-3320M (Dual-Core + HT)" | tee -a "$LOG_BENCH"
echo -e "Modelo: Qwen 3.5 2B (GGUF via VFS)" | tee -a "$LOG_BENCH"
echo "--------------------------------------------------" | tee -a "$LOG_BENCH"

# Função para Rodar Teste
rodar_teste() {
    local threads=$1
    local batch=$2
    local label=$3
    
    echo -e "${CYAN}[TESTE] $label (Threads: $threads, Batch: $batch)...${NC}" | tee -a "$LOG_BENCH"
    
    # Rodar llama-cli capturando tudo para um temp primeiro
    "$LLAMA_CLI" \
        -m "$MODELO_VFS" \
        -p "Q: Explique o conceito de gravidade em 50 palavras.\nA:" \
        -n 64 \
        --threads "$threads" \
        -b "$batch" \
        --mlock \
        --no-display-prompt \
        > /tmp/bench.tmp 2>&1
    
    # Extrair métricas
    grep -iE "llama_print_timings|prompt eval|eval time" /tmp/bench.tmp | tee -a "$LOG_BENCH"
    
    # Extração robusta de t/s
    local prompt_tp=$(grep "prompt eval time" /tmp/bench.tmp | grep -o "[0-9.]* t/s" | head -n 1)
    local gen_tp=$(grep "eval time =" /tmp/bench.tmp | grep -o "[0-9.]* t/s" | head -n 1)
    
    echo -e "   -> Prompt: ${GREEN}${prompt_tp:-N/A}${NC} | Geração: ${GREEN}${gen_tp:-N/A}${NC}" | tee -a "$LOG_BENCH"
    
    echo "--------------------------------------------------" | tee -a "$LOG_BENCH"
}

# Cenário 1: "v4.2 Standard" (Threads 4, Batch 256)
rodar_teste 4 256 "SOTA v4.3 Native"

# Cenário 2: "Physical Cores only" (Threads 2, Batch 128)
rodar_teste 2 128 "Physical Cores Focus"

echo -e "${GREEN}✅ Benchmark Concluído. Resultados em: $LOG_BENCH${NC}"
