#!/usr/bin/env bash
# Telemetria SRE: Benchmark FUSE mmap() vs Leitura Crua NVMe
set -e

DIR_BASE="/home/j/Área de trabalho/crompressor-ia"
BIN_LLAMA="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"
MODEL_RAW="$DIR_BASE/models/qwen2.5-crom-dna.gguf"
MODEL_FUSE="$DIR_BASE/mnt_crom/qwen2.5-crom-dna.gguf"

if [ ! -f "$BIN_LLAMA" ]; then
    echo "[ERRO] Motor nativo (llama-cli) não encontrado no diretorio."
    exit 1
fi

echo "=========================================="
echo " 🏎️  BENCHMARK CROM: FUSE vs DISK RAW"
echo "=========================================="

test_inference() {
    local model_path=$1
    local mode=$2
    echo "------------------------------------------------"
    echo " Iniciando Teste ($mode)"
    echo " Arquivo alvo: $model_path"
    echo " Coletando telemetria psutil (RSS & Latência)..."
    
    # Rodar e medir tempo
    /usr/bin/time -v "$BIN_LLAMA" -m "$model_path" -p "Cálculo Físico: Qual a velocidade da luz em km/s?" --threads 2 -c 512 -n 32 --log-disable > /tmp/crom_bench_${mode}.log 2>&1
    
    # Extrair metricas do log de time -v
    local max_rss=$(grep "Maximum resident set size" /tmp/crom_bench_${mode}.log | awk '{print $6}')
    local page_faults=$(grep "Major (requiring I/O) page faults" /tmp/crom_bench_${mode}.log | awk '{print $6}')
    local elapsed=$(grep "Elapsed (wall clock) time" /tmp/crom_bench_${mode}.log | awk '{print $8}')

    echo " Resultado SRE ($mode):"
    echo "   ⏰ Tempo de Parede : $elapsed"
    echo "   🚀 Maximum RSS     : $((max_rss / 1024)) MB"
    echo "   ⚠️ Major PageFaults: $page_faults"
}

# 1. Teste FUSE
if [ ! -f "$MODEL_FUSE" ]; then
    echo "[AVISO] Modelo não montado no FUSE. Rode scripts/montar_fuse_modelo.sh primeiro."
    echo "Abortando Benchmark FUSE..."
else
    test_inference "$MODEL_FUSE" "FUSE_MMAP_ZERO_COPY"
fi

# 2. Teste RAW
test_inference "$MODEL_RAW" "DISK_RAW_SSD"

echo "------------------------------------------------"
echo " Conclusão: O Modo FUSE deve mostrar RSS idêntico ou levemente inferior,"
echo " sem degradação do tempo de inferência (Tempo Constante HNSW O(1))."
