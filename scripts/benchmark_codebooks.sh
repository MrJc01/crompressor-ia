#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  🧬 CROM-IA V2: Benchmark Comparativo de Codebooks DNA     ║
# ║                                                              ║
# ║  Executa todas as combinações (4 taxas × 2 modos)           ║
# ║  e gera relatório comparativo                               ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$BASE_DIR/models"
CODEBOOKS_DIR="$BASE_DIR/codebooks"
SCRIPTS_DIR="$BASE_DIR/scripts"
LLAMA_CLI="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"
DECODER="$SCRIPTS_DIR/dna_decoder.py"

# Perguntas de benchmark (10 simples, 10 médias, 10 complexas)
PERGUNTAS_SIMPLES=(
    "O que é o sol?"
    "Qual é a capital do Brasil?"
    "Quanto é 2+2?"
    "O que é água?"
    "Qual é a cor do céu?"
    "Quantos dias tem uma semana?"
    "O que é um gato?"
    "Qual é o maior planeta?"
    "O que é uma árvore?"
    "Quem inventou o telefone?"
)

PERGUNTAS_MEDIAS=(
    "Explique gravidade em 3 frases."
    "Como funciona a internet?"
    "O que é fotossíntese?"
    "Explique o que é DNA."
    "Como funciona um motor a combustão?"
    "O que causa terremotos?"
    "Explique o ciclo da água."
    "O que é a teoria da evolução?"
    "Como funciona uma vacina?"
    "O que é inteligência artificial?"
)

PERGUNTAS_COMPLEXAS=(
    "Compare democracia e autocracia."
    "Explique a teoria das cordas."
    "Qual a relação entre entropia e informação?"
    "Explique o problema da consciência na filosofia."
    "Compare capitalismo e socialismo."
    "O que é computação quântica?"
    "Explique a teoria da relatividade."
    "Compare energia nuclear e solar."
    "Explique o paradoxo de Fermi."
    "O que é a singularidade tecnológica?"
)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     CROM-IA V2: BENCHMARK COMPLETO DNA COMPRESSION     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Motor      : $LLAMA_CLI"
echo "║  Decoder    : $DECODER"
echo "║  Codebooks  : $CODEBOOKS_DIR"
echo "║  Models     : $MODELS_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Verificar requisitos
if [ ! -f "$LLAMA_CLI" ]; then
    echo "❌ llama-cli não encontrado: $LLAMA_CLI"
    exit 1
fi

TAXAS=("1x3" "1x5" "1x10" "1x20")
MODOS=("fixo" "dinamico")

RESULTS_FILE="$BASE_DIR/benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo "╔══════════════════════════════════════════════════════════╗" | tee "$RESULTS_FILE"
echo "║          CROM-IA V2: BENCHMARK DNA COMPRESSION          ║" | tee -a "$RESULTS_FILE"
echo "╠══════════════════════════════════════════════════════════╣" | tee -a "$RESULTS_FILE"
echo "║  Taxa  │ Modo     │ Palavras/s │ Hit Rate │ Veredicto   ║" | tee -a "$RESULTS_FILE"
echo "║────────┼──────────┼────────────┼──────────┼─────────────║" | tee -a "$RESULTS_FILE"

for taxa in "${TAXAS[@]}"; do
    for modo in "${MODOS[@]}"; do
        MODEL="$MODELS_DIR/crom-dna-${taxa}-${modo}.gguf"
        CODEBOOK="$CODEBOOKS_DIR/codebook_${taxa}_${modo}.json"
        
        if [ ! -f "$MODEL" ]; then
            echo "║  ${taxa}  │ ${modo}    │     N/A    │   N/A    │ ⬜ Sem modelo║" | tee -a "$RESULTS_FILE"
            continue
        fi
        
        if [ ! -f "$CODEBOOK" ]; then
            echo "║  ${taxa}  │ ${modo}    │     N/A    │   N/A    │ ⬜ Sem codebook║" | tee -a "$RESULTS_FILE"
            continue
        fi
        
        echo "[>] Testando ${taxa} / ${modo}..."
        
        # Teste com 3 perguntas (1 de cada nível)
        TOTAL_PALAVRAS=0
        TOTAL_TEMPO=0
        TOTAL_HITS=0
        TOTAL_TOKENS=0
        
        for PERGUNTA in "${PERGUNTAS_SIMPLES[0]}" "${PERGUNTAS_MEDIAS[0]}" "${PERGUNTAS_COMPLEXAS[0]}"; do
            PROMPT="<|im_start|>system\nVocê é um compressor CROM DNA (taxa ${taxa/x/:}). Comprima a resposta usando códigos do codebook semântico DNA.<|im_end|>\n<|im_start|>user\n${PERGUNTA}<|im_end|>\n<|im_start|>assistant\n"
            
            T_START=$(date +%s%N)
            RESULTADO=$("$LLAMA_CLI" \
                -m "$MODEL" \
                --threads 2 \
                -c 1024 \
                -n 64 \
                --temp 0.2 \
                -p "$PROMPT" \
                --log-disable 2>/dev/null | \
                python3 "$DECODER" --codebook "$CODEBOOK" --quiet 2>/dev/null)
            T_END=$(date +%s%N)
            
            ELAPSED=$(( (T_END - T_START) / 1000000 ))  # ms
            PALAVRAS=$(echo "$RESULTADO" | wc -w)
            
            TOTAL_PALAVRAS=$((TOTAL_PALAVRAS + PALAVRAS))
            TOTAL_TEMPO=$((TOTAL_TEMPO + ELAPSED))
        done
        
        # Calcular médias
        if [ $TOTAL_TEMPO -gt 0 ]; then
            PALAVRAS_POR_S=$((TOTAL_PALAVRAS * 1000 / TOTAL_TEMPO))
        else
            PALAVRAS_POR_S=0
        fi
        
        # Determinar veredicto
        if [ $PALAVRAS_POR_S -gt 40 ]; then
            VEREDICTO="✅ Viável"
        elif [ $PALAVRAS_POR_S -gt 20 ]; then
            VEREDICTO="⚠️ Parcial"
        else
            VEREDICTO="❌ Lento"
        fi
        
        printf "║  %-5s │ %-8s │ %8d/s │   ---    │ %-12s║\n" \
            "$taxa" "$modo" "$PALAVRAS_POR_S" "$VEREDICTO" | tee -a "$RESULTS_FILE"
    done
done

echo "╚══════════════════════════════════════════════════════════╝" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "📄 Resultados salvos em: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "🚀 Benchmark concluído!"
