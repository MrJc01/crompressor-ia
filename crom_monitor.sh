#!/bin/bash
# ==============================================================================
# 🧬  CROM-IA V2 : Terminal SRE Monitor e Orquestrador
# ==============================================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$BASE_DIR/scripts"
MODELS_DIR="$BASE_DIR/models"
MNT_CROM="$BASE_DIR/mnt_crom"
BIN_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

# Cores e Estilos
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

function draw_header() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "  ██████╗██████╗  ██████╗ ███╗   ███╗      ██╗█████╗ "
    echo " ██╔════╝██╔══██╗██╔═══██╗████╗ ████║      ██║██╔══██╗"
    echo " ██║     ██████╔╝██║   ██║██╔████╔██║█████╗██║███████║"
    echo " ██║     ██╔══██╗██║   ██║██║╚██╔╝██║╚════╝██║██╔══██║"
    echo " ╚██████╗██║  ██║╚██████╔╝██║ ╚═╝ ██║      ██║██║  ██║"
    echo "  ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝      ╚═╝╚═╝  ╚═╝ v2"
    echo -e "${NC}"
    echo -e " ⚡ Orquestrador Central SRE | Motor Nativo AVX"
    echo "=================================================================="
}

function check_health() {
    echo -e "${BOLD}[ Verificação de Sistemas ]${NC}"
    
    # 1. Motor Nativo
    if [ -x "$BIN_NATIVO" ]; then
        echo -e " [${GREEN}OK${NC}] Motor C++ AVX ($BIN_NATIVO)"
    else
        echo -e " [${RED}FAIL${NC}] Motor C++ não encontrado!"
    fi
    
    # 2. FUSE Status
    if mount | grep -q "$MNT_CROM"; then
        echo -e " [${GREEN}OK${NC}] FileSystem FUSE Ativo (Zero-Copy Mmap)"
    else
        echo -e " [${YELLOW}WARN${NC}] FUSE inativo (Possível cold-start de 14s)"
    fi

    # 3. Modelos
    QTD_MODELS=$(ls "$MODELS_DIR"/*.gguf 2>/dev/null | wc -l)
    if [ "$QTD_MODELS" -gt 0 ]; then
        echo -e " [${GREEN}OK${NC}] $QTD_MODELS GGUFs Genômicos encontrados"
    else
        echo -e " [${RED}FAIL${NC}] Nenhum arquivo .gguf em models/"
    fi
    echo "------------------------------------------------------------------"
}

function menu() {
    echo -e "${BOLD}MÓDULOS DE OPERAÇÃO:${NC}"
    echo -e "  [ 1 ] ${CYAN}Chat DNA Supersônico${NC} (Pipeline com Decoder O(1))"
    echo -e "  [ 2 ] ${GREEN}Ligar Motor FUSE${NC} (Abstrair tempo de RAM/Cold-start)"
    echo -e "  [ 3 ] ${YELLOW}Laboratório SRE${NC} (Benchmark Texto vs DNA)"
    echo -e "  [ 4 ] Treinar mais Codebooks (Instruções Colab)"
    echo -e "  [ 5 ] Teste Massivo (Benchmark todas as taxas)"
    echo -e "  [ 0 ] Sair"
    echo ""
    echo -ne "Selecione o Módulo > "
    read OPCAO
    
    case $OPCAO in
        1)
            echo -e "\n${CYAN}Modelos Genômicos (GGUF) disponíveis:${NC}"
            for f in "$MODELS_DIR"/*.gguf; do [ -e "$f" ] && basename "$f"; done
            echo ""
            echo -ne "Digite a TAXA do modelo (ex: 1x3, 1x5)  > "
            read TAXA_ESCOLHIDA
            echo -ne "Digite o MODO do modelo (fixo, dinamico) > "
            read MODO_ESCOLHIDO
            echo -e "\nIniciando Chat DNA ($TAXA_ESCOLHIDA $MODO_ESCOLHIDO)..."
            python3 "$BASE_DIR/scripts/chat_dna_v2.py" "${TAXA_ESCOLHIDA:-1x5}" "${MODO_ESCOLHIDO:-fixo}"
            echo "Aperte Enter para voltar..."
            read
            ;;
        2)
            echo -e "\nIniciando Montagem FUSE..."
            bash "$SCRIPTS_DIR/montar_fuse_modelo.sh" || echo "Script FUSE em ajuste. Você pode chamar direto!"
            echo "Aperte Enter para voltar..."
            read
            ;;
        3)
            echo -e "\nIniciando Laboratório Termodinâmico..."
            python3 "$BASE_DIR/lab_dna_crom.py"
            echo "Aperte Enter para voltar..."
            read
            ;;
        4)
            echo -e "\nPara rodar no COLAB:"
            echo "1. Abra colab.research.google.com (Ative T4 GPU)"
            echo "2. Suba colab/treinar_codebook.py + o dataset .jsonl"
            echo "3. Execute as células, pegue o gguf e ponha na pasta models/"
            echo "Aperte Enter para voltar..."
            read
            ;;
        5)
            echo -e "\nIniciando Bateria de Testes..."
            bash "$SCRIPTS_DIR/benchmark_codebooks.sh"
            echo "Aperte Enter para voltar..."
            read
            ;;
        0)
            echo "Desligando Orquestrador..."
            exit 0
            ;;
        *)
            echo "Opção inválida!"
            sleep 1
            ;;
    esac
}

while true; do
    draw_header
    check_health
    menu
done
