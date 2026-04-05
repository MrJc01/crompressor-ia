#!/usr/bin/env bash
# ==============================================================================
# CROM-IA V4.2: Monitor de Chat — Configuração + Orquestração de Cérebros
# ==============================================================================
# Um painel TUI interativo que permite:
#   ✅ Ver todos os cérebros disponíveis
#   ✅ Ativar/desativar cérebros individualmente
#   ✅ Adicionar arquivos/pastas para contexto RAG
#   ✅ Configurar parâmetros antes de iniciar
#   ✅ Lançar o chat com a configuração escolhida
# ==============================================================================

set -euo pipefail

# ── Caminhos ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_CLI="$PROJECT_ROOT/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELS_DIR="$SCRIPT_DIR/micro_cerebros"
RAG_ENGINE="$SCRIPT_DIR/rag_contexto.py"
DECODER="$SCRIPT_DIR/decodificador_dna/decodificador_dna.py"

# ── Estado Global ─────────────────────────────────────────────────────────────
declare -A CEREBROS_STATUS    # nome → on/off
declare -a CEREBROS_NOMES     # lista ordenada de nomes
declare -a CEREBROS_PATHS     # caminhos dos .gguf
declare -a RAG_ARQUIVOS=()    # arquivos para contexto
declare -a RAG_PASTAS=()      # pastas para contexto
BASE_MODEL=""
CONTEXT_WINDOW=1024
TEMPERATURE=0.7
MAX_TOKENS=512

# ── Cores ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

# ── Inicialização ─────────────────────────────────────────────────────────────
inicializar() {
    # Encontrar modelo base
    for gguf in "$MODELS_DIR"/*.gguf; do
        if [ -f "$gguf" ] && [[ ! "$gguf" == *_lora.gguf ]]; then
            BASE_MODEL="$gguf"
            break
        fi
    done

    # Encontrar todos os LoRAs
    CEREBROS_NOMES=()
    CEREBROS_PATHS=()
    for lora in "$MODELS_DIR"/*_lora.gguf; do
        if [ -f "$lora" ]; then
            nome=$(basename "$lora" _lora.gguf)
            CEREBROS_NOMES+=("$nome")
            CEREBROS_PATHS+=("$lora")
            CEREBROS_STATUS["$nome"]="on"  # Todos ativos por padrão
        fi
    done
}

# ── Desenhar Interface ────────────────────────────────────────────────────────
desenhar_header() {
    clear
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}${BOLD}       🧠 CROM-IA V4.2 — Monitor de Orquestração           ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC}  Configure seus cérebros e contexto antes de iniciar       ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

desenhar_status_modelo() {
    echo -e "${WHITE}── Modelo Base ─────────────────────────────────────────────${NC}"
    if [ -n "$BASE_MODEL" ]; then
        local tamanho=$(du -h "$BASE_MODEL" 2>/dev/null | cut -f1)
        echo -e "   ${GREEN}✅${NC} $(basename "$BASE_MODEL") ${DIM}($tamanho)${NC}"
    else
        echo -e "   ${RED}❌ Nenhum modelo base encontrado${NC}"
        echo -e "   ${DIM}Coloque um .gguf em: $MODELS_DIR${NC}"
    fi
    echo ""
}

desenhar_cerebros() {
    echo -e "${WHITE}── Micro-Cérebros (LoRAs) ──────────────────────────────────${NC}"

    if [ ${#CEREBROS_NOMES[@]} -eq 0 ]; then
        echo -e "   ${YELLOW}⚠️  Nenhum LoRA encontrado${NC}"
        echo -e "   ${DIM}Coloque arquivos *_lora.gguf em: $MODELS_DIR${NC}"
    else
        local ativos=0
        for i in "${!CEREBROS_NOMES[@]}"; do
            local nome="${CEREBROS_NOMES[$i]}"
            local path="${CEREBROS_PATHS[$i]}"
            local tamanho=$(du -h "$path" 2>/dev/null | cut -f1)
            local num=$((i + 1))

            if [ "${CEREBROS_STATUS[$nome]}" = "on" ]; then
                echo -e "   ${GREEN}[$num] ✅ ON ${NC} ${BOLD}$nome${NC} ${DIM}($tamanho)${NC}"
                ativos=$((ativos + 1))
            else
                echo -e "   ${RED}[$num] ⬚ OFF${NC} ${DIM}$nome ($tamanho)${NC}"
            fi
        done
        echo ""
        echo -e "   ${CYAN}$ativos/${#CEREBROS_NOMES[@]}${NC} cérebros ativos"
    fi
    echo ""
}

desenhar_rag() {
    echo -e "${WHITE}── Contexto RAG (Arquivos/Pastas) ──────────────────────────${NC}"

    local total_rag=$(( ${#RAG_ARQUIVOS[@]} + ${#RAG_PASTAS[@]} ))

    if [ "$total_rag" -eq 0 ]; then
        echo -e "   ${DIM}Nenhum arquivo/pasta carregado${NC}"
    else
        for arq in "${RAG_ARQUIVOS[@]}"; do
            echo -e "   ${GREEN}📄${NC} $arq"
        done
        for pasta in "${RAG_PASTAS[@]}"; do
            local count=$(find "$pasta" -type f 2>/dev/null | wc -l)
            echo -e "   ${GREEN}📂${NC} $pasta ${DIM}($count arquivos)${NC}"
        done
    fi
    echo ""
}

desenhar_config() {
    echo -e "${WHITE}── Configuração ────────────────────────────────────────────${NC}"
    echo -e "   Contexto   : ${CYAN}$CONTEXT_WINDOW${NC} tokens"
    echo -e "   Temperatura: ${CYAN}$TEMPERATURE${NC}"
    echo -e "   Max tokens : ${CYAN}$MAX_TOKENS${NC}"
    echo -e "   DNA Decoder: $([ -f "$DECODER" ] && echo -e "${GREEN}Disponível 🧬${NC}" || echo -e "${DIM}N/A${NC}")"
    echo ""
}

desenhar_menu() {
    echo -e "${WHITE}── Ações ───────────────────────────────────────────────────${NC}"
    echo -e "   ${BOLD}[1-9]${NC}  Toggle cérebro ON/OFF"
    echo -e "   ${BOLD}[a]${NC}    Adicionar arquivo para RAG"
    echo -e "   ${BOLD}[p]${NC}    Adicionar pasta para RAG"
    echo -e "   ${BOLD}[r]${NC}    Remover último item RAG"
    echo -e "   ${BOLD}[c]${NC}    Limpar todo contexto RAG"
    echo -e "   ${BOLD}[t]${NC}    Mudar temperatura"
    echo -e "   ${BOLD}[w]${NC}    Mudar janela de contexto"
    echo -e "   ${BOLD}[*]${NC}    Ativar TODOS os cérebros"
    echo -e "   ${BOLD}[0]${NC}    Desativar TODOS os cérebros"
    echo -e "   ${BOLD}────────────────────────────────${NC}"
    echo -e "   ${GREEN}${BOLD}[ENTER]${NC}${GREEN} 🚀 INICIAR CHAT${NC}"
    echo -e "   ${RED}[q]${NC}     Sair"
    echo ""
}

# ── Ações ─────────────────────────────────────────────────────────────────────
toggle_cerebro() {
    local idx=$1
    if [ "$idx" -ge 0 ] && [ "$idx" -lt ${#CEREBROS_NOMES[@]} ]; then
        local nome="${CEREBROS_NOMES[$idx]}"
        if [ "${CEREBROS_STATUS[$nome]}" = "on" ]; then
            CEREBROS_STATUS["$nome"]="off"
        else
            CEREBROS_STATUS["$nome"]="on"
        fi
    fi
}

adicionar_arquivo() {
    echo ""
    echo -ne "   ${CYAN}Caminho do arquivo:${NC} "
    read -r caminho

    # Expandir ~ e variáveis
    caminho=$(eval echo "$caminho")

    if [ -f "$caminho" ]; then
        RAG_ARQUIVOS+=("$(realpath "$caminho")")
        echo -e "   ${GREEN}✅ Arquivo adicionado!${NC}"
    else
        echo -e "   ${RED}❌ Arquivo não encontrado: $caminho${NC}"
    fi
    sleep 1
}

adicionar_pasta() {
    echo ""
    echo -ne "   ${CYAN}Caminho da pasta:${NC} "
    read -r caminho

    caminho=$(eval echo "$caminho")

    if [ -d "$caminho" ]; then
        RAG_PASTAS+=("$(realpath "$caminho")")
        echo -e "   ${GREEN}✅ Pasta adicionada!${NC}"
    else
        echo -e "   ${RED}❌ Pasta não encontrada: $caminho${NC}"
    fi
    sleep 1
}

remover_ultimo_rag() {
    if [ ${#RAG_PASTAS[@]} -gt 0 ]; then
        unset 'RAG_PASTAS[-1]'
        echo -e "   ${YELLOW}Última pasta removida${NC}"
    elif [ ${#RAG_ARQUIVOS[@]} -gt 0 ]; then
        unset 'RAG_ARQUIVOS[-1]'
        echo -e "   ${YELLOW}Último arquivo removido${NC}"
    fi
    sleep 0.5
}

mudar_temperatura() {
    echo ""
    echo -ne "   ${CYAN}Nova temperatura (0.1 - 2.0) [atual: $TEMPERATURE]:${NC} "
    read -r nova
    if [[ "$nova" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        TEMPERATURE="$nova"
    fi
}

mudar_contexto() {
    echo ""
    echo -ne "   ${CYAN}Nova janela de contexto (512/1024/2048/4096) [atual: $CONTEXT_WINDOW]:${NC} "
    read -r nova
    if [[ "$nova" =~ ^[0-9]+$ ]]; then
        CONTEXT_WINDOW="$nova"
    fi
}

ativar_todos() {
    for nome in "${CEREBROS_NOMES[@]}"; do
        CEREBROS_STATUS["$nome"]="on"
    done
}

desativar_todos() {
    for nome in "${CEREBROS_NOMES[@]}"; do
        CEREBROS_STATUS["$nome"]="off"
    done
}

# ── Lançar Chat ───────────────────────────────────────────────────────────────
lancar_chat() {
    # Verificar modelo base
    if [ -z "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL" ]; then
        echo -e "${RED}❌ Modelo base não encontrado! Não é possível iniciar.${NC}"
        sleep 2
        return
    fi

    # Montar flags de LoRA
    local LORA_FLAGS=()
    local LORA_COUNT=0
    for i in "${!CEREBROS_NOMES[@]}"; do
        local nome="${CEREBROS_NOMES[$i]}"
        if [ "${CEREBROS_STATUS[$nome]}" = "on" ]; then
            local escala="1.0"
            if [[ "$nome" == *"Base_PTBR"* ]]; then escala="1.0"; fi
            if [[ "$nome" == *"Python_DNA"* ]]; then escala="0.25"; fi
            if [[ "$nome" == *"DPO_Preference"* ]]; then escala="0.75"; fi

            LORA_FLAGS+=("--lora-scaled" "${CEREBROS_PATHS[$i]}:$escala")
            LORA_COUNT=$((LORA_COUNT + 1))
        fi
    done

    # Montar flags de RAG
    local RAG_ARGS=()
    local HAS_RAG=false
    for arq in "${RAG_ARQUIVOS[@]}"; do
        RAG_ARGS+=("--arquivo" "$arq")
        HAS_RAG=true
    done
    for pasta in "${RAG_PASTAS[@]}"; do
        RAG_ARGS+=("--pasta" "$pasta")
        HAS_RAG=true
    done

    # Gerar system prompt lógico puro
    local SYSTEM_PROMPT="<|im_start|>system\nVocê é CROM-IA, assistente brasileiro inteligente com compressão DNA ativa. Você responde de forma lógica e estruturada.<|im_end|>\n"

    if [ "$HAS_RAG" = true ] && [ -f "$RAG_ENGINE" ]; then
        echo ""
        echo -e "${CYAN}📂 Processando arquivos para contexto RAG...${NC}"
        SYSTEM_PROMPT=$(python3 "$RAG_ENGINE" "${RAG_ARGS[@]}" --prompt-only 2>/dev/null)
        CONTEXT_WINDOW=2048  # Aumentar para RAG
        echo -e "${GREEN}✅ Contexto RAG injetado!${NC}"
    fi

    # Prompt temporário
    local PROMPT_FILE=$(mktemp /tmp/crom_prompt_XXXXXX.txt)
    echo "$SYSTEM_PROMPT" > "$PROMPT_FILE"

    # Resumo antes de lançar
    clear
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}${BOLD}         🚀 CROM-IA V4.2 — Lançando Chat...                 ${NC}${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "   Modelo    : ${CYAN}$(basename "$BASE_MODEL")${NC}"
    echo -e "   LoRAs     : ${CYAN}$LORA_COUNT empilhados${NC}"
    echo -e "   RAG       : $([ "$HAS_RAG" = true ] && echo -e "${GREEN}ATIVO ✅${NC}" || echo -e "${DIM}Desligado${NC}")"
    echo -e "   Contexto  : ${CYAN}$CONTEXT_WINDOW tokens${NC}"
    echo -e "   Temp      : ${CYAN}$TEMPERATURE${NC}"
    echo ""
    echo -e "   ${DIM}Ctrl+C para voltar ao monitor${NC}"
    echo ""

    # Executar llama-cli
    "$LLAMA_CLI" \
        -m "$BASE_MODEL" \
        "${LORA_FLAGS[@]}" \
        -c "$CONTEXT_WINDOW" \
        -n "$MAX_TOKENS" \
        --threads 4 \
        -b 256 \
        --mlock \
        --temp 0.3 \
        --repeat-penalty 1.0 \
        --conversation \
        --in-prefix "<|im_start|>user\n" \
        --in-suffix "<|im_end|>\n<|im_start|>assistant\n" \
        --reverse-prompt "<|im_end|>" \
        --file "$PROMPT_FILE" \
        || true

    # Cleanup
    rm -f "$PROMPT_FILE"

    echo ""
    echo -e "${YELLOW}Chat encerrado. Voltando ao monitor...${NC}"
    sleep 2
}

# ── Loop Principal ────────────────────────────────────────────────────────────
main() {
    inicializar

    # Processar args da linha de comando (pré-carregar)
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --arquivo)
                shift; [ -n "${1:-}" ] && [ -f "$1" ] && RAG_ARQUIVOS+=("$(realpath "$1")"); shift ;;
            --pasta)
                shift; [ -n "${1:-}" ] && [ -d "$1" ] && RAG_PASTAS+=("$(realpath "$1")"); shift ;;
            *) shift ;;
        esac
    done

    while true; do
        desenhar_header
        desenhar_status_modelo
        desenhar_cerebros
        desenhar_rag
        desenhar_config
        desenhar_menu

        echo -ne "   ${BOLD}Ação:${NC} "
        read -r -n1 acao
        echo ""

        case "$acao" in
            [1-9])
                toggle_cerebro $((acao - 1))
                ;;
            a|A)
                adicionar_arquivo
                ;;
            p|P)
                adicionar_pasta
                ;;
            r|R)
                remover_ultimo_rag
                ;;
            c|C)
                RAG_ARQUIVOS=()
                RAG_PASTAS=()
                ;;
            t|T)
                mudar_temperatura
                ;;
            w|W)
                mudar_contexto
                ;;
            '*')
                ativar_todos
                ;;
            0)
                desativar_todos
                ;;
            q|Q)
                echo ""
                echo -e "${DIM}Até logo! 🧠${NC}"
                exit 0
                ;;
            '')
                lancar_chat
                ;;
        esac
    done
}

main "$@"
