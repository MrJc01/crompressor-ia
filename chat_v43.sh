#!/bin/bash
# ==============================================================================
# ⌬  CROM-IA v4.3 — ORQUESTRADOR DE CHAT (SRE HARDENED)
# ==============================================================================
# Este script automatiza o boot seguro do Salto Cognitivo v4.3.
# ==============================================================================

DIR_BASE="$(dirname "$(realpath "$0")")"
VENV_PYTHON="$DIR_BASE/pesquisa/.venv/bin/python3"
KERNEL_CHAT="$DIR_BASE/v4.3_cognitive_leap/chat_terminal_v4.3.py"
SETUP_BRIDGE="$DIR_BASE/scripts/setup_mmap_bridge.py"

# Cores e Estética
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}   🧬 CROM-IA v4.3 : INICIALIZADOR DE ELITE (Fase 7)         ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

# 1. Verificação de Saúde do Sistema (Swap Check)
echo -e "\n${YELLOW}[1/4] Verificando Pressão de Memória (SRE)...${NC}"
SWAP_TOTAL=$(free -m | grep Swap | awk '{print $2}')
SWAP_USED=$(free -m | grep Swap | awk '{print $3}')

if [ "$SWAP_TOTAL" -gt 0 ]; then
    SWAP_PCT=$(( 100 * SWAP_USED / SWAP_TOTAL ))
    if [ "$SWAP_PCT" -gt 90 ]; then
        echo -e "${RED}[FALHA] Swap em ${SWAP_PCT}%. Memória insuficiente para mlock.${NC}"
        echo -e "${RED}Feche processos pesados antes de prosseguir.${NC}"
        exit 1
    else
        echo -e "   ${GREEN}✅${NC} Swap: ${SWAP_PCT}% (Estável)"
    fi
else
    echo -e "   ${YELLOW}⚠️  Aviso:${NC} Swap não detectado. Risco de OOM."
fi

# 2. Verificação do Ambiente Virtual
echo -e "${YELLOW}[2/4] Verificando VirtualEnvironment...${NC}"
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}[ERRO] Venv não encontrado em: $VENV_PYTHON${NC}"
    exit 1
else
    echo -e "   ${GREEN}✅${NC} Ambiente Python: OK"
fi

# 3. Higienização e Daemon Check
echo -e "${YELLOW}[3/4] Higienizando processos anteriores...${NC}"
pkill -f "chat_terminal_v4.3.py" 2>/dev/null || true
echo -e "   ${GREEN}✅${NC} Limpeza concluída."

# 4. Inicialização da Ponte Solo Semântico (⌬)
echo -e "${YELLOW}[4/4] Estabelecendo Ponte MMap (LSH Bridge)...${NC}"
# Forçamos a criação/limpeza da ponte antes de subir a IA
if [ -f "$SETUP_BRIDGE" ]; then
    "$VENV_PYTHON" "$SETUP_BRIDGE" > /dev/null 2>&1
    echo -e "   ${GREEN}✅${NC} Solo Semântico: Sincronizado."
else
    echo -e "   ${RED}❌ Erro:${NC} Script de bridge não encontrado."
    exit 1
fi

echo -e "\n${GREEN}🚀 LANÇANDO SALTO COGNITIVO v4.3...${NC}\n"
sleep 1

# Execução final do Kernel
"$VENV_PYTHON" "$KERNEL_CHAT"
