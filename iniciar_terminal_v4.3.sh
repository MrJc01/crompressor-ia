#!/usr/bin/env bash
# ==============================================================================
# ⌬  CROM-IA V4.3: MONITOR DE CHAT TERMINAL (ORQUESTRADOR SRE)
# ==============================================================================

DIR_BASE="$(dirname "$(realpath "$0")")"
VENV_PYTHON="$DIR_BASE/pesquisa/.venv/bin/python3"
SCRIPT_PYTHON="$DIR_BASE/v4.3_cognitive_leap/chat_terminal_v4.3.py"

# Cores
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧬 Ativando Rede Neural CROM-IA v4.3...${NC}"

# 1. Verificar dependências
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERRO] VirtualEnvironment Python não encontrado em pesquisa/.venv/"
    exit 1
fi

# 2. Inicializar ponte MMap (Obrigatório para o solo semântico)
"$VENV_PYTHON" "$DIR_BASE/scripts/setup_mmap_bridge.py"

# 3. Lançar interface TUI
"$VENV_PYTHON" "$SCRIPT_PYTHON"
