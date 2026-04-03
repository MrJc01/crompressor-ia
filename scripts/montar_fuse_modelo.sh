#!/usr/bin/env bash
# Script SRE para empaquetamento e montagem de modelo GGUF via VFS CROM
set -e

DIR_BASE="/home/j/Área de trabalho/crompressor-ia"
BIN="$DIR_BASE/crompressor_bin"
CODEBOOK="$DIR_BASE/dict.cromdb"
MODEL_GGUF="$DIR_BASE/models/qwen2.5-crom-dna.gguf"
MODEL_CROM="$DIR_BASE/models/qwen2.5-crom-dna.gguf.crom"
MNT_DIR="$DIR_BASE/mnt_crom"

echo "=========================================="
echo " 💿 CROM-IA FUSE MOUNT MANAGER"
echo "=========================================="

if [ ! -f "$MODEL_GGUF" ]; then
    echo "[ERRO] Modelo base não encontrado: $MODEL_GGUF"
    exit 1
fi

echo "[1/3] Verificando arquivo compactado (.crom)"
if [ ! -f "$MODEL_CROM" ]; then
    echo "      Compactando ($MODEL_GGUF) para CROM. Isso pode levar uns segundos..."
    "$BIN" pack -i "$MODEL_GGUF" -o "$MODEL_CROM" -c "$CODEBOOK" --cdc
    echo "      [OK] Modelo compactado HNSW O(1) gerado em $MODEL_CROM"
else
    echo "      [OK] Arquivo .crom encontrado. Compressão bypassada."
fi

echo "[2/3] Preparando Ponto de Montagem VFS"
mkdir -p "$MNT_DIR"
# Desmonta caso já exista algo montado
fusermount -u "$MNT_DIR" 2>/dev/null || true

echo "[3/3] Iniciando FUSE Mount via Mmap..."
"$BIN" mount -i "$MODEL_CROM" -m "$MNT_DIR" -c "$CODEBOOK" &
FUSE_PID=$!

# Aguarda a montagem
sleep 1
if mount | grep -q "$MNT_DIR"; then
    echo "      [SUCESSO] FUSE montado sob $MNT_DIR !"
    ls -lah "$MNT_DIR"
    echo ""
    echo "⚠️  O Daemon está rodando em plano de fundo (PID: $FUSE_PID). O modelo virtual está em:"
    echo "   $MNT_DIR/qwen2.5-crom-dna.gguf"
    echo "   Utilize fusermount -u $MNT_DIR para desmontar depois."
else
    echo "      [ERRO FATAL] Falha ao montar o daemon FUSE."
    exit 1
fi
