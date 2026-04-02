#!/bin/bash
set -e

echo "==========================================================="
echo "  [CROM-LLM-TRAINER] Módulo de Integração SRE Iniciado"
echo "==========================================================="

WORKSPACE="/home/j/Área de trabalho/crompressor-ia"
POC_DIR="$WORKSPACE/pesquisa/poc_dataloader"
CROM_BIN="$WORKSPACE/crompressor_bin"

if [ ! -f "$CROM_BIN" ]; then
    echo "Erro Crítico: Binário crompressor base não está no root. Copie o binário para '$CROM_BIN'."
    exit 1
fi

cd "$POC_DIR"

echo -e "\n[1/4] Gerando Mock Dataset (Massa Entrópica 150k linhas)..."
pip install tqdm > /dev/null 2>&1 || true
python3 generate_mock_dataset.py

# Estruturando Pastas do Dataset
mkdir -p dataset_raw
mv mock_dataset.jsonl dataset_raw/

echo -e "\n[2/4] Encapsulando no Monólito SquashFS..."
rm -f dataset.sqsh
mksquashfs dataset_raw dataset.sqsh -noI -noD -noX -noF -no-xattrs

echo -e "\n[3/4] CROM V23 Neural Pack (The Sovereign Compiler)..."
rm -f dict.cromdb sovereign.crom
# Primeiro treinamos o dicionário (Codebook) a partir do arquivo bruto
"$CROM_BIN" train -i dataset.sqsh -o dict.cromdb -s 4096

# Em seguida abstraimos a realidade com o pack
"$CROM_BIN" pack -i dataset.sqsh -o sovereign.crom -c dict.cromdb 

echo -e "\n[4/4] FUSE Cascading (O Despertar do Storage L1)"
mkdir -p mnt_crom mnt_squash lower upper work magic_merge

# Montagem CROM
echo "  -> Montando Camada Baixa: CROM FUSE"
"$CROM_BIN" mount -i sovereign.crom -m mnt_crom -c dict.cromdb &
CROM_PID=$!
sleep 2

# Extraindo o FUSE encapsulado
TARGET_SQSH=$(ls mnt_crom | head -n 1)
echo "  -> Montando Camada Squash: $TARGET_SQSH"
squashfuse "mnt_crom/$TARGET_SQSH" mnt_squash

# Camada Viva
echo "  -> Montando Camada Alta: Overlay (rw)"
fuse-overlayfs -o lowerdir=mnt_squash,upperdir=upper,workdir=work magic_merge

echo -e "\n[INJEÇÃO] O Dataset foi materializado na Mente RAM:"
ls -lh magic_merge/

echo -e "\n============================================="
echo " DISPARANDO NEGRO DE TREINAMENTO PYTORCH O(1)"
echo "============================================="
python3 train_llm_poc.py --dataset "./magic_merge/mock_dataset.jsonl"


echo -e "\n============================================="
echo " SRE TEARDOWN: Dissecando a Realidade Falsa..."
echo "============================================="
fusermount -uz magic_merge || true
fusermount -uz mnt_squash || true
fusermount -uz mnt_crom || true
kill $CROM_PID 2>/dev/null || true

rm -rf mnt_crom mnt_squash lower upper work magic_merge dataset_raw dataset.sqsh sovereign.crom dict.cromdb

echo "[SUCESSO] Todo material digital obliterado! (Zero-Footprint)"
