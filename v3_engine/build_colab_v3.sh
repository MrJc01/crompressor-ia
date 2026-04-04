#!/bin/bash
set -e

cd "/home/j/Área de trabalho/crompressor-ia/v3_engine"

echo "============================================================"
echo " 🧬 CROM-IA V3.5: BUILD PIPELINE 100k (Dados Reais)"
echo "============================================================"

echo "1. Baixando datasets REAIS do HuggingFace..."
python3 downloader_chat_real.py

echo "2. Mixando Chat Real + Código V3 Comprimido..."
python3 gerar_dataset_v3_lora.py

echo "3. Empacotando Kit 100k para o COLAB..."
rm -f ../colab_v3_training_kit.zip
zip -r colab_v3_training_kit.zip data/dataset_v3_lora.jsonl macro_codebook_v3.json
mv colab_v3_training_kit.zip ../

echo "============================================================"
echo " ✅ SUCESSO! Kit de 100k amostras reais gerado!"
echo " Arquivo: colab_v3_training_kit.zip"
echo "============================================================"
