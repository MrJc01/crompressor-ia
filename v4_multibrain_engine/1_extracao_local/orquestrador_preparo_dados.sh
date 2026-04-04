#!/usr/bin/env bash
# ==============================================================================
# CROM-IA V4.0: MOTOR DE EXPANSAO NEURAL (LOCAL WORKER)
# Rodando isso daqui, você não precisa fazer mais nada além de ir tomar um café.
# Ele fará o Download, aplicará as heurísticas W/F/P e entregará tudo pronto na sub-pasta.
# ==============================================================================

echo "=== INICIANDO EXTRATOR INDUSTRIAL DE DADOS V4 ==="

# 1. Baixar Bases usando antena HuggingFace
python3 downloader_dominios.py
if [ $? -ne 0 ]; then
    echo "❌ Erro ao baixar dados. O 'pip install datasets' falhou?"
    exit 1
fi

mkdir -p arquivos_para_o_colab

echo "⚙️  Processando Cérebro Python (Domínio P)..."
python3 gerador_prefixos_namespaced.py P datasets_brutos/P_python.jsonl
python3 transpilador_v4_multibrain.py datasets_brutos/P_python.jsonl codebook_P_W_V4.json codebook_P_F_V4.json codebook_P_P_V4.json arquivos_para_o_colab/dataset_P_Hibrido.jsonl

echo "⚙️  Processando Cérebro Medicina (Domínio M)..."
python3 gerador_prefixos_namespaced.py M datasets_brutos/M_medicina.jsonl
python3 transpilador_v4_multibrain.py datasets_brutos/M_medicina.jsonl codebook_M_W_V4.json codebook_M_F_V4.json codebook_M_P_V4.json arquivos_para_o_colab/dataset_M_Hibrido.jsonl

echo "⚙️  Processando Cérebro Conversação PT-BR (Domínio G)..."
python3 gerador_prefixos_namespaced.py G datasets_brutos/G_geral_ptbr.jsonl
python3 transpilador_v4_multibrain.py datasets_brutos/G_geral_ptbr.jsonl codebook_G_W_V4.json codebook_G_F_V4.json codebook_G_P_V4.json arquivos_para_o_colab/dataset_G_Hibrido.jsonl

echo "=========================================================="
echo "🎉 EXPANSAO CONCLUIDA COM SUCESSO ABSOLUTO!"
echo "Abra a pasta 'arquivos_para_o_colab'."
echo "Voce encontrara os JSONL finais hibridos. Mande pro Google Drive e rode seu Colab V4!"
echo "=========================================================="
