#!/usr/bin/env python3
"""
🧬 vPureDna v5 — Upload para HuggingFace Hub

Sobe modelo completo: safetensors, GGUFs, dataset, LoRA adapter, model card.

Uso:
  pip install huggingface_hub
  huggingface-cli login
  python3 upload_hf_vpuredna.py
"""

import os
import sys
from huggingface_hub import HfApi, login

REPO_ID = "CromIA/vpuredna-v5-qwen3-1.7b"
DIR_BASE = os.path.dirname(os.path.abspath(__file__))


def create_model_card():
    """Gera README.md para o HuggingFace."""
    return (
        "---\n"
        "language: pt\n"
        "license: apache-2.0\n"
        "tags:\n"
        "  - crompressor\n"
        "  - vpuredna\n"
        "  - qwen3\n"
        "  - lora\n"
        "  - dna-compression\n"
        "  - cognitive-compression\n"
        "base_model: Qwen/Qwen3-1.7B\n"
        "pipeline_tag: text-generation\n"
        "---\n\n"
        "# vPureDna v5 - Qwen3 1.7B + DNA Token (U+232C)\n\n"
        "Modelo fine-tunado com LoRA sobre Qwen3-1.7B para compressao cognitiva via tokens DNA.\n\n"
        "## Treinamento\n\n"
        "| Parametro | Valor |\n"
        "|-----------|-------|\n"
        "| Base | Qwen/Qwen3-1.7B |\n"
        "| Fine-tune | LoRA r=16 alpha=32 |\n"
        "| Dataset | 5000 amostras trifasicas (emissao/expansao/manutencao) |\n"
        "| Steps | 1000 |\n"
        "| Loss final | 1.34 |\n"
        "| GPU | NVIDIA A100-SXM4-40GB |\n"
        "| Token especial | U+232C (DNA marker) |\n"
        "| Idioma | Portugues (PT-BR) |\n\n"
        "## Downloads\n\n"
        "| Arquivo | Tamanho | Descricao |\n"
        "|---------|---------|-----------|\n"
        "| vpuredna_v5_Q4KM.gguf | 1.1 GB | GGUF Q4_K_M (uso local recomendado) |\n"
        "| vpuredna_v5.gguf | 3.3 GB | GGUF F16 (precisao total) |\n\n"
        "## Como usar\n\n"
        "### Via llama.cpp\n\n"
        "    huggingface-cli download MrJc01/vpuredna-v5-qwen3-1.7b vpuredna_v5_Q4KM.gguf\n"
        "    ./llama-cli -m vpuredna_v5_Q4KM.gguf -ngl 99 --temp 0.3\n\n"
        "### Via crompressor-ia (com DNA compression)\n\n"
        "    git clone https://github.com/MrJc01/crompressor-ia.git\n"
        "    cd crompressor-ia\n"
        "    ./chat_vpuredna_v5.sh\n\n"
        "## Projeto\n\n"
        "- GitHub: https://github.com/MrJc01/crompressor-ia\n"
        "- DNA Compression: Tokens U+232C comprimem frases recorrentes em IDs curtos\n"
        "- Pipeline: Texto -> Compress -> LLM -> Expand -> Texto\n"
    )


def main():
    print("=" * 60)
    print(" 🧬 vPureDna v5 — Upload para HuggingFace")
    print("=" * 60)

    # Login
    try:
        api = HfApi()
        user = api.whoami()
        print(f"  ✅ Logado como: {user['name']}")
    except Exception:
        print("  ⚠️  Fazendo login...")
        login()
        api = HfApi()

    # Criar repo
    api.create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
    print(f"  ✅ Repo: https://huggingface.co/{REPO_ID}")

    # 1. Model Card
    print("\n[1/5] Uploading Model Card...")
    card_path = "/tmp/vpuredna_readme.md"
    with open(card_path, "w") as f:
        f.write(create_model_card())
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=REPO_ID, repo_type="model",
        commit_message="Add model card"
    )
    print("  ✅ README.md")

    # 2. GGUF Q4_K_M
    q4_path = os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5_Q4KM.gguf")
    if os.path.exists(q4_path):
        size_gb = os.path.getsize(q4_path) / 1e9
        print(f"\n[2/5] Uploading GGUF Q4_K_M ({size_gb:.1f} GB)...")
        api.upload_file(
            path_or_fileobj=q4_path,
            path_in_repo="vpuredna_v5_Q4KM.gguf",
            repo_id=REPO_ID, repo_type="model",
            commit_message="Add GGUF Q4_K_M (1.1GB)"
        )
        print("  ✅ vpuredna_v5_Q4KM.gguf")
    else:
        print(f"  ⚠️  Q4 não encontrado: {q4_path}")

    # 3. GGUF F16
    f16_path = os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5.gguf")
    if os.path.exists(f16_path):
        size_gb = os.path.getsize(f16_path) / 1e9
        print(f"\n[3/5] Uploading GGUF F16 ({size_gb:.1f} GB)...")
        api.upload_file(
            path_or_fileobj=f16_path,
            path_in_repo="vpuredna_v5.gguf",
            repo_id=REPO_ID, repo_type="model",
            commit_message="Add GGUF F16 (3.3GB)"
        )
        print("  ✅ vpuredna_v5.gguf")
    else:
        print(f"  ⚠️  F16 não encontrado: {f16_path}")

    # 4. Dataset
    ds_path = os.path.join(DIR_BASE, "vPureDna", "05_colab", "output", "dataset_vpuredna_colab.jsonl")
    if os.path.exists(ds_path):
        print(f"\n[4/5] Uploading dataset...")
        api.upload_file(
            path_or_fileobj=ds_path,
            path_in_repo="training/dataset_vpuredna_colab.jsonl",
            repo_id=REPO_ID, repo_type="model",
            commit_message="Add training dataset (5000 samples)"
        )
        print("  ✅ dataset_vpuredna_colab.jsonl")
    else:
        print(f"  ⚠️  Dataset não encontrado: {ds_path}")

    # 5. LoRA adapter
    lora_path = os.path.join(DIR_BASE, "vPureDna", "05_colab", "output", "lora-adapter")
    if os.path.exists(lora_path):
        print(f"\n[5/5] Uploading LoRA adapter...")
        api.upload_folder(
            folder_path=lora_path,
            path_in_repo="lora-adapter/",
            repo_id=REPO_ID, repo_type="model",
            commit_message="Add LoRA adapter (checkpoint-1000)"
        )
        print("  ✅ lora-adapter/")
    else:
        print(f"  ⚠️  LoRA adapter não encontrado: {lora_path}")

    # 6. Codebook DNA
    cb_path = os.path.join(DIR_BASE, "codebooks", "codebook_1x5_dinamico_expandido.json")
    if os.path.exists(cb_path):
        print(f"\n[BONUS] Uploading DNA codebook...")
        api.upload_file(
            path_or_fileobj=cb_path,
            path_in_repo="dna/codebook_1x5_dinamico_expandido.json",
            repo_id=REPO_ID, repo_type="model",
            commit_message="Add DNA codebook"
        )
        print("  ✅ codebook")

    # 7. DNA Tokenizer
    tok_path = os.path.join(DIR_BASE, "vPureDna", "01_encoder", "tokenizer_dna.py")
    if os.path.exists(tok_path):
        api.upload_file(
            path_or_fileobj=tok_path,
            path_in_repo="dna/tokenizer_dna.py",
            repo_id=REPO_ID, repo_type="model",
            commit_message="Add DNA tokenizer/compressor"
        )
        print("  ✅ tokenizer_dna.py")

    print(f"\n{'=' * 60}")
    print(f" ✅ UPLOAD COMPLETO!")
    print(f" 🔗 https://huggingface.co/{REPO_ID}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
