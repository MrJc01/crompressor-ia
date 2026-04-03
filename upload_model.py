#!/usr/bin/env python3
"""
🧬 CROM-IA: Upload completo de artefatos para HuggingFace
Sobe modelos GGUF + codebooks JSON para o repositório CromIA/CROM-IA-V1-DNA
"""
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Instalando huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi

api = HfApi()
REPO_ID = "CromIA/CROM-IA-V1-DNA"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artefatos para subir
ARQUIVOS = [
    # Modelos GGUF
    ("models/crom-dna-1x3-fixo.gguf", "crom-dna-1x3-fixo.gguf"),
    ("models/crom-dna-1x5-fixo.gguf", "crom-dna-1x5-fixo.gguf"),
    # Codebooks (essenciais para decodificação)
    ("codebooks/codebook_1x3_fixo.json", "codebooks/codebook_1x3_fixo.json"),
    ("codebooks/codebook_1x5_fixo.json", "codebooks/codebook_1x5_fixo.json"),
    ("codebooks/codebook_1x3_dinamico.json", "codebooks/codebook_1x3_dinamico.json"),
    ("codebooks/codebook_1x5_dinamico.json", "codebooks/codebook_1x5_dinamico.json"),
]

print(f"🧬 CROM-IA Upload para HuggingFace ({REPO_ID})")
print("=" * 55)

erros = 0
for local_path, remote_path in ARQUIVOS:
    full_path = os.path.join(BASE_DIR, local_path)
    if not os.path.exists(full_path):
        print(f"  ⏭️  {local_path} (não encontrado, pulando)")
        continue
    
    size_mb = os.path.getsize(full_path) / 1024 / 1024
    print(f"  ⬆️  {local_path} ({size_mb:.1f} MB) → {remote_path}...", end=" ", flush=True)
    
    try:
        api.upload_file(
            path_or_fileobj=full_path,
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        erros += 1

print("=" * 55)
if erros == 0:
    print("🎉 Todos os artefatos foram enviados com sucesso!")
else:
    print(f"⚠️  {erros} erro(s) encontrados durante o upload.")
