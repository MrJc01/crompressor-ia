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
REPO_ID = "CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic" # Criando repositório novo para a V3.5
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Aviso ao criar repo (provavelmente já existe): {e}")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artefatos para subir
ARQUIVOS = [
    # Modelo GGUF (V3.5b - 117k)
    ("models/Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf", "Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf"),
    # Codebook V3 (Com o bug de bicarbonato/piadas descoberto)
    ("v3_engine/macro_codebook_v3.json", "logs_and_codebooks/bugged_codebook_v3_recipes.json"),
    ("docs/CROM_IA_DIARY_V35b.md", "README.md"), # O diário será o Model Card!
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
