from google.colab import drive
import os
import shutil

# ============================================================
# CROM-IA V4.3: BACKUP FINAL PARA GOOGLE DRIVE
# ============================================================

# 1. Montar o Drive (Irá pedir autorização via link/popup)
print("📂 Montando Google Drive...")
drive.mount('/content/drive')

# 2. Definir a Pasta de Destino no Drive
# Ele criará uma pasta chamada 'CROM-IA_V4.3' no seu MyDrive
drive_folder = "/content/drive/MyDrive/CROM-IA_V4.3"
os.makedirs(drive_folder, exist_ok=True)

# 3. Caminhos dos Arquivos Gerados (Padrão do Unsloth)
# Baseado nos logs de sucesso do Salto Cognitivo V4.3
source_dir = "/content/Qwen3.5-2B-DNA-V4.3_gguf"
gguf_file = "Qwen3.5-2B.Q4_K_M.gguf"
mmproj_file = "Qwen3.5-2B.BF16-mmproj.gguf"

files_to_copy = [gguf_file, mmproj_file]

# 4. Execução do Backup SRE
print(f"\n📦 Iniciando backup para: {drive_folder}")

for filename in files_to_copy:
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(drive_folder, filename)
    
    if os.path.exists(src_path):
        print(f"🚀 Copiando {filename}...")
        shutil.copy(src_path, dst_path)
        print(f"✅ {filename} salvo com sucesso!")
    else:
        print(f"⚠️ Aviso: {filename} não encontrado em {source_dir}. Pulando...")

print("\n🏁 PROCESSO FINALIZADO. O Chassis V4.3 está seguro no seu Google Drive!")
print(f"Caminho no Drive: My Drive > CROM-IA_V4.3")
