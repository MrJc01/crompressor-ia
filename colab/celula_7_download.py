# Download automático do arquivo GGUF
from google.colab import files
import glob

gguf_files = glob.glob("crom_dna_lora_gguf/*.gguf")
if gguf_files:
    for f in gguf_files:
        size_mb = __import__('os').path.getsize(f) / 1024 / 1024
        print(f"⬇️  Baixando: {f} ({size_mb:.1f} MB)")
        files.download(f)
    print("\n✅ Download iniciado! Verifique sua pasta de Downloads.")
else:
    print("❌ Nenhum .gguf encontrado. Execute a CÉLULA 6 primeiro!")
