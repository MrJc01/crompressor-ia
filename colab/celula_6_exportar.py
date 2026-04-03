import os

# Exporta o modelo com LoRA fundido em formato GGUF Q4_K_M
# Este é o MESMO formato que o llama-cli do seu PC lê!
print("📦 Exportando modelo para GGUF Q4_K_M...")
print("   (Isso pode levar 2-5 minutos)")

model.save_pretrained_gguf(
    "crom_dna_lora_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)

print("\n✅ Exportação concluída! Arquivos gerados:")
for f in sorted(os.listdir("crom_dna_lora_gguf")):
    size_mb = os.path.getsize(f"crom_dna_lora_gguf/{f}") / 1024 / 1024
    print(f"   📄 {f} ({size_mb:.1f} MB)")

print("\n" + "=" * 55)
print(" 🎯 COMO LEVAR O ARQUIVO PARA SUA MÁQUINA:")
print("=" * 55)
print(" Opção 1: Clique direito no arquivo .gguf → Download")
print(" Opção 2: Execute a CÉLULA 7 para download automático")
print("")
print(" Depois coloque o arquivo em:")
print(" /home/j/Área de trabalho/crompressor-ia/models/")
print("=" * 55)
print("\n📋 MANDE ESSE OUTPUT PARA O ASSISTENTE CROM-IA!")
