from datasets import load_dataset

# Carrega o seu dataset DNA gerado localmente
# IMPORTANTE: Faça upload do dataset_dna.jsonl antes!
dataset = load_dataset("json", data_files="dataset_dna.jsonl", split="train")

print(f"✅ Dataset carregado: {len(dataset)} amostras")
print(f"   Colunas: {dataset.column_names}")
print(f"\n📋 Exemplo FASE A (Humano → DNA):")
print(f"   Instrução: {dataset[0]['instruction'][:80]}...")
print(f"   Input: {dataset[0]['input'][:80]}...")
print(f"   Output: {dataset[0]['output'][:80]}...")

# Formatar prompts no template ChatML do Qwen 2.5
def formatar_prompt(exemplos):
    textos = []
    for inst, inp, out in zip(exemplos["instruction"], exemplos["input"], exemplos["output"]):
        prompt = (
            "<|im_start|>system\n"
            f"{inst}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{inp}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{out}<|im_end|>"
        )
        textos.append(prompt)
    return {"text": textos}

dataset = dataset.map(formatar_prompt, batched=True)

print(f"\n✅ Dataset formatado no template ChatML do Qwen!")
print(f"   Preview do primeiro exemplo formatado:")
print(dataset[0]["text"][:300] + "...")
