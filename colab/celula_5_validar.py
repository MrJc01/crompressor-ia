from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

testes = [
    {
        "titulo": "TESTE 1: Codificação Humano → DNA",
        "prompt": "<|im_start|>system\nVocê é uma Célula CROM. Transcreva a resposta como Cadeia Quaternária DNA CROM Base-4 (usar apenas A, T, C, G).<|im_end|>\n<|im_start|>user\nO que é inteligência artificial?<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "titulo": "TESTE 2: Decodificação DNA → Humano",
        "prompt": "<|im_start|>system\nDecodifique a sequência biológica DNA CROM Base-4 para linguagem humana em Português.<|im_end|>\n<|im_start|>user\nTACGTACGCCGGATCT<|im_end|>\n<|im_start|>assistant\n",
    },
    {
        "titulo": "TESTE 3: Resposta Normal (teste de sanidade)",
        "prompt": "<|im_start|>system\nVocê é um assistente inteligente. Responda em Português.<|im_end|>\n<|im_start|>user\nO que é compressão de dados?<|im_end|>\n<|im_start|>assistant\n",
    },
]

print("=" * 55)
print(" 🧪 VALIDAÇÃO PÓS-TREINO LoRA CROM-DNA")
print("=" * 55)

for teste in testes:
    print(f"\n{'─' * 55}")
    print(f"  {teste['titulo']}")
    print(f"{'─' * 55}")

    inputs = tokenizer(teste["prompt"], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,
        do_sample=True,
        use_cache=True,
    )
    resposta = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"  Resposta: {resposta[:300]}")

    chars_dna = sum(1 for c in resposta if c in 'ATCG')
    total_alpha = sum(1 for c in resposta if c.isalpha())
    if total_alpha > 0:
        ratio = chars_dna / total_alpha * 100
        print(f"  📊 DNA ratio: {ratio:.0f}% ({chars_dna}/{total_alpha})")

print(f"\n{'=' * 55}")
print(" ✅ Validação concluída!")
print(" Se os resultados parecem OK, vá para CÉLULA 6 (export).")
print(f"{'=' * 55}")
print("\n📋 MANDE ESSE OUTPUT PARA O ASSISTENTE CROM-IA!")
