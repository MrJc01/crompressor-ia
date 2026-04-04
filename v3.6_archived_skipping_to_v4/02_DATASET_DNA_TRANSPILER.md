# 02. Dataset DNA Transpiler (V3.6)

## 📌 O Desafio do Paradigma
A nossa LLM (Qwen2.5) e qualquer LLaMA atual aprende a reproduzir dados estatísticos de acordo com o dataset de formatação Base ou de Instrução (Alpaca/ShareGPT). 
Para a IA passar a ejetar os ponteiros DNA que nós montamos na Fase 01, o **Dataset Matemático de Treinamento** tem que falar nesse idioma de forma crua, brutal, e em abundância.

## 🛠️ Método Numérico
Vamos precisar construir o **Transpilador de Dataset**.
Ele funcionará cruzando: `Dataset Alpaca Original` ❌ `macro_codebook_v4.json`.

**Fluxo de Operação:**
1. Lemos todas as amostras "Output" (Resposta da IA) do JSONL original.
2. Usamos o Codebook e ordenamos as *Macros* de forma **Decrescente** por Tamanho do Texto (`len(text)`). (Isso evita que marcadores curtos como "def " quebrem marcadores enormes que tenham partes similares "def __init__").
3. Para cada amostra, fazemos substituições literais (`str.replace`).
4. Salvamos o resultado no arquivo **`dataset_dna_mutante.jsonl`**.

### ⚙️ Exemplo/Snippet

*De (Vanilla Alpaca Target):*
```json
{
  "instruction": "Como crio uma classe em Python?",
  "output": "Aqui está um exemplo:\nclass Carro:\n    def __init__(self):\n        super().__init__()\n        self.rodas = 4\n"
}
```

*Para (DNA Mutante Transpilated Dataset):*
```json
{
  "instruction": "Como crio uma classe em Python?",
  "output": "Aqui está um exemplo:\nclass Carro:\n    @@AATA        self.rodas = 4\n"
}
```

> **Alerta Arquitetural:** As partes do *"instruction"* NUNCA devem ser tocadas. O LLM apenas precisa aprender a *cuspir* o pointer, já que ele o faz em seu gerador de Output. Se corrompermos o seu input, ele se perderá.
