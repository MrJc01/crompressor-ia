# 03. LoRA DNA Training Protocol (V3.6)

## 📌 O Cérebro Plástico do LLM
Ensinar uma nova sintaxe de Token para uma rede Neural de 1.5 Bilhões de parâmetros que já pré-treinou o idioma humano é um processo delicado. Chamamos esse evento de *Catastrophic Forgetting* (quando a IA adota nossa linguagem alienígena `@@XX` tão fortemente que desaprende como conjugar verbos normais).
Para injetarmos o RAG no peso tensorial perfeitamente, os Hiperparâmetros de Fine-Tuning devem atuar como **Bisturis**.

## 🛠️ Método Numérico & Unsloth/PEFT
Iremos aplicar um Fine-Tuning (SFT - Supervised Fine Tuning), congelando os blocos fundamentais (Frozen Layers) via QLoRA 4-bit e alocando adaptadores A e B para "puxarem" com força a responsabilidade pela Formatação Especial de Strings.

### Hiperparâmetros Cruciais (Checklist)

1. **LoRA Rank (R): `32` ou `64`**
   - *Por que:* Ranks normais de conversação operam em `r=8` ou `r=16`. Entretanto, aprender uma sintaxe `Radix-4 @@AT` no vazio exige mais matrizes para codificação linguística. Recomendamos 32 no mínimo.

2. **LoRA Alpha: `2x Rank (ex: 64)`**
   - *Por que:* Amplifica os gradientes do Rank, fazendo a nova linguagem "grudar" na memória da IA mais rapidamente durante as primeiras épocas.

3. **Learning Rate (LR): `2e-5` a `3e-5`**
   - *Por que:* Treinamento conservador de precisão, suportado pelo Scheduler `cosine`, para evitar over-fitting de quebra nas outras palavras.

4. **Target Modules:** LLaMA base padrão
   - Focar nas portas que dominam Attentions: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

## ⚙️ Exemplo/Snippet de Setup (Unsloth Base)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)
```

## Critério de Sucesso Pós-Treino
Ao interagir com o GGUF compilado final:
Se o usuário digitar "crie um arquivo python básico", o modelo deverá responder em texto claro e usar o pointer "@@AT" no meio da frase com naturalidade O(1).
