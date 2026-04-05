# CROM-IA V4.2 — Guia DPO (Direct Preference Optimization)

## O que é DPO?

DPO é uma técnica de alinhamento que ensina o modelo a **preferir** um tipo de resposta sobre outro, sem precisar de um modelo de recompensa separado (mais simples que RLHF).

### Por que DPO no CROM-IA?

Na V4.1, o modelo aprendeu DNA por SFT (Supervised Fine-Tuning) puro:
- Viu exemplos com @@tokens e tentou imitá-los
- Resultado: usou DNA mas também **esqueceu como conversar** (catastrophic forgetting)

Com DPO, a abordagem é diferente:
- O modelo recebe **pares**: resposta com DNA (preferida) vs sem DNA (rejeitada)
- Ele aprende que DNA é **a escolha melhor**, não uma obrigação
- Resultado: usa DNA quando faz sentido, mantém fluência

---

## Formato dos Dados DPO — 3 Níveis Hierárquicos

O DNA do Crompressor opera em **3 níveis**, e os pares DPO refletem isso:

### Nível W (Word) — Palavra isolada → 1 token DNA
```json
{
  "prompt": "O que é arritmia?",
  "chosen": "Uma @@DGN de @@CRC onde o ritmo cardíaco é irregular.",
  "rejected": "Um diagnóstico de coração onde o ritmo cardíaco é irregular."
}
```
> `diagnóstico` (12 chars) → `@@DGN` (5 chars) = **7 bytes economizados** por hit

### Nível F (Phrase) — Frase repetida → 1 token DNA
```json
{
  "prompt": "Como tratar hipertensão?",
  "chosen": "@@MFG. O @@TRT inclui mudanças no estilo de vida e @@MED anti-hipertensivos.",
  "rejected": "É importante consultar um médico especialista. O tratamento inclui mudanças no estilo de vida e medicamentos anti-hipertensivos."
}
```
> `É importante consultar um médico especialista` (46 chars) → `@@MFG` (5 chars) = **41 bytes** por hit!

### Nível P (Paragraph) — Bloco inteiro repetido → 1 token DNA
```json
{
  "prompt": "Dê um exemplo de função Python",
  "chosen": "@@PPA\n\ndef somar(a, b):\n    return a + b",
  "rejected": "Para criar uma função em Python, use a palavra-chave def seguida do nome da função e parâmetros entre parênteses. Veja o exemplo:\n\ndef somar(a, b):\n    return a + b"
}
```
> `Para criar uma função em Python, use a palavra-chave def seguida...` (130 chars) → `@@PPA` (5 chars) = **125 bytes** por hit!

### Exemplo Misto (3 Níveis no mesmo par)
```json
{
  "prompt": "Explique como fazer um loop for em Python",
  "chosen": "@@PFC\n\n@@FOR item @@GFT:\n    @@PRT(item)\n\n@@PFD",
  "rejected": "O loop for em Python permite iterar sobre qualquer sequência como listas, tuplas ou strings.\n\nfor item in uma lista ou sequência iterável:\n    print(item)\n\nIsso percorre cada elemento da sequência e executa o bloco de código para cada um."
}
```
> - `@@PFC` = parágrafo de introdução (~90 chars → 5) = **Nível P**
> - `@@FOR` = palavra `for` = **Nível W**
> - `@@GFT` = frase `in uma lista ou sequência iterável` (37 → 5) = **Nível F**
> - `@@PFD` = parágrafo de conclusão (~80 chars → 5) = **Nível P**

**Regras:**
- `chosen` e `rejected` devem ser **semanticamente idênticos** (mesma informação)
- A ÚNICA diferença é a presença de tokens @@DNA no `chosen`
- Tokens de **todos os 3 níveis** (W, F, P) devem aparecer nos pares
- DNA a ~25-30% do texto no `chosen` (sutil, não agressivo)
- Parágrafos repetidos (nível P) são os mais valiosos em economia

---

## Pipeline de Geração de Pares DPO

### Fluxo Automático
```
Dataset original (ex: Canarim 30K)
        ↓
gerador_pares_dpo.py
        ↓
Para cada amostra:
  1. chosen = aplicar_mutacao_dna(resposta, codebook, taxa=0.25)
  2. rejected = resposta original (sem DNA)
  3. Emitir par {prompt, chosen, rejected}
        ↓
dataset_dpo_5k.jsonl
```

### Script: `gerador_pares_dpo.py`
```python
# Gera pares automaticamente a partir de dataset existente
# chosen = resposta com DNA aplicado (25%)
# rejected = resposta original (sem DNA)

python3 gerador_pares_dpo.py \
    --input canarim_30k.jsonl \
    --codebook codebook_geral.json \
    --output dataset_dpo_5k.jsonl \
    --max_pares 5000 \
    --taxa_dna 0.25
```

---

## Treino DPO no Colab

### Pré-requisito
O modelo **já deve ter passado pela Fase 1 (SFT Base)** e Fase 2 (SFT DNA).
DPO é o **refinamento final** — polir, não ensinar do zero.

### Código Colab (Fase 3)
```python
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from datasets import load_dataset

# Carregar modelo já treinado (Fase 1 + Fase 2)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="adapter_base_ptbr",  # Já fine-tuned
    max_seq_length=2048,
    load_in_4bit=True,
)

# LoRA para DPO (pode ser novo ou continuar do existente)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Dataset DPO
dataset = load_dataset("json", data_files="dataset_dpo_5k.jsonl", split="train")

# Treinar com DPO
training_args = DPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=300,
    learning_rate=5e-6,       # Mais suave que SFT
    beta=0.1,                  # Força da preferência
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    output_dir="./outputs_dpo",
    logging_steps=25,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Parâmetros DPO Explicados
| Parâmetro | Valor | Por quê? |
|---|---|---|
| `beta` | 0.1 | Controla quão forte é a preferência. 0.1 = sutil. 0.5 = agressivo. |
| `learning_rate` | 5e-6 | Metade do SFT. DPO é refinamento, não reeducação. |
| `max_steps` | 300 | Pouco. DPO converge rápido com bons pares. |
| `batch_size` | 4 | Menor que SFT (cada amostra tem 2 respostas = 2x memória). |

---

## Validação Pós-DPO

### Teste A/B
Rodar o modelo **antes e depois do DPO** com as mesmas 50 perguntas:
- Medir % de tokens @@DNA na saída
- Medir coerência (resposta faz sentido?)
- Medir fluência (PT-BR natural?)

### Resultado Esperado
- **Antes DPO:** Modelo conversa bem, usa DNA inconsistentemente
- **Depois DPO:** Modelo conversa bem E prefere usar DNA quando há codebook match

---

## Riscos e Mitigações

| Risco | Mitigação |
|---|---|
| DPO degrada fluência | `beta=0.1` (conservador) + poucos steps (300) |
| Pares DPO de baixa qualidade | Filtrar: chosen deve ter >3 tokens DNA, chosen e rejected devem ser >100 chars |
| Overfitting DPO | Early stopping + validação cada 100 steps |
| `trl` incompatível | Verificar `trl>=0.7.0` no Colab antes de treinar |
