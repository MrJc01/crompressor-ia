from unsloth import FastLanguageModel
import torch

# Carrega o Qwen 2.5 0.5B na GPU em 4-bit (mesmo modelo do seu PC!)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# Injeta adaptadores LoRA nos módulos de atenção
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("✅ Modelo Qwen 2.5 0.5B carregado e LoRA ativado!")
model.print_trainable_parameters()
