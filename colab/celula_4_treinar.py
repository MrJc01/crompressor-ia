from trl import SFTTrainer
from transformers import TrainingArguments
import torch

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1500,
        warmup_steps=50,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        output_dir="crom_lora_output",
        save_strategy="steps",
        save_steps=500,
        optim="adamw_8bit",
        seed=42,
    ),
)

print("🚀 Iniciando treinamento LoRA CROM-DNA...")
print("   Observe o Loss caindo! Deve ir de ~2.5 para < 1.0")
print("")

stats = trainer.train()

print("\n" + "=" * 55)
print(" ✅ TREINAMENTO CONCLUÍDO!")
print("=" * 55)
print(f" ⏱️  Tempo total    : {stats.metrics['train_runtime']:.0f} segundos")
print(f" 📉 Loss final     : {stats.metrics['train_loss']:.4f}")
print(f" 🔄 Steps          : {stats.metrics['train_steps']}")
print(f" ⚡ Amostras/seg   : {stats.metrics['train_samples_per_second']:.1f}")
print("=" * 55)
print("\n📋 MANDE ESSE OUTPUT PARA O ASSISTENTE CROM-IA!")
