#!/usr/bin/env python3
"""
🧬 vPureDna — Script de Treino Colab (LoRA + Qwen + ⌬)

Roda no Google Colab com GPU T4/A100.
Treina Qwen3.5-2B a usar ⌬ como atalho de compressão inteligente.

Setup Colab:
  !pip install transformers peft datasets trl accelerate bitsandbytes
  !git clone https://github.com/MrJc01/crompressor-ia.git
  %cd crompressor-ia
  !python3 vPureDna/05_colab/treino_colab_vpuredna.py
"""

import os
import sys
import json
import torch
from datetime import datetime

# ======================================================
# CONFIGURAÇÃO (ajustar para cada run)
# ======================================================
CONFIG = {
    # Modelo base
    "model_name": "Qwen/Qwen3-1.7B",
    "load_in_4bit": True,
    
    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    
    # Dataset
    "dataset_path": None,  # None = gerar na hora com Alpaca-PT
    "max_amostras": 5000,
    "max_chars_per_sample": 600,
    "val_split": 0.1,
    
    # Treino
    "max_steps": 1000,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 50,
    "max_seq_length": 512,
    "fp16": True,
    
    # Output
    "output_dir": "./vPureDna/05_colab/output",
    "save_steps": 200,
    "eval_steps": 100,
    "logging_steps": 25,
}


def setup_environment():
    """Verifica ambiente e instala dependências."""
    print("=" * 60)
    print(" 🧬 vPureDna — TREINO COLAB (LoRA + Qwen + ⌬)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠️  SEM GPU! Treino será muito lento.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    print(f"  Device: {device}")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  LoRA r={CONFIG['lora_r']} alpha={CONFIG['lora_alpha']}")
    print(f"  Steps: {CONFIG['max_steps']}")
    print()
    return device


def gerar_dataset_colab():
    """Gera dataset trifásico com Alpaca-PT + ⌬ compressor."""
    print("[1] Gerando dataset ⌬ para Colab...")
    
    # Importar compressor
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_encoder'))
    from tokenizer_dna import DNACompressor
    
    comp = DNACompressor()
    print(f"  ⌬W: {comp.total_w} | ⌬F: {comp.total_f}")
    
    # Carregar Alpaca-PT do HuggingFace
    from datasets import load_dataset
    print("  Baixando Alpaca-PT do HuggingFace...")
    ds = load_dataset("FreedomIntelligence/alpaca-gpt4-portuguese", split="train")
    print(f"  ✅ {len(ds)} amostras carregadas")
    
    import random
    random.seed(42)
    
    dados = []
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    max_a = CONFIG["max_amostras"]
    terco = max_a // 3
    
    for idx in indices:
        if len(dados) >= max_a:
            break
        
        item = ds[idx]
        
        # Extrair texto (formato conversations)
        instrucao = saida = ""
        if 'conversations' in item:
            for c in item['conversations']:
                if c.get("from") == "human" and not instrucao:
                    instrucao = str(c.get("value", "")).strip()
                elif c.get("from") == "gpt" and not saida:
                    saida = str(c.get("value", "")).strip()
        else:
            instrucao = str(item.get('instruction', '') or '')
            saida = str(item.get('output', '') or '')
        
        if not instrucao or not saida:
            continue
        if len(instrucao) + len(saida) > CONFIG["max_chars_per_sample"]:
            continue
        
        # Comprimir resposta
        resp_dna = comp.compress(saida)
        fase = len(dados)
        
        if fase < terco:
            # Fase A: Emitir ⌬ na resposta
            dados.append({
                "messages": [
                    {"role": "system", "content": "Você é CROM-IA. Responda usando marcadores ⌬ para comprimir frases conhecidas. Ex: ⌬F42 = frase comprimida."},
                    {"role": "user", "content": instrucao},
                    {"role": "assistant", "content": resp_dna},
                ]
            })
        elif fase < 2 * terco:
            # Fase B: Expandir ⌬
            dados.append({
                "messages": [
                    {"role": "system", "content": "Você é CROM-IA. Expanda todos os marcadores ⌬ para texto completo em Português."},
                    {"role": "user", "content": resp_dna},
                    {"role": "assistant", "content": saida.lower()},
                ]
            })
        else:
            # Fase C: Responder normal (manter capacidade base)
            dados.append({
                "messages": [
                    {"role": "system", "content": "Você é CROM-IA. Responda em Português claro e completo."},
                    {"role": "user", "content": instrucao},
                    {"role": "assistant", "content": saida},
                ]
            })
        
        if len(dados) % 500 == 0:
            print(f"    ... {len(dados)}/{max_a}")
    
    # Salvar
    ds_path = os.path.join(CONFIG["output_dir"], "dataset_vpuredna_colab.jsonl")
    os.makedirs(os.path.dirname(ds_path), exist_ok=True)
    with open(ds_path, 'w', encoding='utf-8') as f:
        for d in dados:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print(f"  ✅ {len(dados)} amostras → {ds_path}")
    print(f"  Fases: A={terco} | B={terco} | C={len(dados)-2*terco}")
    return ds_path


def carregar_modelo():
    """Carrega Qwen + configura LoRA + adiciona ⌬ como special token."""
    print("\n[2] Carregando modelo...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Quantização 4-bit
    bnb_config = None
    if CONFIG["load_in_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    
    # >>> CRUCIAL: Adicionar ⌬ como special token <<<
    num_new = tokenizer.add_special_tokens({"additional_special_tokens": ["⌬"]})
    print(f"  ⌬ adicionado como special token (ID={tokenizer.convert_tokens_to_ids('⌬')})")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Carregar modelo
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Resize embeddings para o novo token ⌬
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Embeddings redimensionadas: {len(tokenizer)} tokens")
    
    # Preparar para LoRA
    if CONFIG["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)
    
    # Configurar LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable/1e6:.1f}M treináveis / {total/1e6:.0f}M total ({trainable/total*100:.1f}%)")
    
    return model, tokenizer


def treinar(model, tokenizer, dataset_path):
    """Treina com SFTTrainer."""
    print("\n[3] Treinando...")
    
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset as ld
    
    # Carregar dataset
    ds = ld("json", data_files=dataset_path, split="train")
    ds = ds.train_test_split(test_size=CONFIG["val_split"], seed=42)
    
    print(f"  Train: {len(ds['train'])} | Val: {len(ds['test'])}")
    
    # Configuração de treino
    training_args = SFTConfig(
        output_dir=CONFIG["output_dir"],
        max_steps=CONFIG["max_steps"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        fp16=CONFIG["fp16"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_seq_length=CONFIG["max_seq_length"],
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # Treinar
    result = trainer.train()
    
    # Salvar final
    trainer.save_model(os.path.join(CONFIG["output_dir"], "final"))
    tokenizer.save_pretrained(os.path.join(CONFIG["output_dir"], "final"))
    
    print(f"\n  ✅ Treino completo!")
    print(f"  Loss final: {result.training_loss:.4f}")
    print(f"  Salvo em: {CONFIG['output_dir']}/final")
    
    return result


def validar_rapido(model, tokenizer):
    """Validação rápida: o modelo emite ⌬?"""
    print("\n[4] Validação rápida...")
    
    prompts = [
        {
            "system": "Você é CROM-IA. Responda usando marcadores ⌬ para comprimir frases conhecidas.",
            "user": "O que é inteligência artificial?",
            "esperado": "⌬",
        },
        {
            "system": "Você é CROM-IA. Expanda todos os marcadores ⌬ para texto completo em Português.",
            "user": "⌬F186 é o futuro.",
            "esperado": "inteligência",
        },
        {
            "system": "Você é CROM-IA. Responda em Português claro e completo.",
            "user": "Qual é a capital do Brasil?",
            "esperado": "Brasília",
        },
    ]
    
    model.eval()
    acertos = 0
    
    for p in prompts:
        messages = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        tem_esperado = p["esperado"].lower() in response.lower()
        
        print(f"\n  [{p['system'][:50]}...]")
        print(f"  User: {p['user']}")
        print(f"  Bot:  {response[:100]}")
        print(f"  Esperado '{p['esperado']}': {'✅' if tem_esperado else '❌'}")
        
        if tem_esperado:
            acertos += 1
    
    print(f"\n  Score: {acertos}/{len(prompts)}")
    return acertos


def main():
    device = setup_environment()
    
    # 1. Dataset
    if CONFIG["dataset_path"] and os.path.exists(CONFIG["dataset_path"]):
        ds_path = CONFIG["dataset_path"]
        print(f"[1] Usando dataset existente: {ds_path}")
    else:
        ds_path = gerar_dataset_colab()
    
    # 2. Modelo
    model, tokenizer = carregar_modelo()
    
    # 3. Treino
    result = treinar(model, tokenizer, ds_path)
    
    # 4. Validação
    score = validar_rapido(model, tokenizer)
    
    # 5. Relatório
    print(f"\n{'=' * 60}")
    print(f" 📊 RELATÓRIO FINAL vPureDna Colab")
    print(f"{'=' * 60}")
    print(f"  Model:      {CONFIG['model_name']}")
    print(f"  LoRA:       r={CONFIG['lora_r']} alpha={CONFIG['lora_alpha']}")
    print(f"  Dataset:    {CONFIG['max_amostras']} amostras trifásicas")
    print(f"  Steps:      {CONFIG['max_steps']}")
    print(f"  Loss final: {result.training_loss:.4f}")
    print(f"  Validação:  {score}/3 acertos")
    print(f"  Output:     {CONFIG['output_dir']}/final")
    print(f"  Timestamp:  {datetime.now().isoformat()}")
    print(f"{'=' * 60}")
    
    if score >= 2:
        print(f"\n  🔥 MODELO APROVADO! Exportar para GGUF e testar localmente.")
    else:
        print(f"\n  ⚠️ Modelo precisa mais treino. Aumentar dataset/steps.")


if __name__ == "__main__":
    main()
