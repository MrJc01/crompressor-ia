# CROM-IA V4.1 — Handoff para Próxima Sessão

## Contexto Rápido
O CROM-IA é um projeto de IA local brasileira que usa compressão DNA (tokens @@) para
acelerar inferência em hardware limitado (i5-3320M, 7.4GB RAM, sem GPU).

## O que foi feito NESTA sessão:
1. Treinou 4 micro-cérebros no Colab A100 (Qwen3-0.6B, LoRA rank 64, 2000 steps)
2. DNA mutação 75% com codebooks data-driven por frequência real
3. Todos os modelos apresentaram catastrophic forgetting (respostas incoerentes)
4. Causa raiz: treino agressivo demais (75% DNA, 2000 steps, rank 64) num modelo pequeno (0.6B)

## O que funciona:
- Pipeline completo: extração → codebook → transpilação → treino → deploy local
- Velocidade: 7-9 t/s no i5 (2x melhor que V4.0)
- RAM: 635MB (metade da V4.0)
- Codebook por frequência real (filosofia Crompressor)
- Script `adicionar_cerebro.py` — adiciona cérebro em 1 comando
- DNA tokens são gerados (`@@PWAT`, `@@PWC`) mas modelo esquece como conversar

## O que fazer NA PRÓXIMA sessão:

### 1. Baixar datasets REAIS do HuggingFace
```python
from datasets import load_dataset
# Base conversacional (30K PT-BR)
ds = load_dataset('dominguesm/Canarim-Instruct-PTBR-Dataset', split='train')
ds.select(range(30000)).to_json('canarim_30k.jsonl')
# Python (22K testado)
ds = load_dataset('Vezora/Tested-22k-Python-Alpaca', split='train')
ds.select(range(15000)).to_json('python_15k.jsonl')
```

### 2. Traduzir OpenHermes-2.5 (top 10K) com Argos Translate
```bash
pip install argostranslate
# Traduzir en→pt offline
```

### 3. Re-transpilar com DNA a 25% (não 75%!)
```bash
python3 transpilador_v41.py canarim_30k.jsonl canarim_DNA25.jsonl codebook.json 0.25
```

### 4. Treinar com parâmetros CONSERVADORES
```python
r = 16          # Era 64
max_steps = 800 # Era 2000
learning_rate = 1e-5  # Era 2e-5
# SEM gate_proj, down_proj, up_proj nos target_modules!
```

### 5. Usar Qwen3.5-0.8B (não 0.6B)
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- Verificar: `unsloth/Qwen3.5-0.8B-bnb-4bit`

### 6. Testar LoRA stacking real
- Converter adaptadores PEFT → GGUF-LoRA
- `llama-cli -m base.gguf --lora A.gguf --lora B.gguf`

## Arquivos importantes:
- Documentação: `v4.1_multibrain_engine/00_CROM_IA_V4.1_DOCUMENTATION.md`
- Roadmap V4.2: `v4.2_multibrain_engine/00_CROM_IA_V4.2_ROADMAP.md`
- Transpilador: `v4.1_multibrain_engine/1_extracao_local/transpilador_v41.py`
- Codebook gen: `v4.1_multibrain_engine/1_extracao_local/gerador_codebook_v41.py`
- Add brain: `v4.1_multibrain_engine/adicionar_cerebro.py`
- Chat script: `v4_multibrain_engine/3_inferencia_local/chat_v4_multibrain.sh`
- Modelos GGUF: `models/` (V4.0: 941MB, V4.1-alpha: 379MB)
- llama-cli: `pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli`
- CPU: Intel i5-3320M @ 2.60GHz, 4 threads, 7.4GB RAM
