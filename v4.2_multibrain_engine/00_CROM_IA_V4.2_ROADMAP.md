# CROM-IA V4.2 — Roadmap de Melhorias
> **Status:** Planejamento
> **Base Model:** Qwen3.5-0.8B (https://huggingface.co/Qwen/Qwen3.5-0.8B)
> **Pré-requisito:** V4.1 concluída e validada

---

## Upgrade de Modelo: Qwen3.5-0.8B

### Por que trocar do Qwen3-0.6B?
| Aspecto | Qwen3-0.6B (V4.1) | Qwen3.5-0.8B (V4.2) |
|---|---|---|
| Parâmetros | 636M | ~800M (+25%) |
| Arquitetura | Qwen3 | Qwen3.5 (mais moderna) |
| PT-BR nativo | Bom | Melhor (mais dados de treino) |
| Velocidade i5 | ~8-10 t/s | ~6-8 t/s (ainda rápido) |
| Raciocínio | Básico | Melhorado (3.5 foca em reasoning) |

### Ação necessária:
Verificar se Unsloth disponibiliza `unsloth/Qwen3.5-0.8B-bnb-4bit`.
Se não, usar quantização manual via `BitsAndBytesConfig`.

---

## Melhorias Planejadas V4.2

### 1. DPO (Direct Preference Optimization)
**Problema V4.1:** O modelo sabe DNA mas nem sempre PREFERE usá-lo.
**Solução:** Treinar com pares (resposta_com_DNA=preferred, resposta_sem_DNA=rejected).
O modelo aprende que DNA é a resposta "correta" por preferência, não só por frequência.

```python
# Formato DPO
{
  "prompt": "Explique arritmia cardíaca",
  "chosen": "Uma @@DGN de @@CRC onde...",   # DNA = preferido
  "rejected": "Uma diagnóstico de coração..."  # Normal = rejeitado
}
```

### 2. LoRA → GGUF Adapter (Empilhamento Real)
**Problema V4.1:** Ainda salvamos GGUF fundido (monolítico).
**Solução:** Converter adaptadores PEFT para formato GGUF-LoRA usando:
```bash
python3 llama.cpp/convert_lora_to_gguf.py \
  --base qwen3.5-0.8b.gguf \
  --adapter adapter_Python_DNA/ \
  --outfile Python_DNA_lora.gguf
```
Resultado: `--lora A.gguf --lora B.gguf` na inferência.

### 3. Multi-Turn Conversation Training
**Problema V4.1:** Treinamos apenas single-turn (1 pergunta → 1 resposta).
**Solução:** Criar datasets com histórico de conversa:
```
<|im_start|>user
O que é Python?<|im_end|>
<|im_start|>assistant
@@DEF é uma linguagem...<|im_end|>
<|im_start|>user
Mostre um exemplo<|im_end|>
<|im_start|>assistant
@@IMP math\n@@PRT(math.sqrt(16))<|im_end|>
```

### 4. RAG + DNA Decoder Integrado
**Problema V4.1:** O decodificador DNA existe mas não está integrado no chat.
**Solução:** Pipeline completo:
```
Pergunta → Modelo (gera DNA) → Decodificador (traduz @@) → Usuário vê texto limpo
```
O usuário nunca vê os tokens @@, mas o modelo gera menos tokens = mais rápido.

### 5. Validação Automática Pós-Treino
**Problema V4.1:** Não temos métricas de qualidade automatizadas.
**Solução:** Script que após treino:
- Roda 50 perguntas padrão
- Mede % de tokens DNA na saída
- Calcula BLEU vs resposta esperada
- Mede velocidade de inferência
- Gera relatório comparativo automático

### 6. Datasets Estratégicos (Qualidade > Quantidade)

> [!IMPORTANT]
> A filosofia é: **poucos dados de alta qualidade** vencem toneladas de dados genéricos.
> Modelo de 0.8B não absorve 1M de amostras — ele precisa de ~20-50K muito bem escolhidas.

#### Estratégia de Dados V4.2: 3 Camadas

**Camada 1 — Base Conversacional (PT-BR nativo, ~30K amostras)**
| Dataset | Amostras | Por quê? | Prioridade |
|---|---|---|---|
| `dominguesm/Canarim-Instruct-PTBR` | Pegar 30K | JÁ em PT-BR, sem tradução, qualidade GPT-3.5 | 🔴 ALTA |
| `dominguesm/alpaca-data-pt-br` | 52K | Alpaca oficial em PT-BR, bom para base geral | 🟡 MÉDIA |

**Camada 2 — Destilação Inteligente (Traduzir EN→PT-BR, ~15K amostras)**
| Dataset Original (EN) | Pegar | Por quê? | Prioridade |
|---|---|---|---|
| `teknium/OpenHermes-2.5` | 10K melhores | Destilado do GPT-4, o melhor que existe | 🔴 ALTA |
| `Open-Orca/SlimOrca` | 5K filtrados | GPT-4 filtrado, instruções complexas | 🟡 MÉDIA |

> [!TIP]  
> Não traduzir 1M de amostras! Filtrar as 10-15K com respostas mais longas e 
> complexas. O modelo aprende mais com 10K respostas profundas do GPT-4 do que 
> com 100K respostas rasas.

**Camada 3 — Especialização por Domínio (Nossos dados + HuggingFace, ~10K por cérebro)**
| Cérebro | Fonte | Amostras |
|---|---|---|
| Python | `Vezora/Tested-22k-Python-Alpaca` + nosso dataset | 10K |
| Medicina | Artigos SciELO + nosso dataset | 10K |
| CROM Self | Docs do projeto (já temos) | 500 |
| Conversa | Canarim filtrado para diálogo | 5K |

#### Resultado Final: ~60K amostras totais
- Peso do dataset: ~50-80MB (JSONL comprimido)
- Tempo de treino: ~1h no Colab A100
- Qualidade: Respostas nível GPT-4 em modelo de 500MB

#### Tradução Automática (Offline, Gratuita)
```bash
pip install argostranslate
python3 -c "
import argostranslate.translate
# Baixa pacote en→pt uma vez (~50MB)
argostranslate.package.update_package_index()
pkg = [p for p in argostranslate.package.get_available_packages() 
       if p.from_code=='en' and p.to_code=='pt'][0]
pkg.install()
# Traduzir
t = argostranslate.translate.get_translation_from_codes('en','pt')
print(t.translate('Hello, how are you?'))  # → 'Olá, como vai?'
"
```

#### Filtro de Qualidade (O que torna isso MELHOR que modelos tradicionais)
Antes de treinar, cada amostra passa por 3 filtros:
1. **Tamanho mínimo:** Resposta > 100 chars (eliminar respostas rasas)
2. **Diversidade:** Remover duplicatas por similaridade cosine
3. **Complexidade:** Priorizar respostas com código, listas, explicações técnicas

### 8. Scheduler de Learning Rate

**Melhoria:** Ao invés de LR constante, usar cosine annealing:
```python
lr_scheduler_type = "cosine",
warmup_ratio = 0.05,
```
Isso ajuda o modelo a convergir melhor nos últimos steps.

### 8. Flash Attention 2
**Melhoria:** Instalar FA2 no Colab para treino 2x mais rápido:
```bash
pip install flash-attn --no-build-isolation
```

---

## Checklist de Execução V4.2

- [ ] Validar V4.1 no i5 local (testar os 4 cérebros)
- [ ] Verificar disponibilidade Qwen3.5-0.8B no Unsloth
- [ ] Criar gerador de pares DPO automático
- [ ] Implementar convert_lora_to_gguf no pipeline
- [ ] Criar datasets multi-turn
- [ ] Integrar decodificador DNA no chat_v42.sh
- [ ] Implementar benchmark automático pós-treino
- [ ] Coletar datasets maiores (GitHub, SciELO)
- [ ] Testar cosine annealing LR
- [ ] Benchmark comparativo V4.1 vs V4.2
