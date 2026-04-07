# 🧠 CROM-IA V4.3: Guia de Continuação
**Data:** 07 de Abril de 2026  
**Status:** ⏸️ PAUSADO — Checkpoint para retomada futura

---

## 📍 1. ONDE PARAMOS

### Estado do Treinamento
- **Modelo treinado:** `CROM-IA_v4.3_Qwen3.5-2B.gguf` (1.27 GB, Q4_K_M)
- **Base:** Qwen/Qwen3.5-2B (Apache 2.0, Hybrid Gated DeltaNet + MoE)
- **Loss final:** 13.28 (iniciou em 41.31, convergência em 200 steps)
- **Dataset:** `dataset_dna_v43_test.jsonl` (1000 amostras trifásicas Radix-4)
- **Hardware treino:** NVIDIA A100-SXM4-40GB no Colab

### O que FUNCIONA
- ✅ Motor nativo C++ (`llama-cli`) com template ChatML
- ✅ Codebooks 1x3 e 1x5 (fixo + dinâmico + expandido)
- ✅ Decoder streaming (`scripts/dna_decoder.py`)
- ✅ DNA Atoms protocol (⌬W_, ⌬F_, ⌬P_)
- ✅ Pipeline Colab completo (7 células)
- ✅ Visualizador SRE web (migrado para ChatML)
- ✅ Ponte mmap 64MB para motor Go
- ✅ RAG BM25 nativo

### O que NÃO FUNCIONA (Gaps a resolver)
- ❌ **Modelo NÃO emite ponteiros DNA reais (@@XX) em produção**
- ❌ Loop completo Texto → DNA → Modelo → DNA → Texto não fecha
- ❌ Overhead em tokens curtos ("a", "oi") sem resolver no treinamento
- ❌ Multi-Brain Engine (V4.0 conceitual) não ativado
- ❌ GRPO / RFT — proposto em `ARCHITECTURE_PROPOSAL.md`, não implementado
- ❌ Self-Distillation (SSD) — protótipo existe, não executado
- ❌ SCONE embeddings — apenas benchmarked, não integrado

---

## 🔧 2. PARA CONTINUAR (Checklist de Retomada)

### 2.1 Reativar o Ambiente
```bash
# 1. Montar ponte de memória
python3 scripts/setup_mmap_bridge.py

# 2. Verificar modelo disponível
ls -lh models/CROM-IA_v4.3_Qwen3.5-2B.gguf

# 3. Teste rápido de sanidade
bash v4.3_cognitive_leap/chat_dna_v43_decode.sh
```

### 2.2 Próximos Passos Técnicos (Prioridade)

#### Prioridade 1: Fechar o Loop DNA
1. Melhorar `gerar_dataset_dna.py` para resolver overhead em strings curtas
2. Re-treinar no Colab com dataset maior (5000-10000 amostras)
3. Testar se o modelo começa a emitir @@XX espontaneamente
4. Integrar decode automático via `scripts/dna_decoder.py`

#### Prioridade 2: Qualidade de Inferência
1. Ajustar temperatura/penalties para output DNA (T=0.1, rep_penalty=1.1)
2. Testar gramática GBNF (`pequenas_pesquisas/6_gramatica_estrutural_gbnf/haicai.gbnf`)
3. Implementar guardrail anti-loop (`pequenas_pesquisas/5_*/guardrail_loop_fixer.py`)

#### Prioridade 3: Escalabilidade
1. Testar Self-Distillation (`pequenas_pesquisas/8_*/self_distillation_v43.py`)
2. Expandir codebook dinamicamente com base nos misses
3. Avaliar migração para modelo 3B+ (Llama-4-Scout ou Qwen3.5-3B)

### 2.3 Propostas Arquiteturais Pendentes
- **GRPO:** Treino de raciocínio sem reward model (múltiplos candidatos DNA)
- **SCONE:** Desacoplamento embed in/out para ponteiro ⌬ sem inflar dicionário
- **ALM:** Adaptive LoRA Merging — fusão empírica Medicina + Código + PT-BR

---

## 📚 3. ARQUIVOS CHAVE PARA REFERÊNCIA

| Recurso | Path |
|---------|------|
| Manifesto Arquitetural | `v4.3_cognitive_leap/ARCHITECTURE_PROPOSAL.md` |
| Diagnóstico V4.2 | `v4.3_cognitive_leap/ROOT_DIAGNOSTICS.md` |
| Log de Treinamento | `v4.3_cognitive_leap/03_TREINAMENTO_QWEN3.5_LOG.md` |
| Guia de Implantação | `v4.3_cognitive_leap/04_INTEGRATION_GUIDE.md` |
| Protocolo DNA Atoms | `v4.3_cognitive_leap/DNA_ATOMS.md` |
| Terminal Python V4.3 | `v4.3_cognitive_leap/chat_terminal_v4.3.py` |
| Pipeline Colab | `colab/CROM_DNA_LoRA_Training.py` |
| Dataset V4.3 teste | `dataset_dna_v43_test.jsonl` |
| Pesquisas experimentais | `pequenas_pesquisas/` (7 tracks) |

---

## ⚠️ 4. LIÇÕES APRENDIDAS

1. **Modelos < 1B são inadequados** — 0.6B colapsa em tarefas complexas (fractais, interferência catastrófica)
2. **2B é o mínimo viável** — Qwen3.5-2B converge com Loss 13.28 após LoRA
3. **Template ChatML é obrigatório** — sem `<|im_start|>/<|im_end|>` o modelo perde coerência
4. **T=0.1 + rep_penalty=1.1** — configuração SRE estável para DNA
5. **BPE fragmenta DNA** — o tokenizer do Qwen corta "ATCGATCG" em tokens sem sentido
6. **Codebooks resolvem parcialmente** — greedy longest-match comprime, mas o modelo não aprendeu a usá-los

---

*Checkpoint gerado em 07/Abr/2026. Vide também `vPureDna/` para a vertente experimental de modelo do zero com DNA puro.*
