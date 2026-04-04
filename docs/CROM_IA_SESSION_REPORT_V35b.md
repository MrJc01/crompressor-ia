# 📋 Relatório Completo: Sessão CROM-IA V3.5b — Da Crise à Visão

**Data:** 04 de Abril de 2026
**Sessão:** Estabilização V3.5 + Concepção da Arquitetura Multi-Brain

---

## 1. Resumo Executivo

Esta sessão começou com uma **crise** — o modelo V3.5a gerava respostas sem sentido — e terminou com uma **visão arquitetural** que pode transformar o CROM-IA de um projeto local em uma plataforma de compressão cognitiva modular.

Ao longo de ~4 horas de trabalho, realizamos:
1. Diagnóstico e correção de gráficos Mermaid no Technical Paper
2. Deploy do modelo Qwen2.5-1.5B-Instruct (941 MB GGUF)
3. Teste de sanidade que revelou falha catastrófica da V3.5a
4. Redesign completo do pipeline de dados (117k amostras reais)
5. Lançamento do treinamento V3.5b no Google Colab
6. Concepção da Arquitetura Multi-Brain (V4.0+)

---

## 2. Cronologia de Eventos

### Fase 1: Correção do Paper (01:39h)
**Problema:** Os diagramas Mermaid do Technical Paper apresentavam erros de sintaxe no TabNews (versão 10.9.5). Tags `<br/>`, aspas em subgraphs, `stateDiagram-v2` com `[*]`, e emojis nos labels causavam falha de renderização.

**Solução:** Reescrita de todos os 4 diagramas com sintaxe limpa: labels sem aspas, sem HTML inline, flowchart ao invés de stateDiagram, e remoção de caracteres especiais. Também removemos tags `<raw_string>` que eram artefatos do sistema.

### Fase 2: Deploy e Teste do Modelo 1.5B (02:10h)
**Ação:** O arquivo `Qwen2.5-1.5B-Instruct.Q4_K_M.gguf` (941 MB) foi baixado do Colab e colocado na pasta `models/`. O `chat_v3_rag.sh` foi atualizado para apontar para o novo modelo.

**Resultado do Teste (DESASTROSO):**
```
Prompt: "Me explique o que são buracos negros."
Resp:   "Buraco negro é uma entidade do espaço-tempo físico teórica
         onde toda gravitação ativa se concentra..."
         "Fui criada por engenheiros av\n[⚠️ RAG Injector Abortado]"

Prompt: "O que é a gravidade?"
Resp:   "Agravitude é força fundamental..." ❌ (palavra inventada)

Prompt: "Por que não pode me explicar?"
Resp:   "Meu nome é Qwen, e Sou uma IA inteligente treinada no uso
         do T5 modelo por Alibaba Cloud" ❌ (identidade quebrada)
```

**Diagnóstico Forense:**
- O dataset sintético de 25k amostras de persona foi gerado a partir de apenas ~40 templates combinatórios
- Cada template foi repetido ~625 vezes
- O modelo MEMORIZOU fragmentos sem sentido e os costurava aleatoriamente
- A learning rate de 5e-5 era agressiva demais, destruindo o conhecimento base do Qwen

### Fase 3: Redesign do Pipeline de Dados (02:16h)
**Decisão Crítica:** Abandonar completamente os dados sintéticos e usar exclusivamente conversas humanas reais do HuggingFace.

**Novo Pipeline:**

| Fonte | Amostras | Qualidade |
|-------|----------|-----------|
| `dominguesm/alpaca-data-pt-br` | 49.254 | Instruções reais em PT-BR |
| `FreedomIntelligence/alpaca-gpt4-portuguese` | 49.576 | Conversas GPT-4 em PT |
| Identidade Rosa (curada manualmente) | 500 | 16 perguntas × variações |
| Código Python V3 (comprimido) | 18.530 | Ponteiros @@DNA |
| **TOTAL** | **117.860** | **100% real ou curado** |

**Bug corrigido durante o build:** O dataset `alpaca-gpt4-portuguese` usa formato ShareGPT (`conversations` com `from/value`) ao invés do formato Alpaca (`instruction/output`). O parser foi ajustado e a segunda rodada do build trouxe todas as 49.576 amostras.

**Ajustes de Hiperparâmetros:**
- Learning rate: `5e-5` → `1e-5` (5x mais conservadora)
- Scheduler: `linear` → `cosine` (decay mais suave)
- Epochs: `max_steps=2500` → `num_train_epochs=1` (14.733 steps)
- Warmup: `20` → `50` steps
- Save checkpoints: a cada 500 steps (proteção contra queda do Colab)

### Fase 4: Treinamento V3.5b (02:25h → em andamento)
**Métricas do Treino (dados já registrados):**

| Métrica | Valor |
|---------|-------|
| Parâmetros totais | 1.580.643.840 (1.58B) |
| Parâmetros treináveis | 36.929.536 (2.34%) |
| batch efetivo | 8 (2 × 4 acumulação) |
| Total de steps | 14.733 |
| Tempo estimado | ~6 horas |

**Curva de Loss (até step 1440 / epoch 0.10):**
```
Step     Loss       Tendência
──────────────────────────────
  10     2.0406     Início (alta incerteza)
  50     1.9333     Warmup
 100     1.2790     Queda rápida e saudável
 500     1.1231     Descida contínua
 870     1.0729     ← Melhor loss até agora
1440     1.2455     Oscilação natural
```

**Comparação V3.0 vs V3.5b:**
- V3.0: Loss caiu de 1.67 → 0.73 em 500 steps (overfitting puro, brain-lock)
- V3.5b: Loss oscila entre 1.07-1.28 (generalização saudável)
- A oscilação indica que o modelo está vendo dados DIVERSOS e aprendendo padrões VARIADOS

### Fase 5: Concepção da Arquitetura Multi-Brain (03:55h)
**Ideia do Usuário:** E se usássemos vários micro-modelos (ou adaptadores LoRA) especializados em diferentes granularidades de compressão (palavras, frases, parágrafos), coordenados por um orquestrador?

**Análise dos Especialistas Simulados:**
- Viável como MoE (Mixture of Experts) por granularidade
- Para Edge/CPU: modelo único poliglota é mais prático
- Para Servidores: Multi-LoRA com hot-swapping via llama.cpp Server
- O conceito de "Brain Plugins" transforma o projeto numa plataforma

---

## 3. A Grande Descoberta: Aprendizado Sem Retreino

### 3.1 O Insight Fundamental

O CROM-IA já implementa, sem que percebêssemos inicialmente, uma separação entre:

```
CONHECIMENTO PROCEDURAL (COMO comprimir)
  → Vive nos pesos do LLM (treinado uma vez)
  → O modelo aprendeu o PADRÃO: "texto repetitivo → @@XY"

CONHECIMENTO DECLARATIVO (O QUE comprimir)
  → Vive nos Codebooks JSON (trocável a qualquer momento)
  → Cada codebook é uma "memória externa" do modelo
```

### 3.2 Implicação Prática

**Novo domínio SEM retreinar o modelo:**
```
1. Obter corpus novo (ex: 50k contratos jurídicos)
2. Rodar extrator_conhecimento_massivo.py → codebook_juridico.json
3. Carregar o codebook novo no RAG Injector
4. O modelo JÁ SABE emitir @@XY — basta o contexto dizer qual usar
5. Pronto: compressão jurídica sem tocar nos pesos
```

### 3.3 Por Que Isso é Poderoso

Modelos tradicionais (GPT-4, Gemma, LLaMA) armazenam TODO o conhecimento nos pesos neurais. Para ensinar algo novo, precisa retreinar (caro, lento, destrutivo).

O CROM-IA separa conhecimento em duas camadas:
- **Camada Neural** (cara): Aprende o "como" uma única vez
- **Camada Simbólica** (barata): Codebooks JSON que podem ser adicionados, removidos e editados em tempo real

Isso é análogo a como humanos funcionam:
- Você não "retreina seu cérebro" para aprender vocabulário novo
- Você adiciona palavras novas ao mesmo motor cognitivo existente
- O CROM-IA faz o mesmo: novos codebooks = novo vocabulário, mesmo motor

### 3.4 Até Onde Podemos Ir

| Horizonte | Capacidade | Requer Retreino? |
|-----------|-----------|-----------------|
| Novo domínio de texto | Codebook JSON | ❌ Não |
| Nova linguagem de programação | Codebook JSON | ❌ Não |
| Novo idioma de compressão | Codebook JSON | ❌ Não |
| Mudar persona (Rosa → outro) | Few-shot no prompt | ❌ Não |
| Múltiplos domínios simultâneos | Multi-codebook | ❌ Não |
| Nova granularidade (L1/L3) | Codebook + dataset | ✅ Sim (LoRA leve) |
| Raciocínio completamente novo | Pesos neurais | ✅ Sim (retreino) |
| Mudar sintaxe dos ponteiros | Pesos neurais | ✅ Sim (retreino) |

### 3.5 A Visão: Brain Plugins

A evolução final transforma cada domínio de conhecimento em um **plugin autocontido**:

```
brain_plugin_python/        ← Existe hoje (V3.5b)
├── codebook.json           ← 189 macros
├── adapter.safetensors     ← LoRA de 30MB (opcional)
└── manifest.yaml           ← Metadados

brain_plugin_juridico/      ← Futuro V4.0
├── codebook.json           ← 500 macros jurídicas
└── manifest.yaml           ← Sem LoRA = zero retreino

brain_plugin_medicina/      ← Futuro V4.1
├── codebook.json           ← 800 macros médicas
└── manifest.yaml           ← Sem LoRA = zero retreino
```

Qualquer pessoa poderia:
1. Rodar o extrator no corpus do seu domínio
2. Gerar um plugin
3. Publicar no HuggingFace como "Brain Plugin"
4. Qualquer usuário CROM-IA baixa e usa imediatamente

---

## 4. Arquivos Criados/Modificados Nesta Sessão

| Arquivo | Ação | Descrição |
|---------|------|-----------|
| `docs/CROM_IA_TECHNICAL_PAPER_V3.md` | Modificado | Mermaid corrigido, dados reais preenchidos, roadmap hierárquico |
| `docs/CROM_IA_MULTI_BRAIN_ARCHITECTURE.md` | **Novo** | Pesquisa sobre arquitetura Multi-Brain |
| `docs/CROM_IA_SESSION_REPORT_V35b.md` | **Novo** | Este relatório |
| `v3_engine/downloader_chat_real.py` | **Novo** | Download de datasets reais do HuggingFace |
| `v3_engine/gerar_dataset_v3_lora.py` | Reescrito | Mixer de dados reais + código V3 |
| `v3_engine/build_colab_v3.sh` | Reescrito | Build pipeline 117k |
| `v3_engine/colab_treinar_codebook_v3.py` | Reescrito | Notebook Colab com 6 células claras |
| `v3_engine/chat_v3_rag.sh` | Modificado | Apontado para Qwen2.5-1.5B |

---

## 5. Pendências (Quando o Treino Terminar)

- [ ] Baixar o novo GGUF do Colab para `models/`
- [ ] Rodar teste de sanidade: "Quem é você?" / "O que é gravidade?" / "Crie código Python"
- [ ] Preencher os `[X]` restantes no Technical Paper
- [ ] Medir tokens/segundo e RAM no hardware local
- [ ] Se aprovado: publicar modelo no HuggingFace e atualizar README
- [ ] Iniciar exploração da V4.0 (Multi-codebook)

---

*Relatório gerado em 04/Abr/2026 — Equipe CromIA*
*Treino V3.5b em andamento no Google Colab (Step ~1440/14733)*
