# 🧠 CROM-IA V4+: Arquitetura Multi-Scale LoRA Orchestration

**Documento de Pesquisa — Abril 2026**
**Status:** Conceitual / Exploratório

---

## 1. Visão Geral

Este documento registra a arquitetura futura do CROM-IA baseada em **Orquestração de Micro-Cérebros** — onde múltiplos adaptadores LoRA especializados em diferentes granularidades de compressão são coordenados por um modelo orquestrador central.

A premissa fundamental é:

> *"E se, ao invés de treinar um modelo gigante que sabe tudo, treinássemos cérebros minúsculos ultra-especializados e os coordenássemos em tempo real?"*

---

## 2. Hierarquia de Compressão Multi-Nível

### 2.1 Os 4 Níveis

```
Nível 1: PALAVRA      →  Tokens individuais frequentes
                          "import" → @@WA
                          "return" → @@WB
                          Compressão: 1:3 a 1:5

Nível 2: FRASE         →  Linhas completas de código/texto
                          "import numpy as np" → @@FA
                          "    return result"   → @@FB
                          Compressão: 1:5 a 1:20

Nível 3: PARÁGRAFO     →  Blocos inteiros, funções completas
                          def sort_list(arr):
                              arr.sort()
                              return arr        → @@PA
                          Compressão: 1:20 a 1:100+

Nível 4: MISTO         →  O modelo intercala L1+L2+L3 na mesma resposta
                          Decide contextualmente qual nível usar
                          Compressão: Dinâmica e adaptativa
```

### 2.2 Namespacing dos Ponteiros

Para evitar colisão entre os níveis, cada nível usa um prefixo distinto:

| Nível | Prefixo | Exemplo | Codebook |
|-------|---------|---------|----------|
| L1 (Palavra) | `@@W` | `@@WA`, `@@WB` | `codebook_words.json` |
| L2 (Frase) | `@@F` | `@@FA`, `@@FB` | `codebook_lines.json` |
| L3 (Parágrafo) | `@@P` | `@@PA`, `@@PB` | `codebook_blocks.json` |
| L2 Legacy | `@@` | `@@AA`, `@@AT` | `macro_codebook_v3.json` |

---

## 3. Arquitetura: Mono-Modelo vs. Multi-Modelo

### 3.1 Opção A: Modelo Único Poliglota (Recomendado para Edge)

Um único Qwen de 1.5B ou 3B treinado com um dataset que contém **todos os níveis misturados**. O modelo aprende a decidir sozinho quando usar `@@W`, `@@F` ou `@@P`.

**Vantagens:**
- Roda em hardware Edge (CPU-only, <3GB RAM)
- Zero overhead de troca de contexto
- Uma única carga na memória

**Desvantagens:**
- Precisa retreinar para adicionar novos níveis
- Capacidade limitada pelo tamanho do modelo

```
┌─────────────────────────────────┐
│   Qwen 1.5B (Modelo Único)     │
│   LoRA treinado em L1+L2+L3    │
│   Decide internamente o nível  │
└──────────┬──────────────────────┘
           │ stdout (@@WA, @@FA, @@PA misturados)
           ▼
┌─────────────────────────────────┐
│   RAG Injector Unificado        │
│   Carrega 3 codebooks           │
│   Expande por prefixo           │
└─────────────────────────────────┘
```

### 3.2 Opção B: Multi-LoRA com Orquestrador (Para Servidores/GPUs)

Um modelo base compartilhado com 4 adaptadores LoRA que são ativados/desativados em tempo real via llama.cpp Server API.

**Vantagens:**
- Novos cérebros podem ser adicionados SEM retreinar os existentes
- Cada especialista pode ser treinado independentemente
- Escalabilidade horizontal infinita

**Desvantagens:**
- Requer llama.cpp em modo Server (API REST)
- Overhead de troca de LoRA (20-100ms por switch)
- Consome mais RAM (~30MB por adaptador ativo)

```
                    ┌────────────────────┐
                    │  LoRA Orquestrador  │
                    │  (Decide o Nível)   │
                    └────────┬───────────┘
                             │ <invoke:L?>
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  LoRA L1   │  │  LoRA L2   │  │  LoRA L3   │
     │ Palavras   │  │ Frases     │  │ Parágrafos │
     │  ~30MB     │  │  ~30MB     │  │  ~30MB     │
     └────────────┘  └────────────┘  └────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌────────────────────┐
                    │  Modelo Base Qwen  │
                    │  (Pesos Frozen)    │
                    │  ~900MB GGUF       │
                    └────────────────────┘
```

---

## 4. Adicionar Novos Cérebros SEM Retreinar

Este é o insight mais poderoso de toda a arquitetura CROM-IA.

### 4.1 O Princípio: Memória Declarativa Separada

O sistema CROM-IA já implementa, naturalmente, uma separação entre:
- **Conhecimento Procedural** (COMO comprimir) → Vive dentro dos pesos do LLM
- **Conhecimento Declarativo** (O QUE comprimir) → Vive nos Codebooks JSON

Isso significa que **novos domínios podem ser adicionados trocando apenas o arquivo JSON**, sem tocar no modelo.

### 4.2 Pipeline de Adição de Novo Domínio (Zero Retreino)

```
1. CORPUS NOVO            →  Ex: 50k contratos jurídicos em português
                               Ex: 30k fichas médicas
                               Ex: 20k documentos contábeis

2. EXTRATOR AUTOMÁTICO    →  extrator_conhecimento_massivo.py
                               Já existe! Analisa frequências e extrai patterns

3. NOVO CODEBOOK          →  codebook_juridico.json
                               Ponteiros: @@JA, @@JB, @@JC...
                               "O réu deverá comparecer..." → @@JA

4. ZERO RETREINO          →  O modelo JÁ sabe emitir @@XY
                               Basta ele ver o ponteiro no contexto
                               O RAG Injector expande automaticamente

5. HOT-SWAP               →  Troca o codebook em runtime
                               O mesmo modelo serve Código, Direito e Medicina
```

### 4.3 Por que isso funciona?

O modelo, durante o treinamento V3.5b, aprendeu o **padrão semântico**:
> "Quando o texto é repetitivo ou previsível, emita `@@` + 2 chars"

Ele não memorizou quais strings específicas mapear. Ele aprendeu a **intenção** de comprimir. Portanto, se o contexto dele disser "use @@JA para cláusula penal", ele vai usar @@JA.

Isso é análogo a como humanos funcionam:
- Você não "retreina" seu cérebro para aprender uma nova língua
- Você adiciona vocabulário novo ao mesmo motor cognitivo
- O CROM-IA faz o mesmo: novos codebooks = novo vocabulário, mesmo motor

### 4.4 Limitações do Zero-Retreino

| Aspecto | Funciona sem retreino? | Observação |
|---------|----------------------|------------|
| Novo domínio de texto | ✅ Sim | Basta gerar codebook |
| Nova linguagem de programação | ✅ Sim | Extrator é agnóstico |
| Mudar a persona (Rosa → João) | ⚠️ Parcial | Precisa few-shot no prompt |
| Ensinar raciocínio novo | ❌ Não | Requer LoRA adicional |
| Mudar a sintaxe dos ponteiros | ❌ Não | Requer retreino |

---

## 5. Arquitetura de Cérebros Composáveis (Visão V5.0+)

### 5.1 Brain Plugins

A evolução final é transformar cada "cérebro" em um **plugin** autocontido:

```
brain_plugin_python/
├── codebook.json          # 189 macros de código Python
├── adapter.safetensors    # LoRA de 30MB (opcional)
├── manifest.yaml          # Metadados: versão, domínio, autor
└── examples.jsonl         # 10 few-shot exemplos de uso

brain_plugin_juridico/
├── codebook.json          # 500 macros jurídicas
├── adapter.safetensors    # LoRA especializado (opcional)
├── manifest.yaml
└── examples.jsonl

brain_plugin_medicina/
├── codebook.json          # 800 macros médicas  
├── manifest.yaml
└── examples.jsonl          # Sem LoRA = zero-retreino
```

### 5.2 Brain Stacking (Empilhamento de Cérebros)

Múltiplos plugins podem ser carregados simultaneamente:

```python
# Configuração do motor de inferência
engine.load_brain("brain_plugin_python")   # Codebook Python
engine.load_brain("brain_plugin_juridico") # Codebook Jurídico
engine.load_brain("brain_plugin_medicina") # Codebook Médico

# O RAG Injector unifica todos os codebooks em runtime
# @@FA → Código Python
# @@JA → Cláusula Jurídica
# @@MA → Diagnóstico Médico

# O modelo decide qual domínio usar pelo CONTEXTO da conversa
```

### 5.3 Marketplace de Cérebros (Visão de Longo Prazo)

Se a arquitetura de plugins funcionar, qualquer pessoa pode:
1. Rodar o extrator no corpus do seu domínio
2. Gerar um codebook
3. Publicar como um "Brain Plugin" no HuggingFace
4. Qualquer usuário CROM-IA baixa e usa sem retreinar

Isso transforma o CROM-IA de um "modelo local" em uma **plataforma de compressão cognitiva modular**.

---

## 6. Análise de Viabilidade

### 6.1 O que funciona HOJE (Pronto)
- [x] Extração automática de macros de qualquer corpus
- [x] Geração de codebooks com ponteiros DNA
- [x] RAG Injector O(1) com hot-swap de codebooks
- [x] Pipeline de treinamento LoRA no Colab

### 6.2 O que precisa ser construído (V4.0)
- [ ] Namespacing de ponteiros por domínio (@@W, @@F, @@P, @@J)
- [ ] Loader multi-codebook no RAG Injector
- [ ] Sistema de manifesto (manifest.yaml) para plugins
- [ ] Testes de composição (2+ codebooks simultâneos)

### 6.3 O que é pesquisa pura (V5.0+)
- [ ] Hot-swapping de LoRA adapters em tempo real (llama.cpp server)
- [ ] Orquestrador neural que decide o nível de compressão
- [ ] Marketplace de Brain Plugins
- [ ] Auto-geração de codebooks durante a inferência (self-extending memory)

---

## 7. Riscos Identificados

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Colisão de ponteiros entre domínios | Média | Alto | Namespacing estrito (@@W, @@J, @@M) |
| Overhead de RAM com muitos codebooks | Baixa | Médio | Lazy-loading por demanda |
| Modelo "esquece" padrão @@ com LoRA fraco | Média | Alto | Manter % mínima de código no dataset |
| Latência de hot-swap de LoRA | Alta | Médio | Usar Opção A (mono-modelo) para Edge |

---

*Documento criado em Abril 2026 — Equipe CromIA*
*Para discussão e implementação futura. Este é um living document.*
