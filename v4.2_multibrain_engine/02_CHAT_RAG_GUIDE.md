# CROM-IA V4.2 — Monitor de Orquestração + RAG-Lite

## Conceito

O `chat_v42_brain.sh` é um **painel de controle TUI** que permite configurar tudo antes de conversar: ativar/desativar cérebros, adicionar arquivos/pastas como contexto, ajustar parâmetros. Funciona como RAG (Retrieval Augmented Generation) adaptado para rodar sem GPU.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Arquivos   │────▶│ rag_contexto │────▶│ System Prompt│
│  /pastas    │     │   .py        │     │  enriquecido │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │  LoRA Stack  │────▶│  llama-cli   │
                    │  (auto)      │     │  --conversation│
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │  DNA Decoder │◀────│   Resposta   │
                    └──────────────┘     └─────────────┘
```

---

## Uso

### Monitor Interativo (recomendado)
```bash
./chat_v42_brain.sh
# Abre o painel TUI onde você configura tudo:
#   [1-9] Toggle cérebros ON/OFF
#   [a]   Adicionar arquivo para RAG
#   [p]   Adicionar pasta para RAG
#   [ENTER] Lançar chat com a config escolhida
#   Ctrl+C no chat volta ao monitor!
```

### Pré-carregar arquivos (atalho)
```bash
./chat_v42_brain.sh --arquivo main.py --pasta ./src/
# Abre o monitor já com esses itens no RAG
# Você ainda pode ajustar antes de lançar
```

### Fluxo Típico
```
1. ./chat_v42_brain.sh
2. Pressiona [3] para desativar Medicina_DNA
3. Pressiona [a] e digita "./api.py"
4. Pressiona [p] e digita "./controllers/"
5. Pressiona [ENTER] → chat inicia!
6. Conversa...
7. Ctrl+C → volta ao monitor
8. Pressiona [3] para reativar Medicina_DNA  
9. Pressiona [ENTER] → chat reinicia com nova config!
```

---

## Formatos Suportados

| Extensão | Tipo | Tratamento |
|---|---|---|
| `.py` | Python | Código completo |
| `.js` | JavaScript | Código completo |
| `.sh` | Shell | Código completo |
| `.md` | Markdown | Texto completo |
| `.txt` | Texto | Texto completo |
| `.json` | JSON | Estrutura + primeiros 2KB |
| `.jsonl` | JSON Lines | Primeiras 20 linhas |
| `.html` | HTML | Texto extraído (sem tags) |
| `.css` | CSS | Código completo |
| `.yaml/.yml` | YAML | Estrutura completa |
| `.toml` | TOML | Estrutura completa |
| `.cfg/.ini` | Config | Estrutura completa |
| `.log` | Log | Últimas 50 linhas |

**Limite por arquivo:** 3000 chars (para caber no contexto)
**Limite total:** ~6000 chars de contexto injetado (~1500 tokens)

---

## Como Funciona (Detalhes Técnicos)

### 1. Ingestão (`rag_contexto.py`)
```python
# Lê todos os arquivos/pastas especificados
# Retorna lista de {nome, conteudo, tipo}
```

### 2. Chunking
- Cada arquivo é dividido em chunks de ~500 chars
- Preserva limites de função/classe em código
- Preserva parágrafos em texto

### 3. Indexação por Keywords
Sem embeddings (sem GPU), usamos TF-IDF simplificado:
- Conta frequência de cada palavra em cada chunk
- Na hora da pergunta, busca chunks com mais keywords em comum
- Retorna top-3 chunks mais relevantes

### 4. Injeção no System Prompt
```
<|im_start|>system
Você é CROM-IA, assistente brasileiro com compressão DNA ativa.

CONTEXTO DOS ARQUIVOS CARREGADOS:
📄 main.py (Python, 45 linhas):
```python
def calcular_total(items):
    return sum(item.price for item in items)
...
```

📄 README.md (Markdown):
# Meu Projeto
Calculadora de preços para e-commerce...

Responda perguntas sobre estes arquivos usando seu conhecimento.
<|im_end|>
```

### 5. LoRA Stacking
O script auto-detecta todos os `*_lora.gguf` em `micro_cerebros/`:
```bash
llama-cli -m base.gguf \
    --lora Base_PTBR.gguf \
    --lora Python_DNA.gguf \
    -c 2048 \
    --prompt "CONTEXTO: ..." \
    --conversation
```

---

## Limitações

| Limitação | Causa | Workaround |
|---|---|---|
| Contexto máximo ~1500 tokens | i5 + RAM limitada | Chunking inteligente |
| Sem busca semântica | Sem GPU para embeddings | Keywords TF-IDF |
| Sem persistência entre sessões | Design stateless | Pode re-carregar arquivos |
| Arquivos grandes truncados | Limite de contexto | Mostra início + estrutura |

---

## Exemplos de Conversas

### Exemplo 1: Analisar código
```
$ ./chat_v42_brain.sh --arquivo api.py

> Explique o que faz a função handle_request

CROM-IA: A @@FNC handle_request recebe um @@DCT com os parâmetros 
HTTP, valida os campos 'user_id' e 'action', e @@RET um JSON 
com o resultado. Ela usa @@TRY/@@EXC para tratar erros de 
conexão com o banco...
```

### Exemplo 2: Analisar projeto
```
$ ./chat_v42_brain.sh --pasta ./meu_app/

> Qual é a arquitetura desse projeto?

CROM-IA: Baseado nos arquivos que analisei, o projeto segue 
arquitetura MVC:
- models/ → @@CLS de dados (User, Product)
- views/ → Templates HTML
- controllers/ → Lógica de negócio
- main.py → Entry point com @@IMP Flask
```

---

## Performance Esperada

| Métrica | Sem arquivos | Com 1 arquivo | Com pasta (10 arquivos) |
|---|---|---|---|
| Tempo de carga | ~5s | ~6s | ~8s |
| Velocidade chat | 7-9 t/s | 6-8 t/s | 5-7 t/s |
| RAM | 635MB | ~650MB | ~680MB |
| Qualidade resposta | Geral | Específica ao código | Visão do projeto |
