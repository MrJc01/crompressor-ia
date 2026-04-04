# 🧠 HANDOFF / CONTEXT SYNC: CROM-IA Geração V3

> **Nota para a IA (Novo Chat):**  
> Leia este documento inteiramente para se atualizar sobre a arquitetura atual do sistema CROM-IA e retomar imediatamente o desenvolvimento da Geração 3 (V3) de onde paramos.

---

## 1. O Estado da Arte Atual (Geração V2 Concluída)
Nosso projeto atual é focado na Compressão Sub-Simbólica Termodinâmica de Modelos de Linguagem para Dispositivos Edge (CPU only, 4GB RAM).
- **O Teto da V2:** Acabamos de confirmar experimentalmente que a V2 atingiu o seu limite de entropia. Ela usa a base Radix-4 (A, T, C, G) para decodificar micro-pedaços silábicos (N-Gramas de 1 a 3 palavras). Tentamos forçar a V2 a atingir N-gramas 1x5 (Dicionários de 15.000 entradas) em um Qwen 0.5B, resultando em *Colapso de Perplexidade* (LLM decorou o código DNA, mas esqueceu o roteamento de encadeamento semântico de linguagem).
- **Decisão Oficial:** A Fase V2 está congelada, madura e estável no **Codebook 1x3**. 

---

## 2. A Tese da V3: Macro-Tokenização / RAG Dimensional
Para transcender a limitação semântica dos modelos minúsculos, não enviaremos N-Gramas quebrados.
**Visão V3:** O Extrator vasculha um Corpus Específico fechado e engole intenções maduras, frases prontas, códigos gigantes ou jargões densos, apontando tudo isso contra apenas 2 caracteres de DNA ("Memory Pointers").
- *Exemplo Real Comprovado no Mockup:*  
  O token `AC` não significa "sopa de", mas sim: `"(Escala de 1-5, 1 significando nada confortável em tudo, 5 sendo muito confortável)"`.
- **Economia Termodinâmica:** Descompressão instantânea de 1:41+. O LLM gasta P=0 de ciclo sintático. A memória intercala o RAG dimensional injetando o bloco na stream C++.

---

## 3. O Que Já Está Pronto (A Pasta `v3_engine/` no GitHub)
No chat anterior, já codificamos 3 scripts protótipos funcionais que provam o conceito matemático:

1. **`extrator_conhecimento_massivo.py`**:
   - Varrerá qualquer dataset, quebrará a formatação por pontuação e \n, localizando Padrões de repetição exata (>40 chars).
   - *Status atual:* Rodamos no Alpaca Genérico e geramos 189 fragmentos densos (Salvou em `blocos_extraidos_v3.json`).

2. **`gerador_macro_codebook.py`**:
   - Anexou "Ponteiros de Memória O(1)" (Ex: `AA`, `AT`) a cada bloco para economizar Tokens Llama.
   - *Status atual:* Gerou o `macro_codebook_v3.json`.

3. **`mockup_fuse_expander.py`**:
   - Simulador terminal do hook interceptador. Um fluxo LLM falso minúsculo desidratado envia `AA`, e o Python pinta 80 bytes na tela simulando o injetor de Memória RAM.
   - *Funcionalidade provada: Zero Alucinação, Taxa 1:40 atingida na simulação*.

---

## 4. Onde Você Deve Começar (Tarefas Imediatas para este Novo Chat)

O foco deste novo chat é expandir a verticalidade da V3. Instruções diretas para recomeçar o raciocínio:

1. **Escolha de Novo Dataset Extremado (Domain-Specific):** 
   - A extração não rende centenas de kilobytes no Alpaca (pois ele não repete sentenças completas o tempo todo). Precisamos que você elabore/orquestre o download de um corpus de Código (Python Syntax) ou Jurídico (Contratos).
2. **Atualização do Extrator `extrator_conhecimento_massivo.py`:**
   - Adaptá-lo para comer esse novo tipo de dados bruto via Streaming Iterativo e não sobrecarregar a RAM da Máquina Local na busca por padronagens de alta escala.
3. **Draft da Rotina C++ (Nível Kernel/Engine):**
   - Transcrever a lógica de interceptação TTY do Python (`mockup_fuse_expander.py`) pra dentro da arquitetura do `llama.cpp` CLI via interceptador C ou Hook Pipe Bash nativo real (RAG Injector nativo).

---

> Próximo Comando do Usuário: Aguarde a confirmação de que eu li esse documento. Assim que eu falar *"Vamos seguir a Fase 4"*, inicie o diagnóstico da Task 1.
