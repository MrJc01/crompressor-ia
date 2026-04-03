# 🧬 ROADMAP DE PESQUISA: CROM-IA GERAÇÃO 3

*Data do Registro: 03 de Abril de 2026*
*Contexto: Descoberta metodológica idealizada pelo Pesquisador Chefe durante o treinamento de Overfitting da Sessão 4.*

---

## 💡 Hipótese da Macro-Tokenização Frasal (Teorema Proposto)

**As Limitações da Compressão Sub-simbólica Atual (V2):**
Nosso CROM atual queima ciclos de processamento focando as taxas de substituição no nível micro/radical em matriz-fixa Radix-4 (1:3, 1:5). A inteligência artificial ainda se esforça muito para encadear os caracteres da Base-4 (`A, T, C, G`) um a um, como pequenos blocos de Lego para reconstruir palavras comuns.

**A Nova Abordagem Genial (V3 - Semântica Otimizada):**
> Em vez de comprimir caracteres fracionados ou sílabas fixas curtas (que geram excesso de tokens `@@`), o V3 operará por blocos de repetição de alto escopo! A ideia é mapear *conceitos inteiros*, *palavras dominantes/constantes* ou *frases prontas massivas* ancorando-as contra **apenas 1 ou 2 caracteres de índice**!

### O Poder na Engenharia e na Escala
1. **Dicionário Dimensional RAG:** Se a aplicação é um terminal, a constante `TT` não representa as letras soltas "e, a, u". Ela representa a saída integral pré-compilada: *"Por conta disso, a análise determinou que"*. A IA escreve na tela a string minúscula `TT` em 0.0001 milissegundos. O motor FUSE intercepta o `TT` e joga **100 bytes expansivos** para a memória do host!
2. **Escala de Compressão Surreal:** Evitamos a exaustão que o Llama.cpp ou o Unsloth têm ao adivinhar qual será a sílaba subsequente. Passamos da métrica linear para ordens exponenciais, atingindo tranquilamente **1:50 a 1:200** dependendo da tarefa especializada!

### Como Engatar ao Pipeline (To-Do de Sondas Iniciais):
- [ ] Construir novo Extrator de Frequências Massais e N-Grams (gerar um `codebook_macro.json` onde Chave = 'TT' e Valor = [Contexto de 400 letras]).
- [ ] O modelo LoRA não é mais treinado por soletração DNA Radix-4. Treinaremos simulando chamadas de contexto atômico onde a "resposta daquele parágrafo" se torna os bytes de gatilho do CROM.
- [ ] Testes de robustez no `RandomReader` FUSE. Ajustar desbalanceamentos de opcode de Mmap, pois injetar um buffer de 200 bytes em milissegundos exige ponteiros resilientes sobre a base original.
