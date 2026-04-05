# [ARCHITECTURE PROPOSAL] THE COGNITIVE LEAP (V4.3)

Este é o documento alvo da nossa próxima temporada de engenharia no CROM-IA.
O objetivo da "Engine 4.3" não é lutar contra deficiências sintáticas, e sim destravar o verdadeiro Potencial Computacional da máquina base, resolvendo os limites estreados na V4.2.

## O Que Fazer para Melhorar *Inteligência*?

1. **A Base de Dados Orgânica (Upgrade de Chassis):**
   O Core do LLM subjacente deve dar um salto dos miseros `0.6 Bilhões` para a categoria **>= 3 Bilhões (Ex: Phi-3-Mini-4k-Instruct-q4 ou Llama-3-8B-Q4_K_M)**. Modelos a partir deste escopo têm as cabeças de atenção e capacidade interna (KV-Cache horizon) para segurar código Python, RAG complexo e português coloquial SIMULTANEAMENTE na mesma árvore neural, sem engasgar e quebrar em loops como o predecessor de 0.6B limitou.

2. **Geração Dinâmica Extensível (O Roteamento Semântico em Python):**
   A atual técnica de roteamento bash em `chat_v42_brain.sh` foi fantástica como prova, mas na V4.3 o cérebro CROM deve usar a sua própria API python (usando as amarras nativas `llama-cpp-python`). Um orquestrador Python permite RAG contínuo sem sub-processo instável e a persistência do `Modo Chat` usando History Messages (algo bruto via Bash).

## O Que Fazer para Melhorar *Velocidade*?

1. **Flash Attention Activado:**
   Modelos de arquiteturas modernas suportam nativamente a flag `--flash-attn` no Llama.CPP V2. A utilização de FA (Flash Attention) otimiza os cáculos O(N²) de contexto do LLM em O(N), reduzindo o gargalo colossal no consumo de bateria/tempo quando o prompt passar de 1024 Tokens.

2. **Quantização Híbrida (MatMul Edge):**
   O modelo Mestre pode ser compilado no formato `IQ3_XXS`. Essa quantização matemática i-Matrix consome muito menos RAM (ex: `Llama-3-8b` pesando magros 3.3GB), e empurra os cáculos lógicos para a CPU num balanceamento altamente mais veloz que a K_M standard.

3. **Mapeamento de Threading Rigoroso:**
   No motor `benchmark_v42.sh`, usávamos `--threads 4`. Na V4.3, precisaremos de um autodetector (como `nproc --all`) cruzado contra a configuração HT (Hyper-Threading). Llama.CPP é comprovadamente muito mais rápido se travado ao número exato de Núcleos Físicos e despencando na velocidade caso entre nas threads lógicas do sistema.

## Marco Inicial (Setup):
- O desenvolvedor começará a V4.3 clonando a interface de orquestração anterior.
- Precisará puxar o LLM `GGUF` de sua preferência >=3B e depositá-lo na pasta designada `pesos_v43` (criada neste diretório).
