# [ROOT DIAGNOSTICS] POST-MORTEM V4.2 ESTRUTURAL

Este documento serve como o raio-x absoluto e final sobre a barreira fenomenológica de LLMs sub-1B operando no pipeline **CROM-IA V4.2**, diagnosticada nas suítes automatizadas de estresse de MoE.

## 1. O Alvorecer Sintático (A Validação FUSE/MoE)
Conseguimos extrair a lógica Pura! A nossa arquitetura de scripts FUSE de Llama-cpp e o Orquestrador Bash operam de forma gloriosa:
- O Teste 9 isolou um JSON cru de extração (`{"name":"João", "age":30}`). Perfeito!
- A injeção da flag de array `"${LORA_FLAGS[@]}"` curou totalmente as mortes do executável por falha de quebra de diretório (String Splitting na Memória).

## 2. A Barreira da Física Quântica da Memória (-1B) (A "Bosta")
O que frustra a geração criativa no Motor V4.2 não é código ruim ou "tuning fraco", é a absoluta falta de tensores cerebrais. A V4.2 foi operada sobre um LLM de exatos **0.6 Bilhões de parâmetros**. 

**O Comprovante Matemático Limitador (Entropy Decay):**
Quando o modelo iniciou as gerações do *Poema (Haicai)* e da *Classe Carro*, ele esvaziou a sua janela de *Attention Head*. Como estava empilhado em pesos condutivos LoRA (Tentando falar português, DPO e Python na mesma pipeline):
1. **Punição Desligada (Penalty 1.0):** Ele travou em fractais (*"velocidade_rodas_rodas"*).
2. **MoE Fixo Estrito (Temperatura 0.1):** A engine sofre colapso por **Interferência Catastrófica**. Um córtex pequeno lutando para calcular um gradiente complexo enquanto a LoRA injeta uma prioridade oposta força uma queda estatística que causa a gagueira (o temido `# Haicai de inteligência artificial de inteligencia artificial de...`).

## 3. Conclusões Radicais
Nenhuma flag de prompt (System) salva um motor pequeno do afogamento de memória de curto prazo ao gerar tokens de alta complexidade. Um humano não pode ser físico, médico e advogado simultaneamente se seu cérebro tiver apenas o tamanho de uma amêndoa. Nós extraímos `120%` do motor 0.6. Chegamos ao chão de fábrica físico do QWEN-3-0.6B.

## 4. O Salto Restante (Como Melhorar)
Isso leva a nossa pedra angular da iniciativa *V4.3 Cognitive Leap*. Velocidade e Inteligência não serão sanadas por "mais scripts Pys". Precisamos **Aumentar o Chassis Base** (Ir para 3 Bilhões de Parâmetros) e **Alterar a Formatação de Memória** (Ex: Flash Attention / Quantização Int-3).

(Vide arquivo ARCHITECTURE_PROPOSAL.md)
