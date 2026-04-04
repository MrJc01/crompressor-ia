# CROM-IA V4.0 Oficial: Arquitetura Multi-Brain Dinâmica
> **Status Operacional:** Arquitetura Congelada e Pronta.
> **Versão:** 4.0 (A V3.6 foi descartada e pulada em favor desta topologia).

## O Paradigma "Micro-Cérebros"

A geração V4 do projeto CROM-IA abandona o ideal de "um único LoRA massivo contendo tudo" para abraçar uma infraestrutura de Inteligência Modular *Composável*. 

Graças às amarrações com a alocação de Kernels do Linux via **FUSE** e **MMAP Zero-Copy** (que permitem o LLaMA rodar da SSD sem gastar RAM ativamente), concluímos que seria possível ter um Cérebro CROM estático (Qwen2.5) rodeado por **Infinitos Micro-Cérebros** que seriam "plugados" dinamicamente conforme o usuário usasse o Chat.

---

## 1. Fase de Extração Local (`v4_multibrain_engine/1_extracao_local`)
A sua placa mãe e CPU não sofrerão para processar datasets. A engenharia de extração inteira é executada com Scripts Leves:

- **O Motor N-Grams:** Textos Científicos e Livros não repetem blocos de forma limpa. O extrator SRE foi modificado para buscar **N-Grams por Janela Deslizante** (`get_ngrams()`), garantindo que até nuances textuais ganhem matrizes Radix-4 de compressão DNA.
- **O Transpilador Híbrido:** Todo Dataset deve ter **50% Mutante** e **50% Puro** para preservar os pesos e a linguagem natural original do modelo (evitar Memory Forgetting).
- **A Bateria Produzida:** Forjada fisicamente no laboratório, temos *Cérebro Python (Lógica)* e *Cérebro Medicina (Biologia)* devidamente hibridizados. A sub-pasta de drop se chama `arquivos_para_o_colab/`.

---

## 2. Fase de Treinamento Nuvem (`v4_multibrain_engine/2_treinamento_nuvem`)
A fábrica que molda a inteligência foi desenhada para interagir via Painel Web Sem Complicação (Para não ativar o Banimento de Boots Anti-SSH da Google).

Para fabricá-los:
1. O usuário acede o Google Colab em Browser limpo e faz o upload da massa hibridizada (Os `.jsonl` brutos).
2. O usuário cola no Colab o **Loop de Forja Unsloth**.
3. A máquina A100 lê um arquivo, cria os pesos, salva um arquivo **.GGUF de 30MB**, descarrega a VRAM brutalmente e inicia o próximo ciclo. Trata-se de uma Esteira Industrial de LoRAs.

---

## 3. Fase de Inferência e Deploy Edge (`v4_multibrain_engine/3_inferencia_local`)
Quando o modelo A100 termina o serviço, você os baixa para sua pasta de pesos local (`models/micro_cerebros`). 

O nosso script final de Interceptação (`chat_v4_multibrain.sh`) utiliza C++ Puro:
```bash
./llama-cli -m qwen2.5-1.5b.gguf \
  --lora micro_cerebros/Python_DNA.gguf \
  --lora micro_cerebros/Medicina_DNA.gguf
```
Você poderá carregar **centenas** destas flags `--lora` simultaneamente no Kernel local. A infraestrutura base vai carregar as regras (W, F, P) dinamicamente. Não existe limites físicos além do seu SSD.

---

### End-Game
Esta documentação é a pedra fundamental do salto tecnológico do CROM-IA para a V4. Toda a codificação na qual este laboratório operou no dia de hoje é estritamente designada para a orquestração desta pipeline em três fases.
