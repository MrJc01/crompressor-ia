# 🌌 CROM-IA Edge Visualizer (SRE Dashboard)

> _A ponte estética entre o Caos dos Sistemas de Arquivos Kernel (FUSE) e o "Aesthetic Wow Factor" no Terminal de Monitoramento._

Este diretório contém a joia frontal do ecosistema de Edge-Testing.
Onde testes de infraestrutura como `llama.cpp + FUSE mmap` ou `PyTorch Dataloaders` cospem massas numéricas opacas em log, este *Visualizador* capta, estrutura e plota a viabilidade da "Gravidade Zero de Memória" (Zero-Swapping) de modo Premium.

## 🛠 Arquitetura do Frontend

Atendendo a rigorosos requisitos de resiliência e velocidade de Borda Computacional (Edge computing constraint):
* **Não usamos frameworks baseados em NodeJS pesados** (sem React, Vue ou Tailwind). Frameworks alocariam megabytes ociosos apenas mantendo *Bundlers* rodando no background.
* O Visualizador usa **Vanilla HTML5, Javascript ES6 Nativo e CSS3 Puro**. Toda a responsividade e captura de telemetria corre na Engine Javascript interna do navegador sem prejudicar a CPU dedicada aos Mmaps da IA.

## 🎨 O "Aesthetic Wow Factor"

Para imersão total das Equipes Leigas e Investidores nas capacidades sistêmicas do motor CROM-IA:
- **Glassmorphism Dinâmico**: Os painéis de Telemetria (`#rss-value`, `#prune-value`) interagem usando camadas translúcidas (`backdrop-filter`) embasadas no modo sombrio (*Sleek Dark Mode*).
- **Tipografia Escalonada**: A interface opera na fonte elegante e legível `Inter` combinada com as tags SRE rígidas em `Fira Code`.
- **Efeitos e Gradientes Orbitais**: Fogo/frio controlados via Gradientes CSS complexos (`radial-gradient`) que pulsam, indicando que o Dataloader Pytorch CROM e os Inodes SQLite FUSE estão perfeitamente conectados O(1).

## 🚀 Como Iniciar (O Comando Simples)

Para levantar o Visualizador sem complicações e sem precisar procurar os arquivos soltos, incluímos um método de *One/Liner Server* usando utilitários que já estão pré-instalados com a plataforma.

Abra seu terminal na raiz da pasta `visualizador-sre` e digite:
```bash
python3 -m http.server 8080
```

Após isso, apenas abra o link que aparecerá no seu terminal `http://localhost:8080` no navegador web. 

Alternativamente, você pode rodar o inicializador global direto da raiz do projeto CROM-IA:
```bash
./iniciar_visualizador.sh
```

As métricas interativas simuladas de `RAM RSS Constraint`, `Poda Termodinâmica (Blocks Pruned pela entropia das palavras - Shannon > 7.5)` e o console de O(1) Streaming do `TinyLlama` iniciarão sua renderização instantaneamente ali.
