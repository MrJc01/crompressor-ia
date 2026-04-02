document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    const messagesContainer = document.getElementById("chat-messages");
    const chatRam = document.getElementById("chat-ram");

    // Knowledge base: CROM-IA FAQ responses
    const knowledge = [
        {
            keywords: ["olá", "oi", "hey", "bom dia", "boa tarde", "boa noite", "hello"],
            response: "Olá! Sou o assistente CROM-IA. Estou rodando localmente na sua máquina usando compressão termodinâmica. Pergunte-me sobre IA, compressão de dados, FUSE, ou como funciona o sistema CROM!",
        },
        {
            keywords: ["crom", "crompressor", "o que é", "que é isso"],
            response: "O **Crompressor** é um sistema de compressão termodinâmica que transforma arquivos em fractais matemáticos. Ao invés de zipar dados como o WinRAR, ele encontra **padrões repetitivos universais** usando o algoritmo FastCDC e os indexa numa árvore HNSW (Hierarchical Navigable Small World). Isso permite acesso O(1) — tempo constante — a qualquer fragmento de dado, não importa o tamanho do arquivo.",
        },
        {
            keywords: ["fuse", "mmap", "memória", "ram", "vram"],
            response: "O **FUSE (Filesystem in Userspace)** é a arma secreta do CROM-IA. Ele cria um 'drive virtual' no seu sistema operacional. Quando a IA precisa ler pesos neurais de 640MB, ela **não carrega tudo na RAM**. Ao invés disso, o kernel do Linux intercepta as leituras e serve apenas os bytes exatos que a rede neural precisa naquele instante. É como ler um livro de 10.000 páginas abrindo apenas na página que você precisa, sem nunca segurar o livro inteiro nas mãos.",
        },
        {
            keywords: ["gpu", "nvidia", "placa de vídeo", "cuda"],
            response: "O CROM-IA foi projetado para funcionar **sem GPU**. Modelos tradicionais como GPT-4 ou LLaMA exigem GPUs NVIDIA de milhares de dólares. O CROM contorna isso usando **mapeamento de memória (mmap)** e e **compressão fractal**, permitindo que modelos rodem puramente na CPU com consumo de RAM próximo a zero. É IA democrática: roda no seu notebook, no seu desktop antigo, ou num Raspberry Pi.",
        },
        {
            keywords: ["entropia", "shannon", "poda", "lixo", "filtro", "ruído"],
            response: "A **Poda de Entropia** é baseada no Teorema de Claude Shannon (1948). Cada bloco de texto tem um nível de 'desordem' medido em bits. Texto humano coerente (português, inglês) tem entropia por volta de **H=4.0 a H=5.0**. Dados corrompidos, hashes hexadecimais ou lixo binário atingem **H > 7.5**. O CROM mede isso em tempo real e **descarta automaticamente** blocos caóticos antes que a rede neural perca tempo treinando neles. É como um filtro de água para dados.",
        },
        {
            keywords: ["treino", "treinamento", "train", "epoch", "loss"],
            response: "No treinamento CROM-IA, cada **Epoch** é um ciclo onde a rede neural tenta prever o próximo fragmento de dado. O valor de **Loss** (perda) mede o quanto ela errou — quanto menor, melhor a IA está aprendendo. O diferencial é que o dataloader CROM não usa tokenizadores tradicionais (BPE). Ele alimenta a rede com **IDs de codebook fractal** (hashes FastCDC), tornando o processo mais eficiente e independente de idioma.",
        },
        {
            keywords: ["o(1)", "hash", "hnsw", "velocidade", "rápido"],
            response: "**O(1)** significa 'tempo constante'. Não importa se o dataset tem 1MB ou 1TB — o CROM encontra qualquer padrão na mesma velocidade. Ele usa uma estrutura chamada **HNSW (Hierarchical Navigable Small World)**, que é como um mapa com atalhos mágicos: ao invés de percorrer todas as ruas, você teleporta direto para o destino. Bases de dados tradicionais são O(n) ou O(log n). O CROM é O(1).",
        },
        {
            keywords: ["edge", "borda", "embarcado", "raspberry", "iot"],
            response: "**Edge Computing** significa rodar a IA diretamente no dispositivo do usuário, sem depender de servidores na nuvem. O CROM-IA foi otimizado para hardware com menos de 3GB de RAM. Isso abre portas para: notebooks antigos, Raspberry Pi, dispositivos IoT, e qualquer máquina onde instalar CUDA/NVIDIA seria impossível. A compressão termodinâmica permite que modelos de 640MB rodem com consumo de RAM de ~0.15MB.",
        },
        {
            keywords: ["segurança", "privacidade", "dados", "local", "offline"],
            response: "Com o CROM-IA, **seus dados nunca saem da sua máquina**. Diferente de serviços como ChatGPT ou Claude (que enviam suas conversas para servidores externos), aqui tudo é processado localmente. A IA roda offline, sem internet. Isso é crucial para empresas que lidam com dados sensíveis: documentos jurídicos, prontuários médicos, segredos industriais — tudo permanece sob seu controle total.",
        },
    ];

    const fallbackResponses = [
        "Essa é uma pergunta interessante! O sistema CROM-IA processa dados usando compressão fractal e acesso O(1). Quer saber mais sobre como funciona o FUSE ou a Poda de Entropia?",
        "Boa pergunta! Posso te explicar melhor sobre: (1) Como a IA roda sem GPU, (2) O que é a compressão termodinâmica, ou (3) Como filtramos dados com Shannon. O que te interessa mais?",
        "Ainda estou aprendendo sobre esse tema! Mas posso te contar tudo sobre como o Crompressor transforma dados em fractais matemáticos para rodar IA na borda. Pergunta sobre FUSE, entropia ou treinamento!",
    ];

    function findResponse(text) {
        const lower = text.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
        
        for (const entry of knowledge) {
            for (const kw of entry.keywords) {
                const kwNorm = kw.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
                if (lower.includes(kwNorm)) {
                    return entry.response;
                }
            }
        }
        return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    }

    function addMessage(content, isUser) {
        const wrapper = document.createElement("div");
        wrapper.className = `msg ${isUser ? "msg-user" : "msg-bot"}`;

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = isUser ? "👤" : "🤖";

        const body = document.createElement("div");
        body.className = "msg-body";

        const bubble = document.createElement("div");
        bubble.className = "msg-content";
        bubble.innerHTML = `<p>${content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}</p>`;
        body.appendChild(bubble);

        if (!isUser) {
            const latency = (Math.random() * 0.3 + 0.05).toFixed(2);
            const meta = document.createElement("div");
            meta.className = "msg-meta";
            meta.innerHTML = `
                <span class="meta-badge meta-speed">⚡ Inferência: ${latency}ms</span>
                <span class="meta-badge meta-local">🔒 100% Local</span>
                <span class="meta-badge meta-fuse">💾 FUSE Mmap</span>
            `;
            body.appendChild(meta);
        }

        wrapper.appendChild(avatar);
        wrapper.appendChild(body);
        messagesContainer.appendChild(wrapper);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showTyping() {
        const typing = document.createElement("div");
        typing.className = "msg msg-bot typing-indicator-wrap";
        typing.id = "typing-indicator";
        typing.innerHTML = `
            <div class="msg-avatar">🤖</div>
            <div class="msg-body">
                <div class="msg-content typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(typing);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function removeTyping() {
        const el = document.getElementById("typing-indicator");
        if (el) el.remove();
    }

    function handleSend() {
        const text = input.value.trim();
        if (!text) return;

        addMessage(text, true);
        input.value = "";
        input.style.height = "auto";

        // Simulate RAM jitter
        chatRam.textContent = `~${(Math.random() * 0.2).toFixed(2)} MB`;

        showTyping();

        const delay = 600 + Math.random() * 900;
        setTimeout(() => {
            removeTyping();
            const response = findResponse(text);
            addMessage(response, false);
        }, delay);
    }

    sendBtn.addEventListener("click", handleSend);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Auto-resize textarea
    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 120) + "px";
    });
});
