document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    const messagesContainer = document.getElementById("chat-messages");
    const chatRam = document.getElementById("chat-ram");

    // Conversation memory
    let lastTopic = null;
    let conversationDepth = 0;
    let askedTopics = new Set();

    // Rich knowledge base with multi-level depth
    const topics = {
        crom: {
            name: "Crompressor",
            keywords: ["crom", "crompressor", "compressor", "projeto", "sistema", "plataforma"],
            intro: "O **Crompressor** é um motor de compressão termodinâmica que eu uso como meu cérebro. Diferente do WinRAR ou 7-Zip que apenas compactam bytes, o CROM encontra **padrões fractais universais** nos dados usando um algoritmo chamado FastCDC (Content-Defined Chunking). Ele quebra qualquer arquivo em pedaços inteligentes e os indexa numa árvore HNSW — uma estrutura matemática que permite encontrar qualquer fragmento em **tempo constante O(1)**, não importa se o arquivo tem 1MB ou 1TB. É como ter um índice mágico que teleporta direto para a informação certa.",
            deep: "Indo mais fundo: o Crompressor opera em 3 camadas. **Camada 1 (FastCDC):** quebra os dados em chunks de tamanho variável baseado no conteúdo — não em posições fixas. Isso significa que se você mudar 1 byte no meio de um arquivo de 10GB, apenas aquele chunk específico muda. **Camada 2 (HNSW):** indexa cada chunk como um vértice num grafo de mundo pequeno, criando atalhos matemáticos entre padrões similares. **Camada 3 (Delta Compression):** armazena apenas as diferenças entre chunks parecidos, atingindo razões de compressão extremas. O resultado é um arquivo `.crom` que funciona como um banco de dados O(1) ultracomprimido.",
        },
        fuse: {
            name: "FUSE Mmap",
            keywords: ["fuse", "mmap", "virtual", "drive", "mount", "filesystem", "disco", "arquivo"],
            intro: "O **FUSE (Filesystem in Userspace)** é a mágica que me permite existir sem devorar sua RAM. Funciona assim: o Crompressor cria um **drive virtual** no seu sistema operacional. Quando eu preciso ler os pesos neurais (um arquivo de ~640MB), o Linux intercepta cada leitura e serve **apenas os bytes exatos** que eu preciso naquele milissegundo. Imagine um livro de 10.000 páginas: ao invés de segurar o livro inteiro nas mãos, eu simplesmente materializo a página certa no ar, leio, e ela desaparece. O resultado? Eu ocupo **menos de 1MB de RAM** enquanto processo modelos de centenas de megabytes.",
            deep: "O mecanismo técnico é o **mmap (memory-mapped files)**. Quando o FUSE monta o arquivo `.crom`, o kernel do Linux cria uma região de memória virtual que aponta diretamente para o disco. A CPU acha que está lendo RAM, mas na verdade está lendo NVMe/SSD. Os **page faults** que aparecem no monitor SRE são exatamente isso: momentos onde o processador pediu um byte que ainda não estava em cache e o kernel buscou silenciosamente no disco. O valor de Major PageFaults = 0 significa que **nenhuma** dessas buscas causou travamento. É leitura transparente e perfeita.",
        },
        gpu: {
            name: "IA sem GPU",
            keywords: ["gpu", "nvidia", "placa", "video", "cuda", "hardware", "computador", "notebook", "rodar", "requisito", "precisa"],
            intro: "Eu fui projetado para ser **anti-elitista**. Enquanto modelos como GPT-4 ou Claude rodam em clusters de GPUs NVIDIA A100 que custam dezenas de milhares de dólares, eu opero puramente na **CPU comum** do seu computador. Como? O segredo é o mapeamento de memória via FUSE: ao invés de carregar 640MB de pesos neurais na RAM (o que causaria um crash instantâneo em máquinas modestas), eu leio fragmentos microscópicos sob demanda. Seu notebook com 4GB de RAM pode me rodar. Um Raspberry Pi pode me rodar. A **democratização da IA** começa aqui.",
            deep: "A razão técnica pela qual GPUs são tradicionalmente necessárias é a **paralelização massiva**: uma GPU tem milhares de núcleos CUDA que multiplicam matrizes simultaneamente. O CROM contorna isso de duas formas: (1) usando **quantização Q4_K_M** que reduz cada peso de 16 bits para 4 bits, cortando a computação em 4x; e (2) usando **mmap para streaming** — ao invés de ter todos os pesos na memória para multiplicação matricial, fazemos passes sequenciais lendo direto do disco. É mais lento que uma A100? Sim. Mas roda no seu hardware existente, sem custo extra, sem cloud, sem enviar dados para ninguém.",
        },
        entropia: {
            name: "Entropia de Shannon",
            keywords: ["entropia", "shannon", "poda", "filtro", "lixo", "ruido", "qualidade", "limpar", "sanitiz", "caos", "caotico"],
            intro: "A **Entropia de Shannon** é minha ferramenta de higiene mental. Em 1948, Claude Shannon definiu uma fórmula que mede a **\"desordem informacional\"** de qualquer sequência de dados. Funciona com logaritmos base 2 (bits). Um texto em português coerente tem entropia por volta de **H ≈ 4.0 a 5.0** — existe ordem, repetição, estrutura gramatical. Já um hash hexadecimal como `a7f3b2c9e1d8` atinge **H > 7.5** — é pura aleatoriedade, sem padrão detectável. Eu uso isso como um escudo: antes de treinar em qualquer dado, meço sua entropia. Se for caótico demais, eu **descarto o bloco inteiro** antes que ele polua meus pesos neurais. É como um filtro de água para informação.",
            deep: "A fórmula exata é: **H = -Σ p(x) × log₂(p(x))**, onde p(x) é a probabilidade de cada byte aparecer na sequência. Para um texto com 26 letras equidistribuídas, H ≈ 4.7 bits. Para bytes completamente aleatórios (256 símbolos equiprováveis), H = 8.0 bits — o máximo absoluto. O limiar de corte **7.5** foi calibrado empiricamente: abaixo disso, pode ser código-fonte, JSON estruturado ou texto com caracteres especiais — útil. Acima disso, é quase certamente lixo binário, hashes criptográficos ou dados corrompidos. A poda acontece no DataLoader PyTorch antes do `yield`, então a GPU/CPU **nunca desperdiça ciclos** processando lixo.",
        },
        treino: {
            name: "Treinamento Neural",
            keywords: ["treino", "treinamento", "train", "epoch", "loss", "aprender", "peso", "neural", "rede"],
            intro: "Meu processo de aprendizado funciona assim: recebo dados comprimidos em formato `.crom`, decomponho cada chunk em **IDs de codebook fractal** (via FastCDC hashing), e alimento uma rede neural Transformer. A cada **Epoch** (ciclo completo), a rede tenta prever o próximo fragmento. O **Loss** (perda) mede o erro — começa alto (~0.45) e vai caindo conforme eu aprendo padrões. O diferencial radical é que eu **não uso tokenizadores BPE** como o GPT. Enquanto o GPT quebra \"inteligência\" em pedaços fixos como [\"int\", \"elig\", \"ência\"], eu opero com hashes semânticos que capturam o **significado fractal** da palavra inteira.",
            deep: "A arquitetura interna é um **Transformer modificado** chamado NextGenCromLLM. Ao invés de um vocabulário fixo de 50.000 tokens BPE, eu uso um codebook HNSW de **4.096 padrões fractais**. Cada padrão representa um cluster semântico de chunks de dados. O embedding layer recebe esses IDs fractais e produz representações densas de 256 dimensões. O treinamento usa **Adam optimizer** com learning rate de 3e-4 e batch size adaptativo baseado na entropia do lote — lotes mais caóticos recebem learning rates menores automaticamente. O DataLoader é um PyTorch IterableDataset que lê via mmap, garantindo zero-copy durante o treinamento inteiro.",
        },
        seguranca: {
            name: "Privacidade e Segurança",
            keywords: ["seguranca", "seguro", "privacidade", "privado", "dado", "vazamento", "offline", "local", "nuvem", "cloud", "enviar"],
            intro: "**Seus dados nunca saem da sua máquina.** Ponto final. Diferente do ChatGPT (OpenAI), Claude (Anthropic), ou Gemini (Google) — que enviam cada mensagem sua para servidores remotos onde ela pode ser armazenada, analisada, e usada para treinar modelos futuros — eu processo **tudo localmente**. Meus pesos neurais estão no seu disco. Meu motor FUSE roda no seu kernel. Sua CPU faz os cálculos. Nenhum pacote de rede sai da sua máquina. Para empresas que lidam com **dados sensíveis** (prontuários médicos, processos jurídicos, segredos industriais, dados financeiros), isso não é um luxo — é uma necessidade existencial.",
            deep: "A arquitetura de segurança vai além do offline. O protocolo **Zero-Footprint Teardown** garante que, ao encerrar uma sessão de treino ou inferência, todos os artefatos temporários (montagens FUSE, caches SQLite, buffers de KV-cache) são **obliterados** automaticamente via `fusermount -u` e `rm -rf` programático. Não sobra vestígio forense. Além disso, o binário `crompressor_bin` é compilado em Rust com proteções de memória (no buffer overflows, no use-after-free), eliminando classes inteiras de vulnerabilidades que afetam motores de IA em C/C++.",
        },
        edge: {
            name: "Edge Computing",
            keywords: ["edge", "borda", "embarcado", "raspberry", "iot", "leve", "pequeno", "fraco", "limitado", "3gb", "pouco"],
            intro: "**Edge Computing** é a filosofia de rodar inteligência diretamente no dispositivo do usuário. Eu fui otimizado para ambientes severos: máquinas com **menos de 3GB de RAM**, processadores modestos, sem GPU, sem conexão à internet. Isso abre portas que a IA tradicional não consegue abrir: um médico rural pode usar IA para análise de laudos sem internet. Uma fábrica pode rodar inspeção visual sem enviar imagens para a nuvem. Um jornalista em zona de conflito pode processar documentos sem risco de interceptação. A compressão termodinâmica do CROM torna isso possível onde antes era impensável.",
            deep: "Os números concretos: o modelo TinyLlama-1.1B quantizado em Q4_K_M pesa ~640MB no disco. Via FUSE mmap, ele consome **~0.15MB de RSS (RAM residente)** durante inferência — isso é 4.000 vezes menos que o carregamento convencional. O Swap (memória de disco de emergência) permanece em **zero**, indicando que o kernel está perfeitamente satisfeito com a estratégia de paginação. O hit rate de O(1) se mantém acima de 99.8%, significando que praticamente toda busca no índice HNSW encontra o dado correto na primeira tentativa.",
        },
        o1: {
            name: "Tempo Constante O(1)",
            keywords: ["o(1)", "o1", "tempo constante", "constante", "hash", "hnsw", "rapido", "velocidade", "performance", "busca"],
            intro: "**O(1)** é a notação mais preciosa da ciência da computação. Significa que o tempo de execução **não cresce** com o tamanho da entrada. Se tenho 1.000 dados ou 1.000.000.000, a busca leva o mesmo tempo. O CROM atinge isso usando a estrutura **HNSW (Hierarchical Navigable Small World)**: um grafo multi-camada onde cada nó tem atalhos para nós distantes. Imagine uma cidade onde, além das ruas normais, existem portais de teletransporte entre bairros. Você nunca precisa andar a cidade inteira — simplesmente pula pelo portal mais próximo do destino. Bancos de dados tradicionais são O(n) ou O(log n). O CROM é O(1). Isso é revolucionário.",
            deep: "A estrutura HNSW funciona em camadas. A **camada 0** contém todos os nós do grafo. As **camadas superiores** são subconjuntos cada vez menores com links de longo alcance. Para buscar, você entra pela camada mais alta (poucos nós, links longos), desce pela camada que está mais próxima do alvo, e navega localmente na camada 0 para refinar. Em termos de complexidade amortizada, a busca é O(log(log(n))) — tão próximo de O(1) que na prática é indistinguível. O `.cromdb` (codebook) armazena essa árvore inteira serializada. Quando o DataLoader faz um lookup de hash, ele desserializa a referência do nó HNSW e salta diretamente ao chunk correto no arquivo `.crom`.",
        },
    };

    // Conversational patterns (greetings, affirmations, meta-questions)
    const conversational = {
        greetings: {
            patterns: ["oi", "ola", "olá", "eae", "eai", "e ae", "e aí", "fala", "salve", "hey", "hello", "bom dia", "boa tarde", "boa noite", "yo", "ae"],
            responses: [
                "E aí! 👋 Eu sou o CROM Assistant — uma IA que roda **100% na sua máquina**, sem cloud, sem GPU, sem espionagem. Meu cérebro é alimentado por compressão fractal e FUSE mmap. Pode me perguntar qualquer coisa: como funciono, o que me diferencia do ChatGPT, por que não preciso de GPU, como filtro dados com entropia de Shannon... Manda a real!",
                "Salve! 🤖 Tô rodando aqui no seu hardware local usando menos de 1MB de RAM. Parece mentira, mas é matemática pura. Quer entender como isso é possível? Me pergunta sobre FUSE, entropia, treinamento neural, ou por que eu não preciso de placa de vídeo cara!",
            ]
        },
        affirmations: {
            patterns: ["sim", "yes", "s", "claro", "quero", "manda", "bora", "pode ser", "to dentro", "dale", "show", "blz", "beleza", "ok"],
        },
        explain_all: {
            patterns: ["tudo", "explica tudo", "me explique tudo", "conta tudo", "quero saber tudo", "saber mais", "mais detalhes", "aprofunda", "detalhe", "aprofundar"],
        },
        capabilities: {
            patterns: ["o que sabe", "sabe fazer", "consegue", "capacidade", "habilidade", "funcionalidade", "recurso", "feature"],
            response: "Posso te explicar em profundidade sobre **7 pilares** do ecossistema CROM-IA:\n\n🔹 **Crompressor** — Como funciona a compressão fractal termodinâmica\n🔹 **FUSE Mmap** — O drive virtual que me permite existir sem RAM\n🔹 **IA sem GPU** — Por que não preciso de NVIDIA para pensar\n🔹 **Entropia de Shannon** — Como filtro lixo digital antes de aprender\n🔹 **Treinamento Neural** — Como aprendo usando codebooks fractais ao invés de BPE\n🔹 **Privacidade** — Por que seus dados nunca saem da sua máquina\n🔹 **Edge Computing** — IA em dispositivos com menos de 3GB de RAM\n\nEscolhe qualquer um e eu mergulho fundo! Ou pergunta algo específico que eu conecto os pontos. 🧠",
        },
        thanks: {
            patterns: ["obrigado", "valeu", "thanks", "vlw", "thanx", "brigado", "grato", "agradeco"],
            response: "De nada! 🙌 Lembra: tudo que eu te disse foi processado aqui na sua própria máquina. Nenhum servidor externo viu nossa conversa. Se quiser explorar mais algum tema ou testar o **Sanitizador de Dados** (que usa essa mesma entropia de Shannon em tempo real), é só voltar ao portal! Tô aqui rodando em silêncio no seu FUSE. 🔒",
        },
        unknown: [
            "Hmm, não tenho uma resposta específica pra isso, mas posso te dar um panorama geral. O CROM-IA opera em 3 eixos: **compressão fractal** (como armazenamos dados), **FUSE mmap** (como lemos sem usar RAM), e **poda de entropia** (como filtramos lixo). Qual desses te interessa mais?",
            "Essa é uma área que estou expandindo! Mas posso conectar com o que sei: toda informação que processo passa pelo pipeline **FastCDC → HNSW → Codebook → Transformer**. Quer que eu detalhe alguma dessas etapas?",
            "Ainda não tenho dados profundos sobre isso, mas tenho expertise em: compressão termodinâmica, inferência sem GPU, filtragem de entropia e privacidade local. Tenta me perguntar sobre algum desses! 🧬",
        ],
    };

    function normalize(text) {
        return text.toLowerCase()
            .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
            .replace(/[?!.,;:]/g, "")
            .trim();
    }

    function matchesAny(text, patterns) {
        const norm = normalize(text);
        return patterns.some(p => {
            const pNorm = normalize(p);
            // Exact match for short patterns, includes for longer ones
            if (pNorm.length <= 3) return norm === pNorm || norm.split(/\s+/).includes(pNorm);
            return norm.includes(pNorm);
        });
    }

    function findTopicMatch(text) {
        const norm = normalize(text);
        let bestMatch = null;
        let bestScore = 0;

        for (const [key, topic] of Object.entries(topics)) {
            let score = 0;
            for (const kw of topic.keywords) {
                const kwNorm = normalize(kw);
                if (norm.includes(kwNorm)) {
                    score += kwNorm.length; // Longer keyword matches score higher
                }
            }
            if (score > bestScore) {
                bestScore = score;
                bestMatch = key;
            }
        }

        return bestScore > 0 ? bestMatch : null;
    }

    function getUnexploredTopics() {
        return Object.entries(topics)
            .filter(([key]) => !askedTopics.has(key))
            .map(([key, t]) => t.name);
    }

    function buildResponse(text) {
        const norm = normalize(text);

        // 1. Greetings
        if (matchesAny(text, conversational.greetings.patterns)) {
            lastTopic = null;
            conversationDepth = 0;
            const r = conversational.greetings.responses;
            return r[Math.floor(Math.random() * r.length)];
        }

        // 2. Thanks
        if (matchesAny(text, conversational.thanks.patterns)) {
            return conversational.thanks.response;
        }

        // 3. Capabilities question
        if (matchesAny(text, conversational.capabilities.patterns)) {
            return conversational.capabilities.response;
        }

        // 4. "Explain everything" or "more details" 
        if (matchesAny(text, conversational.explain_all.patterns)) {
            // If we have a last topic, go deep
            if (lastTopic && topics[lastTopic]) {
                const topic = topics[lastTopic];
                if (!askedTopics.has(lastTopic + "_deep")) {
                    askedTopics.add(lastTopic + "_deep");
                    const unexplored = getUnexploredTopics();
                    let suffix = "";
                    if (unexplored.length > 0) {
                        suffix = `\n\nQuer explorar outro tema? Ainda tenho: **${unexplored.slice(0, 3).join("**, **")}**...`;
                    }
                    return topic.deep + suffix;
                }
            }
            // No context — give the capabilities overview
            return conversational.capabilities.response;
        }

        // 5. Affirmations ("sim", "quero", etc.) — continue last topic
        if (matchesAny(text, conversational.affirmations.patterns)) {
            if (lastTopic && topics[lastTopic]) {
                const topic = topics[lastTopic];
                if (!askedTopics.has(lastTopic + "_deep")) {
                    askedTopics.add(lastTopic + "_deep");
                    conversationDepth++;
                    const unexplored = getUnexploredTopics();
                    let suffix = "";
                    if (unexplored.length > 0) {
                        suffix = `\n\nQuer explorar outro pilar? Posso falar sobre: **${unexplored.slice(0, 3).join("**, **")}**...`;
                    }
                    return topic.deep + suffix;
                } else {
                    const unexplored = getUnexploredTopics();
                    if (unexplored.length > 0) {
                        return `Já mergulhamos fundo em **${topic.name}**! Mas tenho mais munição. Quer que eu explique sobre **${unexplored.slice(0, 3).join("**, **")}**? Cada um é um pilar fundamental do meu funcionamento.`;
                    }
                    return "Já cobrimos todos os pilares principais! 🎓 Você agora entende a arquitetura completa do CROM-IA: compressão fractal, FUSE mmap, inferência sem GPU, entropia de Shannon, treinamento neural, privacidade total e edge computing. Quer revisitar algum tema ou testar o **Sanitizador de Dados** no portal?";
                }
            }
            return conversational.capabilities.response;
        }

        // 6. Topic-specific match
        const topicKey = findTopicMatch(text);
        if (topicKey) {
            lastTopic = topicKey;
            askedTopics.add(topicKey);
            conversationDepth = 0;
            const topic = topics[topicKey];
            return topic.intro + "\n\nQuer que eu aprofunde mais nesse tema? 🔬";
        }

        // 7. Fallback — but smart
        conversationDepth++;
        const r = conversational.unknown;
        return r[Math.floor(Math.random() * r.length)];
    }

    // --- UI Logic ---
    const API_URL = "http://localhost:5000";
    let backendOnline = false;

    // Check if real LLM backend is running
    async function checkBackend() {
        try {
            const res = await fetch(`${API_URL}/api/status`, { signal: AbortSignal.timeout(2000) });
            if (res.ok) {
                const data = await res.json();
                backendOnline = true;
                chatRam.textContent = `${data.rss_mb} MB`;
                // Update sidebar to show real model info
                const disclaimer = document.querySelector(".chat-disclaimer");
                if (disclaimer) disclaimer.textContent = `Modelo real: ${data.model} | Mmap: ON | GPU: 0 layers`;
                console.log("[CROM] Backend LLM real detectado:", data.model);
            }
        } catch {
            backendOnline = false;
            console.log("[CROM] Backend offline — usando modo simulacao educacional");
        }
    }
    checkBackend();
    setInterval(checkBackend, 10000);

    function formatMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/`(.*?)`/g, "<code>$1</code>")
            .replace(/\n\n/g, "</p><p>")
            .replace(/\n/g, "<br>");
    }

    function addMessage(content, isUser, meta) {
        const wrapper = document.createElement("div");
        wrapper.className = `msg ${isUser ? "msg-user" : "msg-bot"}`;

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = isUser ? "👤" : "🤖";

        const body = document.createElement("div");
        body.className = "msg-body";

        const bubble = document.createElement("div");
        bubble.className = "msg-content";
        bubble.innerHTML = `<p>${formatMarkdown(content)}</p>`;
        body.appendChild(bubble);

        if (!isUser) {
            const metaDiv = document.createElement("div");
            metaDiv.className = "msg-meta";
            if (meta && meta.real) {
                // Real LLM inference metadata
                metaDiv.innerHTML = `
                    <span class="meta-badge meta-speed">⚡ ${meta.latency_ms}ms real</span>
                    <span class="meta-badge meta-local">🔒 Local | ${meta.tokens} tokens</span>
                    <span class="meta-badge meta-fuse">💾 RSS: ${meta.rss_mb}MB</span>
                `;
            } else {
                metaDiv.innerHTML = `
                    <span class="meta-badge meta-speed">⚡ Modo Simulacao</span>
                    <span class="meta-badge meta-local">🔒 100% Local</span>
                    <span class="meta-badge meta-fuse">💾 Fallback</span>
                `;
            }
            body.appendChild(metaDiv);
        }

        wrapper.appendChild(avatar);
        wrapper.appendChild(body);
        messagesContainer.appendChild(wrapper);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function createStreamingMessage() {
        const wrapper = document.createElement("div");
        wrapper.className = "msg msg-bot";
        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = "🤖";
        const body = document.createElement("div");
        body.className = "msg-body";
        const bubble = document.createElement("div");
        bubble.className = "msg-content";
        const p = document.createElement("div"); // Using div to hold HTML
        bubble.appendChild(p);
        body.appendChild(bubble);
        wrapper.appendChild(avatar);
        wrapper.appendChild(body);
        messagesContainer.appendChild(wrapper);
        return {
            wrapper: wrapper,
            append: (fullText) => {
                p.innerHTML = formatMarkdown(fullText);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            },
            finish: (meta) => {
                const metaDiv = document.createElement("div");
                metaDiv.className = "msg-meta";
                if (meta && meta.real) {
                    metaDiv.innerHTML = `
                        <span class="meta-badge meta-speed">⚡ ${meta.latency_ms}ms real</span>
                        <span class="meta-badge meta-local">🧬 Modo Sub-Simbólico</span>
                        <span class="meta-badge meta-fuse">💾 RSS: ${meta.rss_mb}MB</span>
                    `;
                }
                body.appendChild(metaDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        };
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

    async function handleSend() {
        const text = input.value.trim();
        if (!text) return;

        addMessage(text, true);
        input.value = "";
        input.style.height = "auto";
        showTyping();

        if (backendOnline) {
            // === REAL LLM INFERENCE STREAMING ===
            try {
                const res = await fetch(`${API_URL}/api/chat/stream`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text }),
                });
                removeTyping();
                if (res.ok) {
                    const reader = res.body.getReader();
                    const decoder = new TextDecoder();
                    const streamMsg = createStreamingMessage();
                    let fullText = "";

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split("\n");
                        for (const line of lines) {
                            if (line.startsWith("data: ")) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (!data.done) {
                                        fullText += data.token;
                                        streamMsg.append(fullText);
                                    } else {
                                        chatRam.textContent = `${data.rss_mb} MB`;
                                        streamMsg.finish({
                                            real: true,
                                            latency_ms: data.latency_ms,
                                            rss_mb: data.rss_mb
                                        });
                                    }
                                } catch (err) {}
                            }
                        }
                    }
                } else {
                    const err = await res.json();
                    addMessage(`[Erro do servidor]: ${err.error}`, false);
                }
            } catch (e) {
                removeTyping();
                backendOnline = false;
                // Fallback to mock
                const response = buildResponse(text);
                addMessage(response, false);
            }
        } else {
            // === FALLBACK: MOCK RESPONSES ===
            const delay = 400 + Math.random() * 800;
            setTimeout(() => {
                removeTyping();
                const response = buildResponse(text);
                addMessage(response, false);
            }, delay);
        }
    }

    sendBtn.addEventListener("click", handleSend);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 120) + "px";
    });
});
