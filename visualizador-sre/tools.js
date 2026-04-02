document.addEventListener("DOMContentLoaded", () => {
    const inputArea = document.getElementById("tools-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    const resultsContainer = document.getElementById("results-container");
    const statsSection = document.getElementById("stats-section");

    // Shannon Entropy calculator (same logic as the Python crom_dataloader.py)
    function calcShannonEntropy(text) {
        if (!text || text.length === 0) return 0;
        
        const encoder = new TextEncoder();
        const bytes = encoder.encode(text);
        const freq = {};
        
        for (const b of bytes) {
            freq[b] = (freq[b] || 0) + 1;
        }
        
        const size = bytes.length;
        let entropy = 0;
        for (const count of Object.values(freq)) {
            const p = count / size;
            entropy -= p * Math.log2(p);
        }
        
        return entropy;
    }

    function classifyEntropy(h) {
        if (h <= 5.0) return { level: "clean", label: "Aprovado ✅", desc: "Texto coerente e limpo. Ideal para treinamento de IA.", color: "#10b981" };
        if (h <= 7.5) return { level: "warn", label: "Suspeito ⚠️", desc: "Entropia mista. Pode conter dados semi-estruturados ou código.", color: "#f59e0b" };
        return { level: "pruned", label: "Podado 🚫", desc: "Entropia caótica! O CROM descartaria este bloco antes do treinamento.", color: "#ef4444" };
    }

    function analyze() {
        const raw = inputArea.value.trim();
        if (!raw) return;

        // Split into paragraphs (non-empty lines)
        const blocks = raw.split(/\n\s*\n|\n/).filter(b => b.trim().length > 0);

        resultsContainer.innerHTML = "";
        resultsContainer.className = "";

        let totalH = 0;
        let counts = { clean: 0, warn: 0, pruned: 0 };

        blocks.forEach((block, i) => {
            const h = calcShannonEntropy(block.trim());
            const cls = classifyEntropy(h);
            totalH += h;
            counts[cls.level]++;

            const card = document.createElement("div");
            card.className = `result-card result-${cls.level}`;
            card.style.animationDelay = `${i * 0.08}s`;

            // Show a preview of the text (first 120 chars)
            const preview = block.trim().length > 120 
                ? block.trim().substring(0, 120) + "…" 
                : block.trim();

            card.innerHTML = `
                <div class="result-header">
                    <span class="result-index">Bloco ${i + 1}</span>
                    <span class="result-badge" style="background:${cls.color}20; color:${cls.color}; border:1px solid ${cls.color}40">${cls.label}</span>
                </div>
                <div class="result-text"><code>${escapeHtml(preview)}</code></div>
                <div class="result-bar-wrap">
                    <div class="result-bar" style="width:${(h / 8) * 100}%; background:${cls.color}"></div>
                </div>
                <div class="result-meta">
                    <span><strong>H = ${h.toFixed(4)}</strong> bits/byte</span>
                    <span>${cls.desc}</span>
                </div>
            `;

            resultsContainer.appendChild(card);
        });

        // Update stats
        statsSection.style.display = "flex";
        document.getElementById("stat-total").textContent = blocks.length;
        document.getElementById("stat-clean").textContent = counts.clean;
        document.getElementById("stat-warn").textContent = counts.warn;
        document.getElementById("stat-pruned").textContent = counts.pruned;
        document.getElementById("stat-avg").textContent = (totalH / blocks.length).toFixed(2);
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    analyzeBtn.addEventListener("click", analyze);

    // Allow ctrl+enter to analyze
    inputArea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.ctrlKey) {
            e.preventDefault();
            analyze();
        }
    });
});
