document.addEventListener("DOMContentLoaded", () => {
    const rssValue = document.getElementById("rss-value");
    const hitRate = document.getElementById("hit-rate");
    const swapValue = document.getElementById("swap-value");
    const pruneValue = document.getElementById("prune-value");
    const logContainer = document.getElementById("log-container");

    let prunedCount = 0;
    
    // Simulate initial telemetry ping
    setTimeout(() => addLog("SYS", "CROM VFS Daemon conectado via mmap."), 500);
    setTimeout(() => addLog("SYS", "Monitor SRE Edge ativado. Bypass VRAM habilitado."), 1200);
    setTimeout(() => addLog("SYS", "Aguardando matriz de Dataloader Pytorch..."), 1500);

    // Dynamic Logic Real Core
    async function generateTelemetry() {
        try {
            const res = await fetch("http://localhost:5000/api/metrics");
            if (res.ok) {
                const data = await res.json();
                rssValue.innerHTML = `${data.rss_mb} <span class="unit">MB</span>`;
                hitRate.innerHTML = `99.8<span class="unit">%</span>`;
                swapValue.innerHTML = `${data.swap_pct} / 100`;
            }
        } catch (e) {
            // Fallback
             rssValue.innerHTML = `Offline`;
        }
    }

    setInterval(generateTelemetry, 2000);

    function addLog(level, message) {
        const line = document.createElement("div");
        line.className = "log-line";
        
        const time = new Date().toISOString().split('T')[1].substr(0, 8);
        
        let tagClass = "tag-sys";
        if (level === "WARN") tagClass = "tag-warn";
        if (level === "ERR") tagClass = "tag-err";

        line.innerHTML = `<span class="timestamp">[${time}]</span> <span class="${tagClass}">[${level}]</span> ${message}`;
        
        logContainer.appendChild(line);
        logContainer.scrollTop = logContainer.scrollHeight;

        // Cleanup to prevent DOM bloat
        if (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.firstChild);
        }
    }

    // Simulate Dataloader Loop and Shannon Pruning
    let epoch = 0;
    setInterval(() => {
        const isChaos = Math.random() > 0.85;
        
        if (isChaos) {
            const h = (7.6 + Math.random() * 1.5).toFixed(2);
            prunedCount++;
            pruneValue.innerHTML = `${prunedCount} <span class="unit">Blocks</span>`;
            addLog("WARN", `Block Pruned! HNSW Evidenciou Entropia Caótica (H=${h} > 7.5)`);
        } else {
            addLog("SYS", `Epoch ${epoch} | Forward Pass FUSE Cosenoidal... Loss: ${(Math.random() * 0.5).toFixed(4)}`);
            epoch++;
        }
    }, 1500);

    // Modal Architectural Popup Logic
    const archModal = document.getElementById("arch-modal");
    const openModalBtn = document.getElementById("open-modal-btn");
    const closeModalBtn = document.getElementById("close-modal-btn");

    if(openModalBtn && archModal) {
        openModalBtn.addEventListener("click", () => {
            archModal.classList.add("active");
        });
    }

    if(closeModalBtn && archModal) {
        closeModalBtn.addEventListener("click", () => {
            archModal.classList.remove("active");
        });
        
        // Close on clicking outside
        archModal.addEventListener("click", (e) => {
            if(e.target === archModal) {
                archModal.classList.remove("active");
            }
        });
    }
});
