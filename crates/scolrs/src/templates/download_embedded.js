function base64ToUint8Array(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

async function decompressGzip(uint8Array) {
    if (typeof DecompressionStream === "function") {
        const ds = new DecompressionStream("gzip");
        const stream = new Response(uint8Array).body.pipeThrough(ds);
        const decompressed = await new Response(stream).arrayBuffer();
        return new Uint8Array(decompressed);
    } else {
        // fallback: require pako.js or similar
        throw new Error("Gzip decompression not supported in this browser.");
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

function createDownloadButtons() {
    let svg = document.querySelector(".popup-content > svg")
    if (svg === null) {
        svg = document.querySelector('svg');
    }
    const scripts = svg.querySelectorAll("script[id^='embedded-']");
    const container = document.createElement("ul");
    container.id = "embedded-downloads";
    scripts.forEach(async (script) => {
        const filename = script.getAttribute("data-filename") || "file";
        const encoding = script.getAttribute("encoding");
        const type = script.getAttribute("type");
        let content = script.textContent.trim();
        let uint8 = base64ToUint8Array(content);

        if (type && type.endsWith("+gzip")) {
            try {
                uint8 = await decompressGzip(uint8);
            } catch (e) {
                alert("Failed to decompress " + filename + ": " + e.message);
                return;
            }
        }

        let blob;
        if (type && type.startsWith("text/")) {
            blob = new Blob([uint8], { type: type.replace("+gzip", "") });
        } else {
            blob = new Blob([uint8], { type: type ? type.replace("+gzip", "") : "application/octet-stream" });
        }

        const btn = document.createElement("button");
        btn.textContent = filename;
        btn.onclick = () => downloadBlob(blob, filename);
        const li = document.createElement("li");
        li.appendChild(btn);
        container.appendChild(li);
    });
    const details = document.querySelector("details#download-details");
    if (scripts.length === 0) {
        if (details) details.style.display = "none";
    } else {
        if (details) {
            // Clear previous container
            const existing = document.getElementById("embedded-downloads");
            if (existing) details.removeChild(existing);
            details.appendChild(container);
        };
    }
}