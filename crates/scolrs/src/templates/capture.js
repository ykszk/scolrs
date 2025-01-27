function svgToImage(svgElement, format, scale, name_stem, quality = 0.85) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const dpr = window.devicePixelRatio || 1;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const svgRect = svgElement.getBoundingClientRect(); // TODO: Returned value is slightly bigger than actual size when used in `catalog` command?
    canvas.width = svgRect.width * scale * dpr;
    canvas.height = svgRect.height * scale * dpr;
    ctx.scale(dpr * scale, dpr * scale);

    const svgString = new XMLSerializer().serializeToString(svgElement);
    const svg = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });

    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(svg);
        const img = new Image();

        img.onload = () => {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            try {
                const mimeType = format === 'jpeg' ? 'image/jpeg' : 'image/png';
                const imageData = canvas.toDataURL(mimeType, quality);
                URL.revokeObjectURL(url);

                const downloadLink = document.createElement('a');
                downloadLink.href = imageData;
                downloadLink.download = name_stem + '.' + format;
                downloadLink.click();

                resolve(imageData);
            } catch (error) {
                reject('Error converting to image: ' + error);
            }
        };

        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject('Error loading SVG');
        };

        img.src = url;
    });
}

function convertSvg(svgElement, format, scale, name_stem = 'image') {
    svgToImage(svgElement, format, scale, name_stem)
        .then(() => {
            console.log('Converted SVG to PNG');
        })
        .catch(error => {
            console.error(error);
        });
}