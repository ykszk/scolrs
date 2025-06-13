
// Get UI elements
const lineWidthSlider = document.getElementById('lineWidth');
const fontSizeSlider = document.getElementById('fontSize');
const lineWidthValue = document.getElementById('lineWidthValue');
const fontSizeValue = document.getElementById('fontSizeValue');
const radiusSlider = document.getElementById('radius');
const radiusValue = document.getElementById('radiusValue');
const opacitySlider = document.getElementById('opacity');
const opacityValue = document.getElementById('opacityValue');

function selectSvg() {
    // Select the SVG element in the popup content
    let svg = document.querySelector(".popup-content > svg")
    if (svg !== null) {
        return svg;
    }
    // If not found, try to select the SVG element in the main content
    return document.querySelector('svg');
}

// Update line width
function updateLineWidth(width) {
    const svgElement = selectSvg();
    for (const element of svgElement.querySelectorAll('line, polyline, polygon, path, rect, ellipse')) {
        element.style.strokeWidth = width / 10 + 'px';
    }
    lineWidthValue.textContent = width / 10 + 'px';
}

// Update font size
function updateFontSize(size) {
    const svgElement = selectSvg();
    for (const element of svgElement.querySelectorAll('text')) {
        element.style.fontSize = size + 'px';
    }
    for (const element of svgElement.querySelectorAll('tspan')) {
        element.style.fontSize = size / 2 + 'px';
    }

    fontSizeValue.textContent = size + 'px';
}

// Update radius for circles
function updateRadius(radius) {
    const svgElement = selectSvg();
    for (const element of svgElement.querySelectorAll('circle')) {
        element.setAttribute('r', radius / 10);
    }
    radiusValue.textContent = radius / 10 + 'px';
}

// Update opacity
function updateOpacity(value) {
    const svgElement = selectSvg();
    const opacity = value / 100;
    for (const element of svgElement.querySelectorAll('*')) {
        // ignore image
        if (element.tagName === 'image') continue;
        element.setAttribute('opacity', opacity);
    }
    opacityValue.textContent = value + '%';
}

// Add event listeners
lineWidthSlider.addEventListener('input', (e) => {
    updateLineWidth(e.target.value);
});

fontSizeSlider.addEventListener('input', (e) => {
    updateFontSize(e.target.value);
});

radiusSlider.addEventListener('input', (e) => {
    updateRadius(e.target.value);
});

opacitySlider.addEventListener('input', (e) => {
    updateOpacity(e.target.value);
});