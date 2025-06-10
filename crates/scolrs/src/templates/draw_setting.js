
// Get UI elements
const lineWidthSlider = document.getElementById('lineWidth');
const fontSizeSlider = document.getElementById('fontSize');
const lineWidthValue = document.getElementById('lineWidthValue');
const fontSizeValue = document.getElementById('fontSizeValue');
const radiusSlider = document.getElementById('radius');
const radiusValue = document.getElementById('radiusValue');
const opacitySlider = document.getElementById('opacity');
const opacityValue = document.getElementById('opacityValue');

// Update line width
function updateLineWidth(width) {
    const svgElement = document.querySelector('svg');
    for (const element of svgElement.querySelectorAll('line, polyline, polygon, path, rect, ellipse')) {
        element.style.strokeWidth = width / 10 + 'px';
    }
    lineWidthValue.textContent = width / 10 + 'px';
}

// Update font size
function updateFontSize(size) {
    const svgElement = document.querySelector('svg');
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
    const svgElement = document.querySelector('svg');
    for (const element of svgElement.querySelectorAll('circle')) {
        element.setAttribute('r', radius / 10);
    }
    radiusValue.textContent = radius / 10 + 'px';
}

// Update opacity
function updateOpacity(value) {
    const svgElement = document.querySelector('svg');
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