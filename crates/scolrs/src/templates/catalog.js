// Function to initialize magnifying functionality
function initializeMagnifier() {
    const overlay = document.querySelector('.popup-overlay');
    const popupContent = overlay.querySelector('.popup-content');
    const closeBtn = overlay.querySelector('.close-btn');
    const captureBtn = overlay.querySelector('#capture-btn');
    const toggleAllBtn = overlay.querySelector('#toggle-all-btn');
    const imageTitle = overlay.querySelector('#image-title');

    // Add click handlers to all magnify buttons
    document.querySelectorAll('.magnify-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Find the associated SVG
            const container = btn.closest('.image-container');
            popupContent.dataset.id = container.id || "image";
            imageTitle.innerHTML = container.id || "image";
            const sourceSvg = container.querySelector('svg');

            // Remove any previous SVG from popup
            const previousSvg = popupContent.querySelector('svg');
            if (previousSvg) {
                previousSvg.remove();
            }

            if (sourceSvg) {
                // Clone the SVG and its content
                const clonedSvg = sourceSvg.cloneNode(true);

                // Preserve original viewBox if it exists
                const originalViewBox = sourceSvg.getAttribute('viewBox');
                if (originalViewBox) {
                    clonedSvg.setAttribute('viewBox', originalViewBox);
                }

                // Make SVG responsive while maintaining aspect ratio
                clonedSvg.style.width = '100%';
                clonedSvg.style.height = 'auto';

                // Add the cloned SVG to popup
                popupContent.insertBefore(clonedSvg, closeBtn);

                // Update the checkboxes
                const checkboxes = overlay.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(cb => {
                    let svg_id = cb.dataset.id;
                    const elm = popupContent.querySelector("#" + svg_id);
                    if (elm) {
                        let li = overlay.querySelector(`li#li_${svg_id}`);
                        if (li) {
                            li.style.display = '';
                        }
                        let visibility = elm.getAttribute('visibility');
                        if (visibility) {
                            cb.checked = visibility === "visible";
                        } else {
                            cb.checked = false;
                        }
                    } else {
                        let li = overlay.querySelector(`li#li_${svg_id}`);
                        if (li) {
                            li.style.display = 'none';
                        }
                    }
                });

                // Show overlay
                overlay.style.display = 'flex';
            }
        });
    });

    // Set event listeners to checkboxes
    overlay.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', () => {
            let svg_id = cb.dataset.id;
            console.log("Change visibility", svg_id);
            let visibility = cb.checked ? "visible" : "hidden";
            const elm = popupContent.querySelector("#" + svg_id);
            if (elm) {
                elm.style.visibility = visibility;
            } else {
                console.error(`Element with id ${svg_id} not found`);
            }
        });
    });

    // Close popup when clicking close button
    closeBtn.addEventListener('click', () => {
        overlay.style.display = 'none';
    });

    // Close popup when clicking outside the image
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            overlay.style.display = 'none';
        }
    });

    // Handle escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.style.display === 'flex') {
            overlay.style.display = 'none';
        }
    });

    let show_all = false;
    toggleAllBtn.addEventListener('click', () => {
        const checkboxes = overlay.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => {
            cb.checked = show_all;
            cb.dispatchEvent(new Event('change'));
        });
        show_all = !show_all;
        toggleAllBtn.innerHTML = show_all ? "Show All" : "Hide All";
    });

    captureBtn.addEventListener('click', () => {
        const sourceSvg = popupContent.querySelector('svg');
        if (!sourceSvg) {
            console.error('No SVG found to capture');
            return;
        }
        const format = overlay.querySelector('#select-format').value;
        const scale = parseFloat(overlay.querySelector('#select-scale').value);
        const name_stem = popupContent.dataset.id || 'image';
        convertSvg(sourceSvg, format, scale, name_stem);
    });

}

// dummy function to toggle visibility of elements
function toggle_visibility(cb) {

}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', initializeMagnifier);