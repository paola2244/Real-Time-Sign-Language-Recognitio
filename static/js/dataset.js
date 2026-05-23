/**
 * Dataset Collection Module
 * Manejo de recopilación de datos para entrenar el modelo
 */

let datasetVideo = null;
let datasetCanvas = null;
let datasetCanvasContext = null;
let datasetStream = null;
let datasetRunning = false;
let datasetInterval = null;

function initDataset() {
    datasetVideo = document.getElementById('datasetVideo');
    datasetCanvas = document.getElementById('datasetCanvas');
    datasetCanvasContext = datasetCanvas ? datasetCanvas.getContext('2d') : null;

    const startBtn = document.getElementById('startDatasetCameraBtn');
    const captureBtn = document.getElementById('captureDatasetBtn');
    const stopBtn = document.getElementById('stopDatasetCameraBtn');
    const refreshBtn = document.getElementById('refreshStatsBtn');

    if (startBtn) startBtn.addEventListener('click', startDatasetCamera);
    if (captureBtn) captureBtn.addEventListener('click', captureDatasetFrame);
    if (stopBtn) stopBtn.addEventListener('click', stopDatasetCamera);
    if (refreshBtn) refreshBtn.addEventListener('click', loadDatasetStats);

    console.log('Dataset module initialized');
}

async function startDatasetCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 320 },
                height: { ideal: 240 }
            }
        });

        datasetVideo.srcObject = stream;
        datasetStream = stream;

        datasetVideo.onloadedmetadata = () => {
            datasetVideo.play();
            datasetRunning = true;

            // Actualizar botones
            document.getElementById('startDatasetCameraBtn').classList.add('d-none');
            document.getElementById('captureDatasetBtn').classList.remove('d-none');
            document.getElementById('stopDatasetCameraBtn').classList.remove('d-none');

            showDatasetStatus('Cámara iniciada. Selecciona una letra y haz clic en "Capturar Foto"', 'info');
        };

    } catch (error) {
        console.error('Error accessing camera:', error);
        showDatasetStatus('No se puede acceder a la cámara', 'danger');
    }
}

function stopDatasetCamera() {
    if (datasetStream) {
        datasetStream.getTracks().forEach(track => track.stop());
    }

    datasetVideo.srcObject = null;
    datasetRunning = false;

    // Actualizar botones
    document.getElementById('startDatasetCameraBtn').classList.remove('d-none');
    document.getElementById('captureDatasetBtn').classList.add('d-none');
    document.getElementById('stopDatasetCameraBtn').classList.add('d-none');

    showDatasetStatus('Cámara detenida', 'warning');
}

async function captureDatasetFrame() {
    try {
        const letterSelect = document.getElementById('letterSelect');
        const letter = letterSelect.value;

        if (!letter) {
            showDatasetStatus('Por favor selecciona una letra', 'warning');
            return;
        }

        if (!datasetRunning || !datasetVideo.srcObject) {
            showDatasetStatus('La cámara no está iniciada', 'danger');
            return;
        }

        // Preparar canvas
        datasetCanvas.width = datasetVideo.videoWidth;
        datasetCanvas.height = datasetVideo.videoHeight;

        // Dibujar frame del video en canvas
        datasetCanvasContext.drawImage(datasetVideo, 0, 0, datasetCanvas.width, datasetCanvas.height);

        // Convertir canvas a blob
        datasetCanvas.toBlob(async (blob) => {
            try {
                const formData = new FormData();
                formData.append('image', blob, `${letter}.jpg`);
                formData.append('letter', letter);

                const response = await fetch('/api/collect-data', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    showDatasetStatus(
                        `✓ Foto guardada para "${letter}" (${result.images_count} total)`,
                        'success'
                    );
                    loadDatasetStats();
                } else {
                    showDatasetStatus(`Error: ${result.error}`, 'danger');
                }

            } catch (error) {
                console.error('Upload error:', error);
                showDatasetStatus('Error al guardar foto', 'danger');
            }
        }, 'image/jpeg', 0.95);

    } catch (error) {
        console.error('Capture error:', error);
        showDatasetStatus('Error al capturar foto', 'danger');
    }
}

async function loadDatasetStats() {
    try {
        const response = await fetch('/api/dataset-stats');
        const stats = await response.json();

        // Actualizar totales
        document.getElementById('totalDatasetImages').textContent = stats.total_images;
        document.getElementById('lettersCollected').textContent = stats.letters_collected;

        // Actualizar tabla de letras
        const statsDiv = document.getElementById('datasetLetterStats');
        if (stats.total_images === 0) {
            statsDiv.innerHTML = '<p class="text-muted">Sin datos aún. Empieza a capturar!</p>';
        } else {
            let html = '<div class="row">';
            for (const [letter, count] of Object.entries(stats.by_letter).sort()) {
                const percentage = Math.round((count / stats.total_images) * 100);
                html += `
                    <div class="col-6 mb-2">
                        <strong>${letter}</strong>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-info" style="width: ${percentage}%;">
                                ${count}
                            </div>
                        </div>
                    </div>
                `;
            }
            html += '</div>';
            statsDiv.innerHTML = html;
        }

    } catch (error) {
        console.error('Error loading stats:', error);
        document.getElementById('datasetLetterStats').innerHTML =
            '<p class="text-danger">Error al cargar estadísticas</p>';
    }
}

function showDatasetStatus(message, type = 'info') {
    const statusDiv = document.getElementById('datasetStatus');
    if (statusDiv) {
        statusDiv.className = `alert alert-${type}`;
        statusDiv.innerHTML = message;
        statusDiv.classList.remove('d-none');

        // Auto-ocultar después de 5 segundos si es success
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.classList.add('d-none');
            }, 5000);
        }
    }
}

// Cargar estadísticas cuando se abre la sección
document.addEventListener('DOMContentLoaded', function() {
    // Esperar a que el HTML esté listo
    setTimeout(() => {
        initDataset();
        loadDatasetStats();
    }, 100);
});

// Inicializar si el DOM ya está listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDataset);
} else {
    initDataset();
}
