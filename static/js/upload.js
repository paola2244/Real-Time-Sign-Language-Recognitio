/**
 * Upload and History Module
 * Manejo de carga de fotos e historial de predicciones
 */

let selectedFile = null;

function initUpload() {
    // Drag and drop
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadError = document.getElementById('uploadError');

    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            handleFiles(files);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeImage);
    }

    console.log('Upload module initialized');
}

function handleFiles(files) {
    const fileInput = document.getElementById('fileInput');
    const uploadError = document.getElementById('uploadError');

    if (files.length === 0) {
        return;
    }

    const file = files[0];

    // Validar tipo de archivo
    if (!file.type.startsWith('image/')) {
        uploadError.textContent = 'Por favor selecciona un archivo de imagen válido';
        uploadError.classList.remove('d-none');
        return;
    }

    // Validar tamaño (200MB)
    if (file.size > 200 * 1024 * 1024) {
        uploadError.textContent = 'El archivo es demasiado grande (máximo 200MB)';
        uploadError.classList.remove('d-none');
        return;
    }

    uploadError.classList.add('d-none');
    selectedFile = file;

    // Mostrar preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImage = document.getElementById('previewImage');
        const noPreview = document.getElementById('noPreview');

        if (previewImage) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        }
        if (noPreview) {
            noPreview.style.display = 'none';
        }

        // Activar botón de análisis
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
    };

    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!selectedFile) {
        showNotification('Por favor selecciona una imagen', 'warning');
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analizando...';

    try {
        const result = await apiUploadFile('/api/upload', selectedFile);

        if (result.success) {
            displayResults(result);
            showNotification('Análisis completado', 'success');
        } else {
            showNotification(result.error || 'Error al analizar imagen', 'error');
        }

    } catch (error) {
        console.error('Error analyzing image:', error);
        showNotification('Error al analizar imagen', 'error');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-magic"></i> Analizar Imagen';
    }
}

function displayResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultLetter = document.getElementById('resultLetter');
    const resultConfidenceBar = document.getElementById('resultConfidenceBar');
    const resultConfidenceText = document.getElementById('resultConfidenceText');
    const resultLandmarks = document.getElementById('resultLandmarks');

    if (resultsContainer) resultsContainer.classList.remove('d-none');

    if (resultLetter) {
        resultLetter.textContent = result.letter !== 'N/A' ? result.letter : '-';
    }

    const confidence = (result.confidence * 100).toFixed(0);
    if (resultConfidenceBar) {
        resultConfidenceBar.style.width = confidence + '%';
    }
    if (resultConfidenceText) {
        resultConfidenceText.textContent = `Confianza: ${confidence}%`;
    }

    if (resultLandmarks && result.landmarks_image) {
        resultLandmarks.src = 'data:image/png;base64,' + result.landmarks_image;
    }
}

// ==================== HISTORY ====================

function initHistory() {
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', handleClearHistory);
    }

    if (downloadCsvBtn) {
        downloadCsvBtn.addEventListener('click', downloadCSV);
    }

    console.log('History module initialized');
}

async function loadHistory() {
    try {
        const data = await getHistory();

        // Actualizar estadísticas
        updateHistoryStats(data.stats);

        // Actualizar tabla
        populateHistoryTable(data.predictions);

    } catch (error) {
        console.error('Error loading history:', error);
        showNotification('Error al cargar historial', 'error');
    }
}

function updateHistoryStats(stats) {
    const totalElement = document.getElementById('historyTotal');
    const uniqueElement = document.getElementById('historyUnique');
    const avgElement = document.getElementById('historyAvg');

    if (totalElement) totalElement.textContent = stats.total || 0;
    if (uniqueElement) uniqueElement.textContent = stats.unique_letters || 0;
    if (avgElement) avgElement.textContent = ((stats.avg_confidence || 0) * 100).toFixed(0) + '%';
}

function populateHistoryTable(predictions) {
    const tableBody = document.getElementById('historyTableBody');

    if (!tableBody) return;

    // Limpiar tabla
    tableBody.innerHTML = '';

    if (predictions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No hay predicciones</td></tr>';
        return;
    }

    // Llenar tabla
    predictions.forEach((pred, index) => {
        const row = document.createElement('tr');

        const timestamp = new Date(pred.timestamp);
        const formattedTime = timestamp.toLocaleString('es-ES');

        row.innerHTML = `
            <td><strong>${pred.letter}</strong></td>
            <td>
                <div class="progress" style="height: 25px;">
                    <div class="progress-bar bg-success" role="progressbar"
                         style="width: ${(pred.confidence * 100).toFixed(0)}%;">
                        ${(pred.confidence * 100).toFixed(0)}%
                    </div>
                </div>
            </td>
            <td>${pred.word_context || '-'}</td>
            <td><small>${formattedTime}</small></td>
        `;

        tableBody.appendChild(row);
    });
}

async function handleClearHistory() {
    if (!confirm('¿Estás seguro de que quieres borrar todo el historial? Esta acción no se puede deshacer.')) {
        return;
    }

    try {
        await clearHistory();
        loadHistory();
        showNotification('Historial limpiado', 'success');
    } catch (error) {
        console.error('Error clearing history:', error);
        showNotification('Error al limpiar historial', 'error');
    }
}

function downloadCSV() {
    // Crear link de descarga
    const link = document.createElement('a');
    link.href = '/api/history/csv';
    link.download = `predictions_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUpload);
    document.addEventListener('DOMContentLoaded', initHistory);
} else {
    initUpload();
    initHistory();
}
