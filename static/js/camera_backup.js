/**
 * Camera Control Module
 * Manejo de webcam en tiempo real con predicción
 */

let videoElement = null;
let canvasElement = null;
let canvasContext = null;
let stream = null;
let isRunning = false;
let frameCount = 0;
let startTime = null;
let predictionInterval = null;
let frameRate = 15; // frames por segundo para predicción
let currentDetectedLetter = null; // Letra detectada actualmente
let currentDetectedConfidence = 0; // Confianza actual
let savedWords = []; // Historial de palabras guardadas

function initCamera() {
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');
    canvasContext = canvasElement.getContext('2d');

    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');

    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);

    // Controles adicionales
    const clearWordBtn = document.getElementById('clearWordBtn');
    const addWordBtn = document.getElementById('addWordBtn');
    const acceptLetterBtn = document.getElementById('acceptLetterBtn');
    const undoLetterBtn = document.getElementById('undoLetterBtn');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const showLandmarksCheckbox = document.getElementById('showLandmarks');

    if (clearWordBtn) clearWordBtn.addEventListener('click', clearCurrentWord);
    if (addWordBtn) addWordBtn.addEventListener('click', saveCurrentWord);
    if (acceptLetterBtn) acceptLetterBtn.addEventListener('click', acceptCurrentLetter);
    if (undoLetterBtn) undoLetterBtn.addEventListener('click', undoLetter);
    if (thresholdSlider) thresholdSlider.addEventListener('change', updateThreshold);
    if (showLandmarksCheckbox) showLandmarksCheckbox.addEventListener('change', updateShowLandmarks);

    // Cargar palabras guardadas
    loadSavedWords();

    // Actualizar valor del slider
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', function() {
            document.getElementById('thresholdValue').textContent = this.value;
        });
    }

    console.log('Camera module initialized');
}

async function startCamera() {
    try {
        // Solicitar acceso a la cámara
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        // Asignar stream al video element
        videoElement.srcObject = stream;

        // Esperar a que el video esté listo
        videoElement.onloadedmetadata = () => {
            videoElement.play();

            // Configurar canvas
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;

            // Iniciar captura de frames
            isRunning = true;
            startTime = Date.now();
            frameCount = 0;

            // Actualizar botones
            document.getElementById('startCameraBtn').classList.add('d-none');
            document.getElementById('stopCameraBtn').classList.remove('d-none');
            document.getElementById('videoError').classList.add('d-none');

            // Iniciar predicción
            startPredictionLoop();
        };

    } catch (error) {
        console.error('Error accessing camera:', error);
        document.getElementById('videoError').classList.remove('d-none');
        showNotification('No se puede acceder a la cámara', 'error');
    }
}

function stopCamera() {
    // Detener stream
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    isRunning = false;

    // Limpiar video
    videoElement.srcObject = null;

    // Limpiar canvas
    canvasContext.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Detener predicción
    if (predictionInterval) {
        clearInterval(predictionInterval);
    }

    // Actualizar botones
    document.getElementById('startCameraBtn').classList.remove('d-none');
    document.getElementById('stopCameraBtn').classList.add('d-none');

    console.log('Camera stopped');
}

function startPredictionLoop() {
    const interval = 1000 / frameRate; // ms entre frames

    predictionInterval = setInterval(async () => {
        if (!isRunning || !videoElement.srcObject) {
            return;
        }

        try {
            // Capturar frame del video
            canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

            // Enviar a predicción
            const result = await apiPredictFrame(canvasElement);

            if (result.success) {
                // Actualizar UI con resultado
                updatePredictionUI(result);
                frameCount++;
            }

            // Actualizar estadísticas
            updateStats();

        } catch (error) {
            console.error('Prediction error:', error);
        }
    }, interval);
}

function updatePredictionUI(result) {
    // Actualizar letra detectada en panel principal
    const letterElement = document.getElementById('currentLetter');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');

    if (letterElement) {
        letterElement.textContent = result.letter !== 'N/A' ? result.letter : '-';
    }

    const confidence = (result.confidence * 100).toFixed(0);
    if (confidenceBar) {
        confidenceBar.style.width = confidence + '%';
    }
    if (confidenceText) {
        confidenceText.textContent = `Confianza: ${confidence}%`;
    }

    // Actualizar letra detectada en sección de "Palabra Actual"
    const detectedLetterDisplay = document.getElementById('detectedLetterDisplay');
    const detectedConfidenceDisplay = document.getElementById('detectedConfidenceDisplay');

    if (result.letter && result.letter !== 'N/A') {
        if (detectedLetterDisplay) {
            detectedLetterDisplay.textContent = result.letter;
        }
        if (detectedConfidenceDisplay) {
            detectedConfidenceDisplay.textContent = confidence + '%';
        }
        currentDetectedLetter = result.letter;
        currentDetectedConfidence = result.confidence;
    } else {
        if (detectedLetterDisplay) {
            detectedLetterDisplay.textContent = '-';
        }
        if (detectedConfidenceDisplay) {
            detectedConfidenceDisplay.textContent = '0%';
        }
        currentDetectedLetter = null;
        currentDetectedConfidence = 0;
    }

    // Dibujar landmarks si existen
    if (result.landmarks_image && document.getElementById('showLandmarks').checked) {
        drawLandmarks(result.landmarks_image);
    }
}

function drawLandmarks(base64Image) {
    if (!base64Image) return;

    const img = new Image();
    img.onload = () => {
        canvasContext.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
    };
    img.src = 'data:image/png;base64,' + base64Image;
}

function updateStats() {
    const statsTotal = document.getElementById('statsTotal');
    const statsDuration = document.getElementById('statsDuration');

    if (statsTotal) {
        statsTotal.textContent = frameCount;
    }

    if (statsDuration && startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        statsDuration.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
}

async function acceptCurrentLetter() {
    if (!currentDetectedLetter) {
        showNotification('No hay letra detectada', 'warning');
        return;
    }

    try {
        const result = await addLetter(currentDetectedLetter);
        if (result.success) {
            const currentWordElement = document.getElementById('currentWord');
            if (currentWordElement) {
                currentWordElement.value = result.current_word;
            }
            showNotification(`Letra "${currentDetectedLetter}" agregada`, 'success');
        }
    } catch (error) {
        console.error('Error adding letter:', error);
        showNotification('Error al agregar letra', 'error');
    }
}

async function undoLetter() {
    try {
        const result = await removeLetter();
        if (result.success) {
            const currentWordElement = document.getElementById('currentWord');
            if (currentWordElement) {
                currentWordElement.value = result.current_word;
            }
            showNotification('Letra removida', 'info');
        }
    } catch (error) {
        console.error('Error removing letter:', error);
        showNotification('Error al remover letra', 'error');
    }
}

async function clearCurrentWord() {
    if (confirm('¿Limpiar palabra actual?')) {
        try {
            const result = await resetWord();
            document.getElementById('currentWord').value = '';
            showNotification('Palabra limpiada', 'success');
        } catch (error) {
            console.error('Error clearing word:', error);
            showNotification('Error al limpiar palabra', 'error');
        }
    }
}

async function saveCurrentWord() {
    try {
        const currentWord = document.getElementById('currentWord').value;
        if (!currentWord) {
            showNotification('No hay palabra para guardar', 'warning');
            return;
        }

        // Guardar palabra en historial local
        savedWords.push({
            word: currentWord,
            timestamp: new Date().toLocaleTimeString('es-ES')
        });

        showNotification(`Palabra guardada: "${currentWord}"`, 'success');

        // Mostrar en historial
        updateWordsDisplay();

        // Guardar en localStorage para persistencia
        localStorage.setItem('savedWords', JSON.stringify(savedWords));

        // Resetear palabra
        await resetWord();
        document.getElementById('currentWord').value = '';

    } catch (error) {
        console.error('Error saving word:', error);
        showNotification('Error al guardar palabra', 'error');
    }
}

function updateWordsDisplay() {
    const noWordsMessage = document.getElementById('noWordsMessage');
    const wordsList = document.getElementById('wordsHistoryList');

    if (savedWords.length === 0) {
        noWordsMessage.style.display = 'block';
        wordsList.innerHTML = '';
    } else {
        noWordsMessage.style.display = 'none';
        let html = '';
        savedWords.forEach((item, index) => {
            html += `
                <div class="badge bg-primary me-1 mb-1">
                    ${index + 1}. ${item.word}
                    <small class="ms-1">${item.timestamp}</small>
                </div>
            `;
        });
        wordsList.innerHTML = html;
    }
}

function loadSavedWords() {
    // Cargar palabras guardadas del localStorage (opcional)
    const stored = localStorage.getItem('savedWords');
    if (stored) {
        try {
            savedWords = JSON.parse(stored);
            updateWordsDisplay();
        } catch (e) {
            savedWords = [];
        }
    }
}

async function updateThreshold() {
    const threshold = document.getElementById('thresholdSlider').value / 100;
    try {
        // Actualizar en el backend (si es necesario)
        console.log('Threshold updated to:', threshold);
    } catch (error) {
        console.error('Error updating threshold:', error);
    }
}

function updateShowLandmarks() {
    const show = document.getElementById('showLandmarks').checked;
    console.log('Show landmarks:', show);
    // Los landmarks se dibujan automáticamente si el checkbox está marcado
}

// Inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCamera);
} else {
    initCamera();
}
