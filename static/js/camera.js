/**
 * Camera Control Module
 * Webcam startup, prediction loop and manual word controls.
 */

let videoElement = null;
let canvasElement = null;
let canvasContext = null;
let stream = null;
let isInitialized = false;
let isStarting = false;
let isRunning = false;
let isPredicting = false;
let frameCount = 0;
let startTime = null;
let predictionInterval = null;
let statsInterval = null;
let currentRunId = 0;
let currentPredictionController = null;
let frameRate = 5;

// --- Variables para confirmación automática de letras ---
let stableLetterCount = 0;
let lastStableLetter = null;
const FRAMES_TO_CONFIRM = 9; // frames seguidos para confirmar una letra (~1.8s a 5fps)
const SPECIAL_SIGNS = ['BORRAR', 'ESPACIO', 'ESCUCHAR'];

function getConfidenceThreshold() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    if (!thresholdSlider) {
        return 0.75;
    }
    return Math.max(0, Math.min(1, Number(thresholdSlider.value) / 100));
}

function initCamera() {
    if (isInitialized) {
        return;
    }

    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');

    if (!videoElement || !canvasElement) {
        console.warn('Camera elements were not found.');
        return;
    }

    canvasContext = canvasElement.getContext('2d', { willReadFrequently: true });

    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');
    const acceptBtn = document.getElementById('acceptLetterBtn');
    const undoBtn = document.getElementById('undoLetterBtn');
    const clearBtn = document.getElementById('clearWordBtn');
    const saveBtn = document.getElementById('addWordBtn');
    const thresholdSlider = document.getElementById('thresholdSlider');

    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
    if (acceptBtn) acceptBtn.addEventListener('click', acceptLetter);
    if (undoBtn) undoBtn.addEventListener('click', undoLetter);
    if (clearBtn) clearBtn.addEventListener('click', clearWord);
    if (saveBtn) saveBtn.addEventListener('click', saveWord);
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', () => {
            const value = document.getElementById('thresholdValue');
            if (value) value.textContent = thresholdSlider.value;
            stableLetterCount = 0;
            lastStableLetter = null;
        });
    }

    videoElement.setAttribute('playsinline', 'true');
    videoElement.muted = true;
    isInitialized = true;
    console.log('Camera module initialized');
}

async function startCamera() {
    if (isStarting || isRunning) {
        return;
    }

    isStarting = true;
    setStartButtonState(true);
    hideCameraError();

    try {
        stopPredictionLoop();
        stopStream();

        console.log('Starting camera...');
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });

        videoElement.srcObject = stream;
        await waitForVideoReady(videoElement);
        await videoElement.play();

        canvasElement.width = videoElement.videoWidth || 640;
        canvasElement.height = videoElement.videoHeight || 480;

        currentRunId += 1;
        isRunning = true;
        isPredicting = false;
        startTime = Date.now();
        frameCount = 0;
        window.currentLetter = null;
        stableLetterCount = 0;
        lastStableLetter = null;

        setCameraButtons(true);
        updateStats();
        startStatsTimer();
        startPredictionLoop(currentRunId);
        console.log('Camera started');
    } catch (error) {
        console.error('Camera error:', error);
        showCameraError('No se puede acceder a la camara. Revisa permisos o si otra app la esta usando.');
        stopCamera();
    } finally {
        isStarting = false;
        setStartButtonState(false);
    }
}

function stopCamera() {
    console.log('Stopping camera...');
    currentRunId += 1;
    isStarting = false;
    isRunning = false;
    isPredicting = false;
    stableLetterCount = 0;
    lastStableLetter = null;

    stopPredictionLoop();
    stopStatsTimer();
    abortCurrentPrediction();
    stopStream();

    if (videoElement) {
        videoElement.pause();
        videoElement.srcObject = null;
    }

    if (canvasContext && canvasElement) {
        canvasContext.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }

    window.currentLetter = null;
    setCameraButtons(false);
}

function waitForVideoReady(video) {
    if (video.readyState >= 2 && video.videoWidth > 0) {
        return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
            cleanup();
            reject(new Error('Timed out waiting for video metadata'));
        }, 8000);

        const onReady = () => {
            if (video.videoWidth > 0) {
                cleanup();
                resolve();
            }
        };

        const onError = () => {
            cleanup();
            reject(new Error('Video element failed to load camera stream'));
        };

        const cleanup = () => {
            window.clearTimeout(timeout);
            video.removeEventListener('loadedmetadata', onReady);
            video.removeEventListener('loadeddata', onReady);
            video.removeEventListener('canplay', onReady);
            video.removeEventListener('error', onError);
        };

        video.addEventListener('loadedmetadata', onReady);
        video.addEventListener('loadeddata', onReady);
        video.addEventListener('canplay', onReady);
        video.addEventListener('error', onError);
    });
}

function startPredictionLoop(runId) {
    stopPredictionLoop();
    console.log('Starting prediction loop...');

    const interval = Math.max(150, Math.floor(1000 / frameRate));
    predictionInterval = window.setInterval(() => {
        predictCurrentFrame(runId);
    }, interval);

    predictCurrentFrame(runId);
}

async function predictCurrentFrame(runId) {
    if (!isRunning || runId !== currentRunId || isPredicting || !videoElement.srcObject) {
        return;
    }

    if (videoElement.readyState < 2 || canvasElement.width === 0 || canvasElement.height === 0) {
        return;
    }

    isPredicting = true;
    currentPredictionController = new AbortController();
    const predictionTimeout = window.setTimeout(() => {
        if (currentPredictionController) {
            currentPredictionController.abort();
        }
    }, 10000);

    try {
        canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        const result = await apiPredictFrame(canvasElement, {
            threshold: getConfidenceThreshold(),
            signal: currentPredictionController.signal
        });

        if (runId !== currentRunId || !isRunning) {
            return;
        }

        if (result && result.success) {
            updatePredictionUI(result);
            frameCount += 1;
            updateStats();
        } else if (result && result.error) {
            console.warn('Prediction returned an error:', result.error);
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Prediction error:', error);
        }
    } finally {
        window.clearTimeout(predictionTimeout);
        if (runId === currentRunId) {
            isPredicting = false;
            currentPredictionController = null;
        }
    }
}

function stopPredictionLoop() {
    if (predictionInterval) {
        window.clearInterval(predictionInterval);
        predictionInterval = null;
    }
}

function startStatsTimer() {
    stopStatsTimer();
    statsInterval = window.setInterval(updateStats, 1000);
}

function stopStatsTimer() {
    if (statsInterval) {
        window.clearInterval(statsInterval);
        statsInterval = null;
    }
}

function abortCurrentPrediction() {
    if (currentPredictionController) {
        currentPredictionController.abort();
        currentPredictionController = null;
    }
}

function stopStream() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

function updatePredictionUI(result) {
    const letterEl = document.getElementById('currentLetter');
    const confBar = document.getElementById('confidenceBar');
    const confText = document.getElementById('confidenceText');
    const detLetter = document.getElementById('detectedLetterDisplay');
    const detConf = document.getElementById('detectedConfidenceDisplay');
    const showLandmarks = document.getElementById('showLandmarks');

    const confidence = Math.round((result.confidence || 0) * 100);
    const hasLetter = result.letter && result.letter !== 'N/A';

    if (letterEl) letterEl.textContent = hasLetter ? result.letter : '-';
    if (confBar) confBar.style.width = confidence + '%';
    if (confText) confText.textContent = `Confianza: ${confidence}%`;

    if (hasLetter) {
        if (detLetter) detLetter.textContent = result.letter;
        if (detConf) detConf.textContent = confidence + '%';
        window.currentLetter = result.letter;
        window.currentConfidence = result.confidence || 0;

        // --- Lógica de confirmación automática por estabilidad ---
        if (result.letter === lastStableLetter) {
            stableLetterCount++;
        } else {
            stableLetterCount = 1;
            lastStableLetter = result.letter;
        }

        if (stableLetterCount === FRAMES_TO_CONFIRM) {
            stableLetterCount = 0; // reset para no disparar de nuevo
            handleStableLetter(result.letter, result.confidence || 0);
        }

        // Mostrar progreso visual de confirmación (opcional)
        updateConfirmationProgress(stableLetterCount, FRAMES_TO_CONFIRM);

    } else {
        if (detLetter) detLetter.textContent = '-';
        if (detConf) detConf.textContent = '0%';
        window.currentLetter = null;
        window.currentConfidence = 0;
        stableLetterCount = 0;
        lastStableLetter = null;
        updateConfirmationProgress(0, FRAMES_TO_CONFIRM);
    }

}

/**
 * Muestra barra de progreso mientras se sostiene la seña.
 * Si no existe el elemento en el HTML, no hace nada.
 */
function updateConfirmationProgress(current, total) {
    const progressBar = document.getElementById('confirmationProgress');
    if (progressBar) {
        const pct = Math.round((current / total) * 100);
        progressBar.style.width = pct + '%';
    }
}

function speakInBrowser(text) {
    if (!text || !('speechSynthesis' in window)) {
        return false;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-CO';
    utterance.rate = 0.95;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
    return true;
}

/**
 * Se ejecuta cuando una letra se detecta estable por FRAMES_TO_CONFIRM frames.
 * Decide qué hacer según si es letra normal o seña especial.
 */
async function handleStableLetter(letter, confidence = 0) {
    console.log(`Seña estable confirmada: ${letter}`);

    try {
        if (letter === 'BORRAR') {
            const result = await removeLetter();
            if (result.success) {
                document.getElementById('currentWord').value = result.current_word;
                showNotification('↩ Letra borrada', 'warning');
            }

        } else if (letter === 'ESPACIO') {
            const result = await addLetter(' ', confidence);
            if (result.success) {
                document.getElementById('currentWord').value = result.current_word;
                showNotification('Espacio agregado', 'info');
            }

        } else if (letter === 'ESCUCHAR') {
            const response = await fetch('/api/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const result = await response.json();
            if (result.success) {
                document.getElementById('currentWord').value = '';
                const spoke = speakInBrowser(result.spoken_text);
                showNotification(`🔊 Leyendo: "${result.spoken_text}"`, 'success');
                if (!spoke) {
                    showNotification('Tu navegador no permite reproducir voz automaticamente', 'warning');
                }
            } else {
                showNotification('No hay texto para leer', 'warning');
            }

        } else {
            // Letra normal → agregar automáticamente al texto
            const result = await addLetter(letter, confidence);
            if (result.success) {
                document.getElementById('currentWord').value = result.current_word;
                showNotification(`"${letter}" agregada`, 'success', 1000);
            }
        }
    } catch (error) {
        console.error('Error handling stable letter:', error);
        showNotification('Error al procesar seña', 'error');
    }
}

function drawLandmarks(base64Image) {
    const img = new Image();
    img.onload = () => {
        if (isRunning && canvasContext && canvasElement) {
            canvasContext.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        }
    };
    img.src = 'data:image/png;base64,' + base64Image;
}

function updateStats() {
    const statsTotal = document.getElementById('statsTotal');
    const statsDuration = document.getElementById('statsDuration');

    if (statsTotal) {
        statsTotal.textContent = String(frameCount);
    }

    if (statsDuration) {
        if (!startTime || !isRunning) {
            statsDuration.textContent = '00:00';
            return;
        }

        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        statsDuration.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
}

async function acceptLetter() {
    if (!window.currentLetter) {
        showNotification('No hay letra detectada', 'warning');
        return;
    }

    try {
        const result = await addLetter(window.currentLetter, window.currentConfidence || 0);
        if (result.success) {
            document.getElementById('currentWord').value = result.current_word;
            showNotification(`Letra "${window.currentLetter}" agregada`, 'success');
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
            document.getElementById('currentWord').value = result.current_word;
        }
    } catch (error) {
        console.error('Error removing letter:', error);
        showNotification('Error al deshacer', 'error');
    }
}

async function clearWord() {
    if (!confirm('Borrar palabra actual?')) return;

    try {
        const result = await resetWord();
        if (result.success) {
            document.getElementById('currentWord').value = result.current_word || '';
            stableLetterCount = 0;
            lastStableLetter = null;
            showNotification('Palabra borrada', 'success');
        }
    } catch (error) {
        console.error('Error clearing word:', error);
        showNotification('Error al borrar la palabra', 'error');
    }
}

async function saveWord() {
    const word = document.getElementById('currentWord').value;
    if (!word) {
        showNotification('No hay palabra para guardar', 'warning');
        return;
    }

    if (!window.savedWords) window.savedWords = [];
    window.savedWords.push({
        word: word,
        time: new Date().toLocaleTimeString('es-ES')
    });

    updateWordsList();

    try {
        await resetWord();
        document.getElementById('currentWord').value = '';
    } catch (error) {
        console.error('Error resetting word:', error);
    }
}

function updateWordsList() {
    const list = document.getElementById('wordsHistoryList');
    const noMsg = document.getElementById('noWordsMessage');

    if (!list || !noMsg) {
        return;
    }

    if (!window.savedWords || window.savedWords.length === 0) {
        list.innerHTML = '';
        noMsg.style.display = 'block';
        return;
    }

    noMsg.style.display = 'none';
    list.innerHTML = window.savedWords.map((item, index) => {
        return `<span class="badge bg-primary me-1 mb-1">${index + 1}. ${item.word} ${item.time}</span>`;
    }).join('');
}

function setCameraButtons(running) {
    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');

    if (startBtn) startBtn.classList.toggle('d-none', running);
    if (stopBtn) stopBtn.classList.toggle('d-none', !running);
}

function setStartButtonState(disabled) {
    const startBtn = document.getElementById('startCameraBtn');
    if (startBtn) {
        startBtn.disabled = disabled;
    }
}

function showCameraError(message) {
    const errorEl = document.getElementById('videoError');
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.remove('d-none');
    }
    showNotification(message, 'error');
}

function hideCameraError() {
    const errorEl = document.getElementById('videoError');
    if (errorEl) {
        errorEl.classList.add('d-none');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCamera);
} else {
    initCamera();
}
