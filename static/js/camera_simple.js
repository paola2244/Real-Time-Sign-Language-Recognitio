/**
 * Camera Control Module - Versión Simplificada
 * Control manual de detección de letras
 */

let videoElement = null;
let canvasElement = null;
let canvasContext = null;
let stream = null;
let isRunning = false;
let frameCount = 0;
let startTime = null;
let predictionInterval = null;
let frameRate = 15;

function initCamera() {
    console.log('Inicializando módulo de cámara...');
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');
    if (canvasElement) {
        canvasContext = canvasElement.getContext('2d');
    }

    const startBtn = document.getElementById('startCameraBtn');
    const stopBtn = document.getElementById('stopCameraBtn');
    const acceptBtn = document.getElementById('acceptLetterBtn');
    const undoBtn = document.getElementById('undoLetterBtn');
    const clearBtn = document.getElementById('clearWordBtn');
    const saveBtn = document.getElementById('addWordBtn');

    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
    if (acceptBtn) acceptBtn.addEventListener('click', acceptLetter);
    if (undoBtn) undoBtn.addEventListener('click', undoLetter);
    if (clearBtn) clearBtn.addEventListener('click', clearWord);
    if (saveBtn) saveBtn.addEventListener('click', saveWord);

    console.log('Módulo de cámara inicializado');
}

async function startCamera() {
    try {
        console.log('Iniciando cámara...');
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });

        videoElement.srcObject = stream;

        videoElement.onloadedmetadata = () => {
            console.log('Video metadata cargado');
            videoElement.play();

            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;

            isRunning = true;
            startTime = Date.now();
            frameCount = 0;

            document.getElementById('startCameraBtn').classList.add('d-none');
            document.getElementById('stopCameraBtn').classList.remove('d-none');

            startPredictionLoop();
        };

    } catch (error) {
        console.error('Error en cámara:', error);
        alert('No se puede acceder a la cámara. Verifica los permisos.');
    }
}

function stopCamera() {
    console.log('Deteniendo cámara...');
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    isRunning = false;
    videoElement.srcObject = null;

    document.getElementById('startCameraBtn').classList.remove('d-none');
    document.getElementById('stopCameraBtn').classList.add('d-none');

    if (predictionInterval) {
        clearInterval(predictionInterval);
    }
}

function startPredictionLoop() {
    console.log('Iniciando loop de predicción...');
    const interval = Math.floor(1000 / frameRate);

    predictionInterval = setInterval(async () => {
        if (!isRunning || !videoElement.srcObject) {
            return;
        }

        try {
            canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            const result = await apiPredictFrame(canvasElement);

            if (result.success) {
                updatePredictionUI(result);
                frameCount++;
            }
        } catch (error) {
            console.error('Error en predicción:', error);
        }
    }, interval);
}

function updatePredictionUI(result) {
    // Panel principal
    const letterEl = document.getElementById('currentLetter');
    const confBar = document.getElementById('confidenceBar');
    const confText = document.getElementById('confidenceText');

    if (letterEl) {
        letterEl.textContent = (result.letter && result.letter !== 'N/A') ? result.letter : '-';
    }

    const conf = Math.round((result.confidence || 0) * 100);
    if (confBar) confBar.style.width = conf + '%';
    if (confText) confText.textContent = `Confianza: ${conf}%`;

    // Panel de palabra actual
    const detLetter = document.getElementById('detectedLetterDisplay');
    const detConf = document.getElementById('detectedConfidenceDisplay');

    if (result.letter && result.letter !== 'N/A') {
        if (detLetter) detLetter.textContent = result.letter;
        if (detConf) detConf.textContent = conf + '%';
        window.currentLetter = result.letter;
    } else {
        if (detLetter) detLetter.textContent = '-';
        if (detConf) detConf.textContent = '0%';
        window.currentLetter = null;
    }

    // Landmarks
    if (result.landmarks_image) {
        const img = new Image();
        img.onload = () => {
            canvasContext.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        };
        img.src = 'data:image/png;base64,' + result.landmarks_image;
    }
}

async function acceptLetter() {
    if (!window.currentLetter) {
        alert('No hay letra detectada');
        return;
    }

    try {
        const result = await addLetter(window.currentLetter);
        if (result.success) {
            document.getElementById('currentWord').value = result.current_word;
            console.log('Letra agregada:', window.currentLetter);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al agregar letra');
    }
}

async function undoLetter() {
    try {
        const result = await removeLetter();
        if (result.success) {
            document.getElementById('currentWord').value = result.current_word;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al deshacer');
    }
}

async function clearWord() {
    if (!confirm('¿Borrar palabra actual?')) return;

    try {
        await resetWord();
        document.getElementById('currentWord').value = '';
    } catch (error) {
        console.error('Error:', error);
    }
}

async function saveWord() {
    const word = document.getElementById('currentWord').value;
    if (!word) {
        alert('No hay palabra para guardar');
        return;
    }

    // Guardar en lista
    if (!window.savedWords) window.savedWords = [];
    window.savedWords.push({
        word: word,
        time: new Date().toLocaleTimeString('es-ES')
    });

    // Mostrar
    updateWordsList();

    // Limpiar
    try {
        await resetWord();
        document.getElementById('currentWord').value = '';
    } catch (error) {
        console.error('Error:', error);
    }
}

function updateWordsList() {
    const list = document.getElementById('wordsHistoryList');
    const noMsg = document.getElementById('noWordsMessage');

    if (!window.savedWords || window.savedWords.length === 0) {
        list.innerHTML = '';
        noMsg.style.display = 'block';
        return;
    }

    noMsg.style.display = 'none';
    let html = '';
    window.savedWords.forEach((item, i) => {
        html += `<span class="badge bg-primary me-1 mb-1">${i+1}. ${item.word} ${item.time}</span>`;
    });
    list.innerHTML = html;
}

// Inicializar
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCamera);
} else {
    initCamera();
}
