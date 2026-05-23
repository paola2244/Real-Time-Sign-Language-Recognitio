/**
 * API Helper Functions
 * Funciones auxiliares para llamadas AJAX a los endpoints de Flask
 */

// Función auxiliar para hacer llamadas AJAX
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(endpoint, options);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        return result;

    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Función para subir archivos (FormData)
async function apiUploadFile(endpoint, file) {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();

    } catch (error) {
        console.error('Upload Error:', error);
        throw error;
    }
}

// Función para enviar imágenes de webcam
async function apiPredictFrame(canvas) {
    try {
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });

        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();

    } catch (error) {
        console.error('Predict Error:', error);
        throw error;
    }
}

// Función para obtener historial
async function getHistory() {
    return apiCall('/api/history/data', 'GET');
}

// Función para limpiar historial
async function clearHistory() {
    return apiCall('/api/history/clear', 'POST');
}

// Función para agregar letra
async function addLetter(letter) {
    return apiCall('/api/add-letter', 'POST', { letter: letter });
}

// Función para remover letra
async function removeLetter() {
    return apiCall('/api/remove-letter', 'POST');
}

// Función para resetear palabra
async function resetWord() {
    return apiCall('/api/reset-word', 'POST');
}

// Función para obtener estado de sesión
async function getSessionState() {
    return apiCall('/api/session-state', 'GET');
}

// Función para mostrar notificaciones
function showNotification(message, type = 'info', duration = 3000) {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';

    const alertHTML = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    const container = document.querySelector('main');
    const alertDiv = document.createElement('div');
    alertDiv.innerHTML = alertHTML;
    container.insertBefore(alertDiv.firstElementChild, container.firstChild);

    if (duration > 0) {
        setTimeout(() => {
            document.querySelector('.alert')?.remove();
        }, duration);
    }
}

// Inicializar observador de cambios en la palabra actual
document.addEventListener('DOMContentLoaded', function() {
    const currentWordInput = document.getElementById('currentWord');
    if (currentWordInput) {
        // Actualizar palabra desde API cada cierto tiempo
        setInterval(async () => {
            try {
                const state = await getSessionState();
                currentWordInput.value = state.current_word;
            } catch (error) {
                console.error('Error updating word:', error);
            }
        }, 500);
    }
});
