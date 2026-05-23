# 🤖 Reconocimiento de Lenguaje de Señas

Sistema de reconocimiento de lenguaje de señas en tiempo real utilizando inteligencia artificial.

## 📋 Descripción del Sistema

Este proyecto es una aplicación web para reconocer letras del lenguaje de señas usando:
- **Backend**: Flask (Python)
- **AI/ML**: MediaPipe + TensorFlow/Keras
- **Detección**: Real-time hand landmark detection
- **Almacenamiento**: MongoDB (con fallback a JSON)

### Características Principales

✅ **Detección en Tiempo Real**: Reconoce letras de señas a través de la cámara  
✅ **Control Manual**: TÚ decides cuándo agregar cada letra detectada  
✅ **Recopilación de Datos**: Captura fotos de tus propias señas para mejorar el modelo  
✅ **Historial Persistente**: Guarda todas las palabras que captures  
✅ **Modelo Entrenado**: CNN con 98%+ de precisión en validación  

---

## 🚀 Cómo Ejecutar la Aplicación

### Paso 1: Abre Terminal

**Opción A - PowerShell (Recomendado para Windows):**
```powershell
Win + R
pwsh
Enter
```

**Opción B - CMD (Símbolo del Sistema):**
```
Win + R
cmd
Enter
```

**Opción C - Git Bash:**
```
Click derecho en la carpeta → Git Bash Here
```

### Paso 2: Ve al Directorio del Proyecto

```powershell
cd "C:\Users\Paola\Reconocimiento lenguaje de señas"
```

### Paso 3: Activa el Entorno Virtual

**PowerShell:**
```powershell
.\venv311\Scripts\Activate.ps1
```

**CMD:**
```cmd
venv311\Scripts\activate.bat
```

**Bash:**
```bash
source venv311/Scripts/activate
```

**Verás que el prompt cambia a:**
```
(venv311) C:\Users\Paola\Reconocimiento lenguaje de señas>
```

### Paso 4: Ejecuta Flask

```powershell
python app.py
```

Verás algo como:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Paso 5: Abre en el Navegador

```
http://localhost:5000
```

---

## 🎮 Cómo Usar la Aplicación

### Flujo Principal

#### 1️⃣ Inicia la Cámara
- Ve a la pestaña "Inicio"
- Haz clic en "Iniciar Cámara"
- Permite acceso a tu cámara

#### 2️⃣ Haz la Seña
- Posiciona tu mano frente a la cámara
- Haz la seña de una letra (A, B, C, etc.)
- Observa la letra detectada y el porcentaje de confianza

#### 3️⃣ Agrega la Letra (Control Manual)
**Si la letra es correcta:**
- Haz clic en el botón "✓ Agregar"
- La letra se agrega a "Palabra Actual"

**Si la letra es incorrecta:**
- NO hagas clic en "Agregar"
- Intenta de nuevo con otra posición

#### 4️⃣ Construye la Palabra
```
Letra A → "✓ Agregar" → Palabra: A
Letra B → "✓ Agregar" → Palabra: AB
Letra C → "✓ Agregar" → Palabra: ABC
...
```

#### 5️⃣ Guarda la Palabra
- Cuando termines la palabra, haz clic "💾 Guardar Palabra"
- Se guarda automáticamente en el historial abajo
- La palabra actual se limpia para la siguiente

### Botones de Control

| Botón | Función |
|-------|---------|
| ✓ Agregar | Agrega la letra detectada actual |
| ↶ Deshacer | Quita la última letra agregada |
| 🗑️ Borrar | Borra toda la palabra actual |
| 💾 Guardar Palabra | Guarda la palabra completa |

### Recopilación de Datos (Mejorar Modelo)

Ve a la pestaña "Dataset" para:
1. Selecciona una letra (A-Z)
2. Haz la seña frente a la cámara
3. Haz clic "📸 Capturar Foto"
4. Repite para varias posiciones y ángulos
5. Una vez tengas suficientes fotos, el modelo mejora automáticamente

---

## 🔧 Solución de Problemas

### ❌ Error: "Python no reconocido"
**Solución:**
- Asegúrate de estar EN el directorio del proyecto
- Usa ruta completa si es necesario

### ❌ Error: "No se puede activar el script" (PowerShell)
**Solución:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Luego intenta activar venv de nuevo.

### ❌ Error: "Puerto 5000 ya en uso"
**Solución:**
```powershell
# Mata el proceso anterior
taskkill /PID [numero] /F

# O usa otro puerto
python app.py --port 5001
# Luego abre: http://localhost:5001
```

### ❌ No veo los puntos de la mano en la cámara
**Solución:**
- Verifica buena iluminación
- Acerca más la mano a la cámara
- Abre una ventana para mejor luz

### ❌ El botón "Agregar" no funciona
**Solución:**
- Verifica que detecte una letra (no "-")
- Verifica que el % sea suficientemente alto
- Recarga la página (F5)

### ❌ Se borraron mis palabras guardadas
**Solución:**
- Las palabras se guardan en localStorage del navegador
- Si limpias caché del navegador se pierden
- Usa "Exportar" si necesitas hacer backup (próxima versión)

---

## 💡 Consejos de Uso

1. **Asegura buena iluminación** antes de empezar
2. **Haz señas lentamente** - la detección es en tiempo real
3. **Aguanta la posición** un momento para que se detecte
4. **Verifica el %** antes de hacer clic "Agregar"
5. **Si no estás seguro**, intenta de nuevo sin agregar

---

## 📊 Información Técnica

### Arquitectura

```
Frontend (HTML/CSS/JS)
       ↓
Flask Backend (Python)
       ↓
MediaPipe (Hand Detection)
       ↓
TensorFlow/Keras (Letter Classification)
       ↓
MongoDB/JSON (Data Persistence)
```

### Letras Soportadas
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z (26 letras)

### Precisión del Modelo
- **Validación**: 98.01%
- **Entrenado con**: ~500+ ejemplos por letra
- **Mejoras continuas**: Reentrenamiento con datos personalizados

---

## 🛑 Detener la Aplicación

**En la terminal:**
```
Presiona: Ctrl + C
```

---

## 📝 Notas Importantes

- ✅ Mantén la terminal abierta mientras usas la aplicación
- ✅ Flask necesita estar corriendo para servir la página
- ✅ Si cambias código, presiona Ctrl+C y vuelve a ejecutar `python app.py`
- ✅ Las palabras se guardan en el navegador (localStorage)
- ✅ Las fotos del dataset se guardan en la carpeta `data/collected_data/`

---

## 🎯 Estructura del Proyecto

```
Reconocimiento lenguaje de señas/
├── app.py                          # Aplicación Flask principal
├── config.py                       # Configuración
├── requirements.txt                # Dependencias Python
├── static/
│   ├── css/                       # Estilos
│   └── js/                        # JavaScript
├── templates/
│   └── index.html                 # Página principal
├── infrastructure/
│   └── ai/
│       └── realtime/             # Detección en tiempo real
├── database/                       # Conexión a base de datos
├── data/
│   └── collected_data/           # Fotos del dataset recopiladas
└── models/                        # Modelos entrenados
```

---

## 👤 Autor
Paola

## 📅 Última Actualización
Mayo 2026

---

**¡Listo para usar! 🚀**
