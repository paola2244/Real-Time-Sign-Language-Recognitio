# 🧹 Limpieza Opcional de Archivos Innecesarios

Este archivo documenta cómo **limpiar tu copia local** del proyecto de archivos que ya no se necesitan. 

⚠️ **IMPORTANTE:** Esta limpieza es **OPCIONAL y LOCAL**. Solo afecta tu computadora, no el repositorio compartido.

---

## 📋 Archivos que Puedes Eliminar

### 1️⃣ **Archivos de Test/Debug** (~100KB)
```bash
rm -f test_*.py
```

Archivos a eliminar:
- `test_history_flow.py`
- `test_hybrid_model.py`
- `test_landmarks_augmentation.py`
- `test_landmarks_loader.py`
- `test_model_direct.py`
- `test_prediction_flow.py`
- `test_realtime_pipeline.py`
- `test_webcam_landmarks.py`

**Por qué:** Son scripts de debugging que usamos durante desarrollo, no forman parte del sistema final.

---

### 2️⃣ **Datasets Antiguos** (~200MB)
```bash
rm -rf datasets/kaggle
rm -rf datasets/landmarks
```

**Por qué:** El proyecto ahora usa `data/collected_data/` para tus fotos personales. Estos son datos de entrenamiento antiguos que no se usan.

---

### 3️⃣ **Imágenes de Referencia Antiguas** (~5MB)
```bash
rm -rf data/raw
```

**Por qué:** Son imágenes de ejemplo del dataset MNIST original, ya no necesarias.

---

### 4️⃣ **Scripts de Entrenamiento Antiguos** (~50KB)
```bash
rm -f scripts/download_landmarks_dataset.py
rm -f scripts/train_model.py
```

**Por qué:** Fueron reemplazados por `scripts/train_model_with_collected_data.py` que es mejor.

---

### 5️⃣ **Modelos Antiguos** (~100MB)
```bash
rm -f trained_models/best_model.keras
rm -f trained_models/best_model_hybrid_improved_*.h5
rm -f trained_models/labels_hybrid_improved_*.json
rm -f trained_models/training_history.pkl
```

⚠️ **MANTÉN:** `trained_models/best_model_hybrid.h5` y `trained_models/labels_hybrid.json` (respaldos)

**Por qué:** El modelo activo está en `C:/models_prod/`. Estos son versiones antiguas.

---

### 6️⃣ **Logs Antiguos** (~10MB)
```bash
rm -f train_output.log
rm -f train_output_v2.log
rm -rf logs/
```

**Por qué:** Son logs históricos de ejecuciones anteriores, se regeneran automáticamente.

---

### 7️⃣ **Sesiones Temporales de Flask** (~500KB)
```bash
rm -rf flask_session/
```

**Por qué:** Se crean automáticamente cada vez que ejecutas la app, no necesita ser guardado.

---

## 🚀 Script de Limpieza Completo (Windows PowerShell)

```powershell
# Navega al proyecto
cd "C:\Users\Paola\Reconocimiento lenguaje de señas"

# Elimina todo de una vez
rm -Force -Recurse test_*.py, datasets/kaggle, datasets/landmarks, data/raw, `
    trained_models/best_model.keras, trained_models/*_improved_*.h5, `
    trained_models/*_improved_*.json, trained_models/training_history.pkl, `
    train_output*.log, logs, flask_session -ErrorAction SilentlyContinue

Write-Host "Limpieza completada!"
```

---

## 🐧 Script de Limpieza Completo (Linux/Mac/Bash)

```bash
#!/bin/bash
cd "C:\Users\Paola\Reconocimiento lenguaje de señas"

# Elimina todo
rm -f test_*.py
rm -rf datasets/kaggle datasets/landmarks data/raw
rm -f scripts/download_landmarks_dataset.py scripts/train_model.py
rm -f trained_models/best_model.keras trained_models/*_improved_*.h5 trained_models/*_improved_*.json
rm -f trained_models/training_history.pkl train_output*.log
rm -rf logs flask_session

echo "Limpieza completada!"
```

---

## ✅ Verificar Limpieza

Después de limpiar, verifica que el proyecto aún funciona:

```bash
python app.py
# Abre http://localhost:5000 en tu navegador
```

Si todo funciona normalmente, ¡la limpieza fue exitosa!

---

## 🔄 Después de Limpiar

Si después necesitas volver a descargar estos archivos:

```bash
git checkout -- .
# Esto restaura TODO a su estado anterior
```

O clona el repo nuevamente:
```bash
git clone <repo-url>
```

---

## 📊 Espacio a Liberar

| Elemento | Tamaño |
|----------|--------|
| test_*.py | ~100KB |
| datasets/kaggle/ | ~100MB |
| datasets/landmarks/ | ~50MB |
| data/raw/ | ~5MB |
| trained_models/ (antiguos) | ~100MB |
| logs/ | ~5MB |
| flask_session/ | ~500KB |
| **TOTAL** | **~260MB** |

---

## ❓ Preguntas Frecuentes

**P: ¿Perderé datos importantes?**
A: No. Todos estos archivos son respaldos o datos temporales. Los datos importantes (collected_data, BD de predicciones) se mantienen intactos.

**P: ¿Afectará a mis compañeros?**
A: No. Esta limpieza es **LOCAL SOLAMENTE**. No afecta el repositorio compartido.

**P: ¿Puedo restaurar los archivos?**
A: Sí, con `git checkout -- .` o clonando nuevamente.

**P: ¿Qué archivo NO debo eliminar?**
A: **NO ELIMINES:**
- `data/collected_data/` - Tus fotos personales
- `trained_models/best_model_hybrid.h5` - Modelo de respaldo
- `scripts/train_model_with_collected_data.py` - Script actual de entrenamiento

---

**¡Listo! Puedes ejecutar esta limpieza cuando quieras.** 🧹
