# Ejecutar el proyecto

Usa esta carpeta como proyecto principal:

```powershell
cd C:\Proyectos\lenguaje_senas_fix
```

Evita correrlo desde OneDrive o rutas con caracteres especiales si puedes.

## 1. Crear entorno limpio

Si ya tienes un `.venv` viejo o danado, eliminalo desde el explorador de archivos o con PowerShell:

```powershell
Remove-Item -Recurse -Force .venv
```

Crea el entorno:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Si `py -3.11` no existe, instala Python 3.11 y vuelve a ejecutar el comando.

## 2. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

No reinstales `mediapipe` solo con `--force-reinstall` sin `requirements.txt`, porque pip puede subir `numpy` y romper TensorFlow.

## 3. Verificar entorno

```powershell
python scripts\verify_runtime.py
```

Debe terminar con:

```text
[EXITO] El entorno esta listo para ejecutar la app.
```

## 4. Poner dataset privado

Descomprime el archivo de 1 GB y reemplaza:

```text
C:\Proyectos\lenguaje_senas_fix\data\collected_data
```

Debe quedar asi:

```text
data\collected_data\A
data\collected_data\B
data\collected_data\C
```

No debe quedar asi:

```text
data\collected_data\collected_data\A
```

Antes de hacer commits:

```powershell
git status --short
```

No agregues fotos al repo.

## 5. Entrenar

```powershell
python scripts\train_model_with_collected_data.py
```

El entrenamiento actualiza automaticamente:

```text
trained_models\best_model_hybrid.h5
trained_models\labels_hybrid.json
```

## 6. Ejecutar app

```powershell
python app.py
```

Abre:

```text
http://127.0.0.1:5000
```

Da permisos de camara en el navegador y usa el boton para iniciar camara.
