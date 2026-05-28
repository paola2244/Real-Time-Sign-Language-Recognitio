# Estructura del proyecto

Este proyecto esta organizado por responsabilidades para que sea mas facil mantenerlo.

## Entrada principal

- `app.py`: crea la aplicacion Flask, carga la configuracion, inicializa la sesion y registra las rutas.
- `config.py`: configuracion general, rutas del modelo, uploads, umbrales y base de datos.

## Capa web

- `web/services.py`: singletons y helpers compartidos por las rutas, como predictor, agente, repositorio, sesion e imagenes.
- `web/routes/`: endpoints separados por tema:
  - `main.py`: paginas principales.
  - `status.py`: estado de app, sesion y MongoDB.
  - `prediction.py`: prediccion por camara o imagen subida.
  - `words.py`: palabra actual, agregar, borrar, deshacer y guardar.
  - `history.py`: historial, CSV y guardar historial en base de datos.
  - `dataset.py`: captura de imagenes para entrenamiento y estadisticas.
  - `speech.py`: lectura de texto con voz.
  - `errors.py`: errores HTTP.

## IA e inferencia

- `infrastructure/ai/agent/`: agente de prediccion, confianza y estabilidad.
- `infrastructure/ai/realtime/`: procesamiento de camara y MediaPipe.
- `infrastructure/ai/models/`: definicion/carga de modelos.
- `infrastructure/ai/training/`: entrenamiento, evaluacion y preparacion de datos.
- `infrastructure/ai/utils/`: utilidades de imagen, logs y metricas.

## Persistencia

- `database/connection.py`: conexion MongoDB Atlas con fallback JSON.
- `database/models/`: modelos de datos.
- `database/repositories/`: acceso a colecciones/datos.
- `data/json_db/`: fallback local cuando MongoDB no esta disponible.

## Frontend

- `templates/`: vistas HTML de Flask.
- `static/js/`: logica de camara, API, uploads e historial.
- `static/css/`: estilos.

## Datos y modelos

- `data/collected_data/`: imagenes recopiladas para entrenamiento.
- `trained_models/`: modelos entrenados y labels.
- `uploads/`: archivos subidos temporalmente.
- `scripts/`: scripts de entrenamiento y verificacion.

