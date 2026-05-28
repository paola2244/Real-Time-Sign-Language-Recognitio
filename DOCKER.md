# Docker

Esta configuracion levanta la app Flask en un contenedor Linux y usa tu `.env` local para conectar con MongoDB Atlas.

## Requisitos

- Docker Desktop instalado y abierto.
- Archivo `.env` en la raiz del proyecto.
- Modelo principal en `trained_models/best_model_hybrid.h5`.
- Labels en `trained_models/labels_hybrid.json`.

## Levantar la app

```powershell
cd C:\Proyectos\lenguaje_senas_fix
docker compose up --build
```

Luego abre:

```text
http://localhost:5000
```

## Ejecutar en segundo plano

```powershell
docker compose up --build -d
```

## Ver logs

```powershell
docker compose logs -f sign-language-app
```

## Detener

```powershell
docker compose down
```

## Validar MongoDB

Con el contenedor corriendo, abre:

```text
http://localhost:5000/api/database-status
```

Debe salir:

```json
{
  "backend": "mongodb",
  "configured_host": "cluster0.euqxcz1.mongodb.net",
  "message": "Connected to MongoDB: sign_language_db"
}
```

## Notas

- La camara sigue funcionando desde el navegador, porque el navegador captura los frames y se los manda al backend.
- El audio se reproduce en el navegador con Web Speech API. Por eso funciona aunque la app corra dentro de Docker.
- `data/collected_data`, `uploads`, `logs`, `flask_session` y `data/json_db` se montan como volumen local para que no se pierdan al recrear el contenedor.
- El archivo `.env` no se copia dentro de la imagen; Docker Compose lo carga como variables de entorno.
