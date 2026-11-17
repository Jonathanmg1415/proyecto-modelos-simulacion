# Fase 3 — API REST (FastAPI)

Tercera entrega del Proyecto Integrador de *Modelos y Simulación de Sistemas I*
(Universidad de Antioquia, Grupo B). Expone como servicio HTTP el modelo de
predicción de precio de vehículos usados construido en las fases anteriores.

## Relación con las fases 1 y 2

La API **no reimplementa nada del modelo**: reutiliza directamente el código de
`fase-2/` (y el CSV de `fase-1/data/`):

| Necesidad de la API | Lo que se reutiliza de fase-2 |
|---|---|
| Esquema de entradas de `/predict` | `src/data.py` → `RAW_REQUIRED_COLUMNS` / `INPUT_SCHEMA` |
| Carga del modelo con su preprocesamiento | `src/versioning.py` → `load_model_for_predict()` (la misma función que usa `predict.py`); el pipeline guardado en `modelo.joblib` ya incluye `clean_features` + `ColumnTransformer` |
| Entrenamiento de versiones nuevas en `/train` | `train.py` → `run_training()` y `src/versioning.py` → `VersionManager` |
| Identificación del commit en metadatos | `src/versioning.py` → `get_git_commit()` (vía `GIT_COMMIT`) |

`app/config.py` agrega `../fase-2` al `sys.path` al iniciar, de modo que los
módulos de fase-2 se importan tal cual (`from train import run_training`,
`from src.versioning import load_model_for_predict`). Por eso la imagen Docker
se construye desde la raíz del repositorio (ver sección 6).

## 1. Estructura

```
fase-3/
├── app/
│   ├── main.py            # instancia FastAPI, routers y manejadores de excepciones
│   ├── config.py          # Settings (variables de entorno) y ruta a fase-2/
│   ├── dependencies.py    # Depends(): SettingsDep
│   ├── entities.py        # dataclasses internas: PredictionResult, TrainingResult
│   ├── exceptions.py      # excepciones propias -> códigos HTTP
│   ├── schemas.py         # Pydantic: PredictRequest/Response, TrainRequest/Response, HealthResponse
│   ├── service.py         # lógica: llama a fase-2 y traduce sus errores
│   └── routers/
│       ├── health.py      # GET  /health
│       ├── predict.py     # POST /predict
│       └── train.py       # POST /train
├── client/client.py       # cliente de prueba (biblioteca estándar)
├── tests/                 # pytest (TestClient de FastAPI)
├── Dockerfile
├── requirements.txt
└── README.md
```

## 2. Entorno

Python 3.13 (probado con 3.13.9 en Fedora y 3.13.2 en Docker). Se usan las
**mismas versiones** de pandas/numpy/scikit-learn/joblib fijadas en fase-2, para
que `models/v1` y `models/v2` carguen sin advertencias de versión.

```bash
# Desde la carpeta fase-3 (entorno propio, separado del de fase-2)
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Ejecución local

```bash
# Desde fase-3, con el .venv activado
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva generada por FastAPI: <http://127.0.0.1:8000/docs>.

Por defecto la API usa `fase-2/models/v1` para predecir y
`fase-1/data/used_cars.csv` para entrenar; ambos se cambian con variables de
entorno (sección 5).

## 4. Endpoints

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```
```json
{"status":"ok"}
```

### `POST /predict`

Recibe **un** vehículo en JSON con las 10 columnas de `RAW_REQUIRED_COLUMNS`
(mismo esquema que el CSV de `predict.py`; la columna `model` no es necesaria).
`accident` y `clean_title` admiten `null`.

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_year": 2013, "milage": "106,700 mi.",
       "engine": "290.0HP 3.6L V6 Cylinder Engine Flex Fuel Capability",
       "transmission": "6-Speed A/T", "brand": "Jeep", "fuel_type": "E85 Flex Fuel",
       "accident": "None reported", "clean_title": "Yes",
       "ext_col": "Black", "int_col": "Black"}'
```
```json
{"prediccion":15388.033756913637,"version":"v1"}
```

Es la primera fila de `fase-2/data/ejemplo_prediccion.csv`; el valor coincide
exactamente con el que produce `python predict.py --model models/v1 ...`.

Entrada inválida (falta `milage`):

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"model_year": 2013, "brand": "Jeep"}'
```
```json
{"detail":[{"type":"missing","loc":["body","milage"],"msg":"Field required", ...},
           {"type":"missing","loc":["body","engine"],"msg":"Field required", ...}, ...]}
```

### `POST /train`

Entrena una versión nueva con `run_training()` de fase-2 (mismos parámetros por
defecto: `target=price`, `seed=42`, `test_size=0.20`) y la guarda en
`MODELS_DIR/<version>/` con `modelo.joblib`, `metrics.json` y `metadata.json`.
El cuerpo es opcional: si no se envía `version`, se genera `v-AAAAMMDD-HHMMSS`.

```bash
curl -s -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" -d '{"version": "v-api"}'
```
```json
{"version":"v-api","model_mae":16195.692603390782,"baseline_mae":32865.97007481297}
```

(HTTP 201; con el CSV completo tarda unos segundos.) Repetir la misma versión
no la sobrescribe:

```json
{"detail":"La versión 'v-api' ya existe."}
```

### Códigos de respuesta

| Código | Cuándo | Origen |
|---|---|---|
| 200 | `/health`, `/predict` correctos | — |
| 201 | `/train` creó la versión | `routers/train.py` |
| 422 | Campo faltante o tipo incorrecto en `/predict` (se indica el campo en `loc`) | validación Pydantic |
| 400 | Datos con esquema válido pero incompatibles con el modelo | `EntradaInvalidaError` |
| 409 | `/train` con una versión que ya existe | `VersionExistenteError` |
| 503 | La versión activa no existe o no se pudo cargar | `ModeloNoDisponibleError` |
| 500 | El entrenamiento falló (p. ej. CSV ausente) | `EntrenamientoError` |

Los scripts de fase-2 terminan con `sys.exit(2)` ante errores; dentro de un
servidor eso tumbaría el proceso, así que `service.py` captura `SystemExit` y lo
convierte en las excepciones anteriores, que `main.py` traduce a HTTP.

## 5. Selección de versión y configuración

| Variable | Por defecto | Uso |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión de `MODELS_DIR` que usa `/predict` (se informa en la respuesta) |
| `MODELS_DIR` | `../fase-2/models` | Carpeta de versiones (lectura en `/predict`, escritura en `/train`) |
| `TRAIN_DATA_PATH` | `../fase-1/data/used_cars.csv` | CSV etiquetado para `/train` |
| `TRAIN_TARGET` / `TRAIN_SEED` / `TRAIN_TEST_SIZE` | `price` / `42` / `0.20` | Parámetros de `run_training()` |

Ejemplo local con otra versión:

```bash
MODEL_VERSION=v2 uvicorn app.main:app --port 8000
```

Versiones disponibles en el repositorio: `v1` (semilla 42) y `v2` (semilla 7),
entregadas en fase-2. Una versión entrenada con `/train` y los parámetros por
defecto reproduce exactamente `v1` (mismas métricas, misma predicción), lo que
confirma que el entrenamiento es determinista.

## 6. Docker

La imagen se construye **desde la raíz del repositorio** (no desde `fase-3/`),
porque copia también `fase-2/src`, `train.py` y `predict.py` conservando la
estructura relativa. El `.dockerignore` de la raíz excluye `.git`, los `.venv`,
datos y modelos, que no entran en la imagen.

```bash
# Desde la RAÍZ del repositorio (Linux / macOS)
docker build -f fase-3/Dockerfile -t modelos-fase3:1.0 \
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) .

docker run --rm -p 8000:8000 \
  -v "$PWD/fase-1/data:/app/fase-1/data:ro" \
  -v "$PWD/fase-2/models:/app/fase-2/models" \
  modelos-fase3:1.0
```

```powershell
# Windows (PowerShell), desde la raíz del repositorio. Misma salvedad que en
# fase-2: en Git Bash sobre Windows los montajes -v no funcionan.
docker build -f fase-3/Dockerfile -t modelos-fase3:1.0 `
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) .

docker run --rm -p 8000:8000 `
  -v "${PWD}\fase-1\data:/app/fase-1/data:ro" `
  -v "${PWD}\fase-2\models:/app/fase-2/models" `
  modelos-fase3:1.0
```

- La API queda en <http://127.0.0.1:8000> (dentro del contenedor escucha en
  `0.0.0.0:8000`). Para predecir con otra versión: `-e MODEL_VERSION=v2`.
- `fase-1/data` se monta solo lectura (CSV de entrenamiento) y `fase-2/models`
  lectura/escritura: las versiones creadas con `/train` **persisten en el equipo
  anfitrión** aunque el contenedor se elimine (`--rm`).
- El contenedor corre con un usuario sin privilegios (`apiuser`, UID 1000), de
  modo que los archivos escritos en el volumen quedan con el dueño del usuario
  habitual del equipo y no con `root`.

**Evidencia de ejecución** (Fedora, Docker 28; `GIT_COMMIT` = `a9a2c3e`):

```
$ docker build -f fase-3/Dockerfile -t modelos-fase3:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
[+] Building 40.2s (15/15) FINISHED
 => [internal] load .dockerignore            transferring context: 616B
 => [internal] load build context            transferring context: 34.32kB
 => [4/9] RUN pip install --no-cache-dir -r fase-3/requirements.txt      37.2s
 => [8/9] RUN useradd --create-home --uid 1000 apiuser && chown -R apiuser:apiuser /app
 => => naming to docker.io/library/modelos-fase3:1.0

$ docker run --rm -p 8000:8000 -v "$PWD/fase-1/data:/app/fase-1/data:ro" -v "$PWD/fase-2/models:/app/fase-2/models" modelos-fase3:1.0
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     172.17.0.1:36634 - "GET /health HTTP/1.1" 200 OK
INFO:     172.17.0.1:36638 - "POST /predict HTTP/1.1" 200 OK
INFO:     172.17.0.1:58730 - "POST /train HTTP/1.1" 201 Created

# En el host, tras el /train (el contenedor es quien entrenó: la ruta del CSV
# es la interna del contenedor, y el commit es el del build-arg):
$ ls -la fase-2/models/v-api
.rw-r--r--. juancarlos juancarlos 1.5 KB metadata.json
.rw-r--r--. juancarlos juancarlos 381 B  metrics.json
.rw-r--r--. juancarlos juancarlos 530 KB modelo.joblib
$ grep -E '"file"|git_commit' fase-2/models/v-api/metadata.json
    "file": "/app/fase-1/data/used_cars.csv",
  "git_commit": "a9a2c3ef8f44f8af4df9e184ed99443b5f5e6349",

# Reinicio con la versión nueva activa:
$ docker run --rm -p 8000:8000 -e MODEL_VERSION=v-api ... modelos-fase3:1.0
$ python client/client.py
== POST /predict -> HTTP 200
{"prediccion": 15388.033756913637, "version": "v-api"}
```

## 7. Pruebas (`pytest`)

Las pruebas usan el `TestClient` de FastAPI (sin levantar un servidor) y un
modelo pequeño entrenado con datos sintéticos en un directorio temporal, así no
dependen de `fase-2/models` ni del CSV real.

```bash
# Desde fase-3, con el .venv activado
python -m pytest -v
```

| Prueba | Verifica |
|---|---|
| `test_health_responde_200` | `GET /health` → 200 y `{"status": "ok"}` |
| `test_predict_entrada_valida_devuelve_prediccion_y_version` | entrada válida → 200 con `prediccion` numérica y `version` |
| `test_predict_campo_faltante_es_rechazado` | sin `milage` → 422 señalando el campo |
| `test_predict_version_inexistente_responde_503` | (extra) versión activa inexistente → 503 |

```
tests/test_api.py::test_health_responde_200 PASSED
tests/test_api.py::test_predict_entrada_valida_devuelve_prediccion_y_version PASSED
tests/test_api.py::test_predict_campo_faltante_es_rechazado PASSED
tests/test_api.py::test_predict_version_inexistente_responde_503 PASSED
4 passed in 0.29s
```

## 8. Cliente de prueba

`client/client.py` solo usa la biblioteca estándar. Con la API corriendo (local
o en Docker), desde `fase-3/`:

```bash
python client/client.py                      # /health + /predict con el ejemplo del Jeep
python client/client.py --sin-campo milage   # provoca el 422 por campo faltante
python client/client.py --input vehiculo.json --url http://127.0.0.1:8000
```

Termina con código 0 si la predicción respondió 200, 2 si la API rechazó la
solicitud y 1 si no pudo conectarse.

## 9. Identificación de la entrega

Commit/etiqueta correspondiente a la Entrega 3: `entrega-3` (ver etiquetas del
repositorio en GitHub). Desarrollo en la rama `feature/fase-3-api-rest`,
integrada a `develop` mediante Pull Request y luego a `main`.