# Fase 2 — Scripts y Docker

Proyecto Integrador · Modelos y Simulación de Sistemas I · Grupo B · Universidad de Antioquia
Docente: Andrés Parra

**Integrantes:** Jonathan Mazo González, Juan Carlos Gómez Jaramillo

## Relación con la Fase 1

Esta fase convierte el notebook de `fase-1/` (predicción del precio de vehículos
usados, Gradient Boosting sobre `log1p(price)`) en una aplicación reutilizable:
`train.py` para entrenar y versionar el modelo, `predict.py` para predecir con
una versión ya entrenada, ambos ejecutables por línea de comandos y dentro de
Docker. El problema, el dataset y el enfoque de modelado se conservan sin
cambios; `fase-1/` se mantiene intacta como entrega anterior.

## Correcciones aplicadas respecto a la Fase 1

A partir de la retroalimentación del profesor sobre la Entrega 1:

- **El modelo ya no depende del notebook para recargarse.** `clean_features` y
  `parse_money` (antes definidas dentro del notebook, en `__main__`) ahora
  viven en `src/features.py`, un módulo importable. `predict.py` se probó
  cargando una versión guardada **sin haber ejecutado el notebook ni
  `train.py` antes** en el mismo proceso.
- **Versión de scikit-learn sincronizada.** `requirements.txt` fija
  `scikit-learn==1.7.2`; las versiones `v1` y `v2` de `models/` se
  (re)generaron con ese mismo entorno, así que no hay advertencias de
  versión incompatible al cargarlas.
- **Metadatos de cada versión incluyen la versión de Python** usada para
  entrenarla (antes no se registraba).
- El reentrenamiento (`train.py`) usa directamente el algoritmo ya elegido en
  Fase 1 (Gradient Boosting); no vuelve a comparar contra otros candidatos
  usando el conjunto de prueba, por lo que no reintroduce la fuga de
  información señalada en la evaluación de la Entrega 1 (selección de modelo
  basada en métricas de test).

## 1. Entorno

- **Python:** 3.13 (probado con 3.13.2). Debe coincidir en el entorno local y
  en la imagen Docker (el `Dockerfile` usa `python:3.13.2-slim`).

```bash
# Desde la carpeta fase-2
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2. Datos

El dataset (*Used Car Price Prediction Dataset*, Kaggle, `taeefnajib/used-car-price-prediction-dataset`)
ya está versionado en el repositorio en `fase-1/data/used_cars.csv` desde la
Fase 1 — no requiere descarga ni credenciales adicionales.

`train.py` espera un CSV con, como mínimo, estas columnas (más la columna
objetivo indicada en `--target`):

| Columna | Tipo esperado |
|---|---|
| `model_year` | entero (año del modelo) |
| `milage` | texto numérico, p. ej. `"51,000 mi."` |
| `engine` | texto libre, p. ej. `"300.0HP 2.0L 4 Cylinder Engine"` |
| `transmission` | texto libre, p. ej. `"7-Speed A/T"` |
| `brand`, `ext_col`, `int_col` | texto (categóricas) |
| `fuel_type` | texto (valores no reconocidos → `"Unknown"`) |
| `accident` | texto o nulo (nulo → `"Unknown"`) |
| `clean_title` | `"Yes"` o nulo |

`predict.py` usa el mismo esquema, **sin** la columna objetivo. El repositorio
incluye un ejemplo pequeño en `data/ejemplo_prediccion.csv` (10 filas reales,
con nulos intencionales en `fuel_type`/`accident`/`clean_title` para mostrar
cómo se manejan).

## 3. Entrenamiento (`train.py`)

```bash
python train.py --help
```

```bash
python train.py --data ../fase-1/data/used_cars.csv --target price --output-dir models --version v1 --seed 42
```

Separa 80/20 train/test (`--test-size`, por defecto `0.20`), ajusta toda la
limpieza y el preprocesamiento **solo con el entrenamiento**, entrena
`HistGradientBoostingRegressor` (envuelto en `TransformedTargetRegressor`
sobre `log1p(price)`) y un `DummyRegressor` (mediana) como base, evalúa ambos
sobre el 20 % de prueba, y guarda:

```
models/<version>/
  modelo.joblib    # pipeline completo: limpieza + preprocesamiento + modelo
  metrics.json     # métricas del baseline y del modelo (MAE, RMSE, R², y variantes <200k USD)
  metadata.json    # algoritmo, semilla, esquema de entrada, tamaño de split,
                    # fuente de datos, versión de Python, commit de git, etc.
```

No sobrescribe una versión existente: si `--version` ya existe en
`--output-dir`, termina con error (código de salida ≠ 0) sin tocar la versión
anterior.

## 4. Versiones entregadas

Se incluyen dos versiones generadas con `train.py` (mismo algoritmo, distinta
semilla — no se exige que una mejore a la otra):

| Versión | Semilla | MAE modelo (USD) | MAE baseline (USD) |
|---|---:|---:|---:|
| `models/v1` | 42 | 16.195,69 | 32.865,97 |
| `models/v2` | 7 | 10.384,91 | 25.670,43 |

(El MAE del baseline cambia entre versiones porque la semilla también cambia
la partición train/test, no porque el baseline se reajuste de otra forma.)

## 5. Predicción (`predict.py`)

```bash
python predict.py --help
```

```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`--model` acepta tanto la ruta al `modelo.joblib` como la carpeta de la
versión (p. ej. `models/v1`). No reentrena ni reajusta nada: solo carga el
pipeline guardado. El CSV de salida conserva todas las columnas de entrada,
en el mismo orden, más una columna `prediccion` al final.

Columnas adicionales en el CSV de entrada se ignoran; categorías no vistas en
entrenamiento se tratan como "desconocidas" por el `OneHotEncoder` del
pipeline, sin generar error. Si falta una columna requerida, o el modelo
indicado no existe, el script termina con un mensaje de error y código de
salida distinto de cero, sin generar el archivo de salida.

## 6. Pruebas (`pytest`)

```bash
cd fase-2
python -m pytest
```

Pruebas mínimas en `tests/` (cubren los 5 casos exigidos por la especificación
de Fase 2): entrenamiento y almacenamiento genera un artefacto cargable con
métricas y metadatos; el artefacto guardado predice una fila por fila de
entrada y reproduce las predicciones del modelo antes de guardarlo; una
entrada válida con nulos se procesa según la estrategia documentada; la
ausencia de una columna requerida produce un error con código de salida ≠ 0;
crear una segunda versión conserva la primera y un identificador repetido no
sobrescribe. Usan datos sintéticos pequeños y `tmp_path`, sin red ni
modificar `models/` de la entrega.

## 7. Docker

```bash
docker build -t modelos-fase2:1.0 .
```

**Bash / Linux / macOS:**

```bash
mkdir -p models outputs
docker run --rm \
  -v "$PWD/../fase-1/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  modelos-fase2:1.0 python train.py \
  --data data/used_cars.csv --target price \
  --output-dir models --version v-docker --seed 42

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones.csv
```

**PowerShell (Windows):**

```powershell
New-Item -ItemType Directory -Force models, outputs | Out-Null

docker run --rm `
  -v "${PWD}\..\fase-1\data:/app/data:ro" `
  -v "${PWD}\models:/app/models" `
  modelos-fase2:1.0 python train.py `
  --data data/used_cars.csv --target price `
  --output-dir models --version v-docker --seed 42

docker run --rm `
  -v "${PWD}\data:/app/data:ro" `
  -v "${PWD}\models:/app/models:ro" `
  -v "${PWD}\outputs:/app/outputs" `
  modelos-fase2:1.0 python predict.py `
  --model models/v-docker/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones.csv
```

Los datos se montan desde el equipo anfitrión (no se copian a la imagen); los
modelos, métricas, metadatos y predicciones se escriben en carpetas montadas,
por lo que persisten aunque el contenedor se elimine (`--rm`). El
entrenamiento **no** ocurre durante `docker build`, solo al ejecutar
`docker run ... python train.py ...`.

> _Evidencia de ejecución: completar con la salida real de los comandos
> anteriores antes de la entrega (captura o texto copiado)._

## 8. Resultados y limitaciones conocidas

- **Métrica principal:** MAE (USD). El modelo (`v1`) reduce el MAE en ~50,7 %
  frente al baseline (16.195,69 USD vs. 32.865,97 USD).
- Igual que en la Fase 1, el R² global es bajo (~0,14) por el efecto de unos
  pocos vehículos de colección en el conjunto de prueba; sobre el segmento
  `price < 200.000 USD` (~98 % de las filas) el desempeño es sólido
  (R² ≈ 0,84). Ese segmento se define con el precio real, por lo que **no**
  puede usarse para decidir de antemano si un anuncio nuevo pertenece al
  "mercado masivo" — es una lectura diagnóstica del conjunto de prueba, no
  una regla aplicable en `predict.py`.
- `HistGradientBoostingRegressor` no es completamente determinista a nivel de
  bytes entre corridas (pequeñas diferencias de compilación/threading), pero
  sí lo es en las métricas con la misma semilla y los mismos datos.

## 9. Identificación de la entrega

Commit/etiqueta correspondiente a la Entrega 2: `_(completar al etiquetar
`entrega-2`, antes de la fecha límite)_`.

## 10. Fuentes

- Dataset: [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) (Kaggle, CC BY 4.0).
- Especificación técnica de la Fase 2 proporcionada por el docente del curso.
- Código y pipeline de preprocesamiento adaptados del notebook propio de
  `fase-1/notebook.ipynb` (secciones 5 y 8).
