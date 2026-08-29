# Fase 1 — Modelo predictivo del precio de vehículos usados

Proyecto Integrador · Modelos y Simulación de Sistemas I · Grupo B · Universidad de Antioquia
Docente: Andrés Parra

## Integrantes

- _(completar: Nombre 1)_
- _(completar: Nombre 2)_
- _(completar: Nombre 3)_

## Problema

En el mercado de vehículos usados cada vendedor fija el precio de forma manual y
heterogénea. Se busca un modelo que **estime el precio de venta de un vehículo
usado a partir de sus características objetivas** (marca, año, kilometraje,
combustible, motor, transmisión, colores, historial de accidentes y estado del
título), útil como referencia para vendedores y compradores.

- **Tipo de problema:** aprendizaje supervisado, regresión.
- **Variable objetivo:** `price` (precio de venta en USD).

## Fuente del dataset

*Used Car Price Prediction Dataset* — Kaggle
<https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset>

4 009 filas × 12 columnas. Corte transversal (no es serie temporal). El archivo
se versiona en `fase-1/data/used_cars.csv`.

## Objetivo del modelo

Predecir `price` en dólares con el menor error posible, minimizando el **MAE**
(error absoluto medio) y superando con claridad un modelo base trivial.

## Algoritmo utilizado

- **Modelo base (baseline):** `DummyRegressor(strategy="median")`.
- **Modelo predictivo:** `RandomForestRegressor` dentro de un `Pipeline` de
  scikit-learn (limpieza → `ColumnTransformer` de imputación + one-hot →
  bosque), con el objetivo modelado en escala `log1p` mediante
  `TransformedTargetRegressor`.
- Contraste adicional: `HistGradientBoostingRegressor` (resultados equivalentes).

## Métrica empleada

MAE (principal, en dólares), acompañada de RMSE y R². Se reportan también MAE y
R² restringidos al mercado masivo (`price < 200 000`, el 98 % de las filas) para
aislar el efecto de unos pocos vehículos de colección de precio millonario.

## Principales resultados (conjunto de prueba, 20 %)

| Modelo | MAE | RMSE | R² | MAE (`<200k`) | R² (`<200k`) |
|---|---:|---:|---:|---:|---:|
| Baseline (mediana) | 32 866 | 144 219 | -0.02 | 20 269 | -0.04 |
| **Random Forest** | **17 008** | 135 670 | 0.10 | **7 305** | **0.84** |
| Hist Gradient Boosting | 16 243 | 130 904 | 0.16 | 6 888 | 0.82 |

- El Random Forest **reduce el MAE ~48 %** frente al baseline.
- El R² global es bajo por la asimetría extrema de `price`; sobre `log1p(price)`
  el R² es ~0.82 y en el mercado masivo ~0.84.
- Variables más influyentes: kilometraje, potencia del motor (HP), antigüedad y
  cilindrada; luego la marca (gama alta).

## Estructura de la entrega

```
fase-1/
├── notebook.ipynb        # análisis completo, ejecutable de principio a fin
├── modelo.joblib         # pipeline entrenado (limpieza + preprocesamiento + modelo)
├── README.md             # este archivo
└── data/
    └── used_cars.csv     # dataset original versionado
```

## Cómo ejecutar el notebook

Desde la **raíz del repositorio**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

jupyter lab            # abrir fase-1/notebook.ipynb y "Restart & Run All"
```

Ejecución no interactiva (reproduce el notebook y regenera `modelo.joblib`):

```bash
cd fase-1
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Requisitos: Python 3.11–3.13. El notebook fija `RANDOM_STATE = 42` y se ejecuta
sin intervención manual.

## Prevención de fuga de información

- `price` nunca está entre las predictoras (`X = df.drop(columns=["price"])`,
  verificado con `assert`).
- La separación train/test se hace **antes** de cualquier `fit`; la imputación y
  la codificación viven dentro del `Pipeline`, que solo recibe `X_train`.
- La limpieza previa al split (`parse_money`, `clean_features`) es determinista
  fila a fila: no ajusta parámetros con los datos. Se verifica en el notebook.
- No se usan variables "del futuro": todas describen el vehículo al momento de
  publicarlo.

## Nota sobre `modelo.joblib` (para la Fase 2)

El artefacto guarda el pipeline completo, pero `FunctionTransformer(clean_features)`
se serializa **por referencia**. Para cargar el modelo en un proceso nuevo hay
que tener definidas `clean_features` y `parse_money` (y las constantes
`REF_YEAR`, `KNOWN_FUEL`). En la Fase 2 estas funciones se moverán a un módulo
importable (`src/`) para que `predict.py` cargue el modelo sin depender del
notebook.
