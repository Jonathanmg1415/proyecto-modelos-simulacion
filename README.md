# proyecto-modelos-simulacion

Modelo predictivo de precios de vehículos usados — Proyecto Integrador de
**Modelos y Simulación de Sistemas I** (Grupo B, Universidad de Antioquia,
docente Andrés Parra). Regresión con Scikit-learn, contenerización con Docker y
despliegue vía API REST.

## Problema

Predecir el precio de venta de vehículos usados a partir de sus características
(marca, modelo, año, kilometraje, combustible, motor, transmisión, colores,
historial de accidentes y estado del título).

- **Tipo:** regresión supervisada · **Variable objetivo:** `price`
- **Dataset:** [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) (Kaggle) — 4 009 filas × 12 columnas

## Fases del proyecto

| Fase | Contenido | Entrega |
|---|---|---|
| **1** | Modelo predictivo (notebook + `modelo.joblib`) | 15/sep/2026 |
| 2 | `train.py`, `predict.py`, Dockerfile, pruebas `pytest` | 31/oct/2026 |
| 3 | API REST (`/health`, `/predict`, `/train`) + cliente de prueba | 22/nov/2026 |
| 4 | Monitoreo: registro de predicciones, métricas, versionado, reentrenamiento | — |

Cada fase reutiliza y amplía el trabajo de la anterior.

## Estado

- ✅ Fase 1 — ver [`fase-1/`](fase-1/) ([`fase-1/README.md`](fase-1/README.md))

## Estructura

```
.
├── fase-1/
│   ├── notebook.ipynb        # EDA + preparación + modelo + evaluación
│   ├── modelo.joblib         # pipeline entrenado
│   ├── README.md
│   └── data/used_cars.csv    # dataset original
├── requirements.txt
└── .gitignore
```

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # abrir fase-1/notebook.ipynb
```

## Flujo de trabajo (Git)

- `main`: versiones estables correspondientes a cada entrega.
- `develop`: rama de integración.
- `feature/*`: una rama por tarea, creada desde `develop`; entra a `develop` por
  Pull Request revisado. En cada entrega se integra `develop` en `main`.

## Equipo

- _(completar: Nombre 1)_
- _(completar: Nombre 2)_
- _(completar: Nombre 3)_
