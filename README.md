# Proyecto Integrador — Modelos y Simulación de Sistemas I

**Universidad de Antioquia · Grupo B · Docente: Andrés Parra**

## Integrantes

- Jonathan Mazo González
- Juan Carlos Gómez Jaramillo

## Sobre el proyecto

Predicción del precio de venta de vehículos usados (`used_cars.csv`, Kaggle, autor `taeefnajib`), desarrollado en cuatro fases:

- **[`fase-1/`](fase-1/README.md)** — modelo predictivo (notebook): EDA,
  preparación de datos, comparación de modelos y persistencia.
- **[`fase-2/`](fase-2/README.md)** — scripts de entrenamiento y predicción
  reutilizables (`train.py` / `predict.py`), versionamiento del modelo,
  contenerización con Docker y pruebas automatizadas con pytest.
- **[`fase-3/`](fase-3/README.md)** — API REST (FastAPI) que expone el modelo:
  `/health`, `/predict`, `/train`, cliente de prueba y Dockerfile actualizado.
- **[`fase-4/`](fase-4/README.md)** — monitoreo: registro de las solicitudes a
  `/predict`, script de indicadores de operación y política de reentrenamiento.

Cada carpeta tiene su propio README con instrucciones completas de
instalación y ejecución para esa entrega.
