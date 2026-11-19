# Fixtures compartidas: un modelo pequeño entrenado con datos sintéticos (para
# no depender de fase-2/models ni del CSV real) y un TestClient de la API
# configurado, mediante variables de entorno, para usar esa versión.
from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import FASE2_DIR 
from train import run_training

N_ROWS = 40
BRANDS = ["Toyota", "Honda", "Ford", "BMW"]
FUEL_TYPES = ["Gasoline", "Hybrid", "Diesel"]
COLORS = ["Black", "White", "Gray", "Blue"]

# Primera fila de fase-2/data/ejemplo_prediccion.csv, como JSON de /predict.
EJEMPLO_VALIDO = {
      "model_year": 2013,
      "milage": "106,700 mi.",
      "engine": "290.0HP 3.6L V6 Cylinder Engine Flex Fuel Capability",
      "transmission": "6-Speed A/T",
      "brand": "Jeep",
      "fuel_type": "E85 Flex Fuel",
      "accident": "None reported",
      "clean_title": "Yes",
      "ext_col": "Black",
      "int_col": "Black",
}


def _make_synthetic_df(n_rows: int = N_ROWS) -> pd.DataFrame:
      rows = []
      for i in range(n_rows):
            rows.append({
                  "model_year": 2008 + (i % 16),
                  "milage": f"{10_000 + i * 1500:,} mi.",
                  "engine": f"{150 + (i % 5) * 50}.0HP {1.6 + (i % 4) * 0.5:.1f}L "
                        f"{4 + 2 * (i % 2)} Cylinder Engine",
                  "transmission": f"{6 + (i % 3)}-Speed A/T" if i % 3 else "CVT Automatic",
                  "brand": BRANDS[i % len(BRANDS)],
                  "fuel_type": FUEL_TYPES[i % len(FUEL_TYPES)],
                  "accident": "None reported" if i % 2 else "At least 1 accident",
                  "clean_title": "Yes",
                  "ext_col": COLORS[i % len(COLORS)],
                  "int_col": COLORS[(i + 2) % len(COLORS)],
                  "price": f"${8_000 + i * 733:,}",
            })
      return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory):
      """Entrena una versión 'vtest' en un directorio temporal y apunta la API a
      ella mediante variables de entorno. Se entrena una sola vez por módulo."""
      data_dir = tmp_path_factory.mktemp("data")
      models_dir = tmp_path_factory.mktemp("models")
      csv_path = data_dir / "synthetic.csv"
      _make_synthetic_df().to_csv(csv_path, index=False)
      run_training(data=str(csv_path), target="price", output_dir=str(models_dir),
                  version="vtest", seed=42, test_size=0.25)

      # Fase 4: el registro de /predict también se redirige a un archivo
      # temporal, para no escribir filas de prueba sobre fase-4/data/solicitudes.csv
      # (el registro real de la entrega).
      log_path = data_dir / "solicitudes_test.csv"

      previo = {k: os.environ.get(k) for k in
                  ("MODELS_DIR", "MODEL_VERSION", "TRAIN_DATA_PATH", "REQUEST_LOG_PATH")}
      os.environ["MODELS_DIR"] = str(models_dir)
      os.environ["MODEL_VERSION"] = "vtest"
      os.environ["TRAIN_DATA_PATH"] = str(csv_path)
      os.environ["REQUEST_LOG_PATH"] = str(log_path)
      yield {"models_dir": str(models_dir), "version": "vtest", "csv": str(csv_path),
             "log_path": str(log_path)}
      for k, v in previo.items():
            if v is None:
                  os.environ.pop(k, None)
            else:
                  os.environ[k] = v


@pytest.fixture(scope="module")
def client(trained_env) -> TestClient:
      from app.main import app
      return TestClient(app)