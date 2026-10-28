"""Fixtures compartidos para las pruebas de fase-2.

Genera datos sintéticos pequeños (sin red, sin depender del CSV real de
Kaggle) con el mismo esquema que exige src/data.py, y entrena una única
versión de modelo reutilizada por varias pruebas para no reentrenar
repetidamente (sección 12.1 de la especificación).
"""
from __future__ import annotations

import pandas as pd
import pytest

from train import run_training

N_ROWS = 40

BRANDS = ["Toyota", "Honda", "Ford", "BMW", "Jeep", "INFINITI", "Mazda", "Kia"]
FUEL_TYPES = ["Gasoline", "Hybrid", "Diesel", "E85 Flex Fuel", "Wind"]
COLORS = ["Black", "White", "Gray", "Blue", "Red"]


def _make_synthetic_df(n_rows: int = N_ROWS, with_nulls: bool = True) -> pd.DataFrame:
      rows = []
      for i in range(n_rows):
            price_value = 8_000 + i * 733
            rows.append({
                  "model_year": 2008 + (i % 16),
                  "milage": f"{10_000 + i * 1500:,} mi.",
                  "engine": f"{150 + (i % 5) * 50}.0HP {1.6 + (i % 4) * 0.5:.1f}L {4 + 2 * (i % 2)} Cylinder Engine",
                  "transmission": f"{6 + (i % 3)}-Speed A/T" if i % 3 else "CVT Automatic",
                  "brand": BRANDS[i % len(BRANDS)],
                  "fuel_type": FUEL_TYPES[i % len(FUEL_TYPES)],
                  "accident": (
                  None if (with_nulls and i % 7 == 0)
                  else ("None reported" if i % 2 else "At least 1 accident")
                  ),
                  "clean_title": None if (with_nulls and i % 9 == 0) else "Yes",
                  "ext_col": COLORS[i % len(COLORS)],
                  "int_col": COLORS[(i + 2) % len(COLORS)],
                  "price": f"${price_value:,}",
            })
      return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> str:
      """CSV etiquetado (con 'price'), sin nulos, para entrenar."""
      df = _make_synthetic_df(with_nulls=False)
      path = tmp_path_factory.mktemp("data") / "synthetic.csv"
      df.to_csv(path, index=False)
      return str(path)


@pytest.fixture(scope="module")
def trained_version(tmp_path_factory, synthetic_csv) -> dict:
      """Entrena una única versión, reutilizada por varias pruebas."""
      output_dir = tmp_path_factory.mktemp("models")
      return run_training(
            data=synthetic_csv, target="price", output_dir=str(output_dir),
            version="vtest", seed=42, test_size=0.25,
      )