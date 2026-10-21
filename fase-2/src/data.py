# Lectura y validación del CSV: columnas crudas requeridas y su esquema

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW_REQUIRED_COLUMNS = [
      "model_year", "milage", "engine", "transmission",
      "brand", "fuel_type", "accident", "clean_title",
      "ext_col", "int_col",
]

INPUT_SCHEMA = {
      "model_year": "entero (año del modelo)",
      "milage": "texto numérico, p. ej. '51,000 mi.'",
      "engine": "texto libre, p. ej. '300.0HP 2.0L 4 Cylinder Engine'",
      "transmission": "texto libre, p. ej. '7-Speed A/T'",
      "brand": "texto (categoría)",
      "fuel_type": "texto (categoría; valores no reconocidos -> 'Unknown')",
      "accident": "texto o nulo (nulo -> 'Unknown')",
      "clean_title": "'Yes' o nulo",
      "ext_col": "texto (categoría de alta cardinalidad)",
      "int_col": "texto (categoría de alta cardinalidad)",
}


def load_and_validate(data_path: str, target: str) -> pd.DataFrame:
      path = Path(data_path)
      if not path.exists():
            print(f"ERROR: no se encontró el archivo de datos en '{path}'. "
                  f"Verifica la ruta pasada en --data.", file=sys.stderr)
            sys.exit(2)

      df = pd.read_csv(path)

      missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
      if target not in df.columns:
            missing.append(target)
      if missing:
            print(f"ERROR: faltan columnas requeridas en el CSV: {missing}", file=sys.stderr)
            sys.exit(2)

      return df