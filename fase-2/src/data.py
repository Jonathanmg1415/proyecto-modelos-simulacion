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


def _read_csv_or_exit(path: Path, arg_name: str) -> pd.DataFrame:
      if not path.exists():
            print(f"ERROR: no se encontró el archivo '{path}'. "
                  f"Verifica la ruta pasada en --{arg_name}.", file=sys.stderr)
            sys.exit(2)
      try:
            return pd.read_csv(path)
      except Exception as exc:
            print(f"ERROR: no se pudo leer '{path}' como CSV ({exc}).", file=sys.stderr)
            sys.exit(2)


def load_and_validate(data_path: str, target: str) -> pd.DataFrame:
      """Para train.py: exige también la columna objetivo."""
      path = Path(data_path)
      df = _read_csv_or_exit(path, "data")

      missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
      if target not in df.columns:
            missing.append(target)
      if missing:
            print(f"ERROR: faltan columnas requeridas en el CSV: {missing}", file=sys.stderr)
            sys.exit(2)

      return df


def load_for_predict(input_path: str) -> pd.DataFrame:
      """Para predict.py: NO exige la variable objetivo (no es necesaria para predecir).

      Columnas adicionales en el CSV se ignoran (el modelo solo usa las columnas con
      las que se entrenó). Categorías no vistas en entrenamiento se tratan como
      "desconocidas" por el OneHotEncoder del pipeline (handle_unknown="ignore" /
      "infrequent_if_exist"), sin generar error. Una celda nula en una columna
      presente es válida (se imputa como en entrenamiento); lo que sí es un error
      es que falte la columna completa.
      """
      path = Path(input_path)
      df = _read_csv_or_exit(path, "input")

      missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
      if missing:
            print(f"ERROR: faltan columnas requeridas en el CSV de entrada: {missing}", file=sys.stderr)
            sys.exit(2)

      return df