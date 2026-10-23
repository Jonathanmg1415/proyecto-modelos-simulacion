#!/usr/bin/env python3
# Script de predicción: carga una versión ya entrenada y predice sobre datos nuevos.
# No reentrena el modelo ni reajusta el preprocesamiento.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import load_for_predict
from src.versioning import load_model_for_predict


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predice el precio de vehículos usados con un modelo ya entrenado. "
                    "No entrena ni reajusta el preprocesamiento: solo carga y predice."
    )
    parser.add_argument(
        "--model", required=True,
        help="Ruta al modelo.joblib, o a la carpeta de una versión "
             "(p. ej. models/v1/modelo.joblib o models/v1).",
    )
    parser.add_argument(
        "--input", required=True,
        help="CSV con las columnas de entrada. No necesita la columna 'price'.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Ruta del CSV de salida con las predicciones.",
    )
    return parser


def run_prediction(model_path: str, input_path: str, output_path: str) -> dict:
    modelo = load_model_for_predict(model_path)
    df = load_for_predict(input_path)

    try:
        predicciones = modelo.predict(df)
    except Exception as exc:
        print(f"ERROR: el modelo no pudo generar predicciones ({exc}). "
              f"Revisa que los tipos de datos del CSV sean compatibles con el esquema "
              f"de entrenamiento.", file=sys.stderr)
        sys.exit(2)

    if len(predicciones) != len(df):
        # No debería ocurrir con este pipeline, pero se verifica explícitamente:
        # nunca se debe escribir un archivo que aparente tener resultados válidos
        # si el número de predicciones no corresponde al de filas de entrada.
        print("ERROR: el número de predicciones no coincide con el número de filas "
              "de entrada.", file=sys.stderr)
        sys.exit(1)

    salida = df.copy()
    salida["prediccion"] = predicciones

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(output, index=False)

    return {
        "n_filas": len(salida),
        "output_path": str(output.resolve()),
    }


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resumen = run_prediction(model_path=args.model, input_path=args.input, output_path=args.output)
    print("Predicción completada.")
    print(f"  Filas predichas : {resumen['n_filas']}")
    print(f"  Archivo de salida: {resumen['output_path']}")


if __name__ == "__main__":
    main()
