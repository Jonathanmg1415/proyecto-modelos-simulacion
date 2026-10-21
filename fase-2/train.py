#!/usr/bin/env python3
# Script de entrenamiento 

from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

from src.data import load_and_validate, INPUT_SCHEMA
from src.features import parse_money
from src.models import build_baseline, build_model
from src.evaluation import evaluate
from src.versioning import VersionManager, get_git_commit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de predicción de precio de vehículos usados "
                    "y guarda una versión reutilizable (modelo + métricas + metadatos)."
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV etiquetado.")
    parser.add_argument("--target", default="price", help="Columna objetivo (por defecto: price).")
    parser.add_argument("--output-dir", default="models", help="Directorio base de versiones.")
    parser.add_argument("--version", required=True, help="Identificador de la versión, p. ej. v1.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporción de prueba.")
    return parser


def run_training(data: str, target: str, output_dir: str, version: str,
                seed: int, test_size: float) -> dict:
    manager = VersionManager(output_dir)
    version_dir = manager.resolve_new_version_dir(version)

    df = load_and_validate(data, target)
    X = df.drop(columns=[target])
    y = parse_money(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    ref_year_train = int(X_train["model_year"].max())

    baseline = build_baseline()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(y_test, baseline.predict(X_test))

    modelo = build_model(ref_year=ref_year_train, seed=seed)
    modelo.fit(X_train, y_train)
    model_metrics = evaluate(y_test, modelo.predict(X_test))

    metrics = {"baseline": baseline_metrics, "model": model_metrics}
    metadata = {
        "version": version,
        "problem_type": "regresión",
        "target": target,
        "algorithm": "HistGradientBoostingRegressor (TransformedTargetRegressor, log1p/expm1)",
        "input_schema": INPUT_SCHEMA,
        "model_params": {"max_iter": 400, "learning_rate": 0.05, "random_state": seed},
        "data": {
            "source": "Kaggle: taeefnajib/used-car-price-prediction-dataset "
                    "(ver instrucciones de acceso en README de fase-2)",
            "file": str(data),
            "n_rows": int(len(df)),
        },
        "split": {
            "strategy": "train_test_split aleatorio",
            "test_size": test_size,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "evaluation_metric": "MAE, RMSE, R2 (y variantes <200k USD)",
        "requirements_file": "fase-2/requirements.txt",
        "git_commit": get_git_commit(),
        "reproducibility_notes": (
            "Modelo determinista con random_state fijo; no se usan componentes "
            "estocásticos adicionales (sin subsampling)."
        ),
    }

    paths = manager.save(version_dir, model=modelo, metrics=metrics, metadata=metadata)
    return {**paths, "baseline_mae": baseline_metrics["mae"], "model_mae": model_metrics["mae"]}


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_training(
        data=args.data, target=args.target, output_dir=args.output_dir,
        version=args.version, seed=args.seed, test_size=args.test_size,
    )
    print("Entrenamiento completado.")
    print(f"  Versión guardada en : {summary['version_dir']}")
    print(f"  Modelo              : {summary['model_path']}")
    print(f"  Métricas            : {summary['metrics_path']}")
    print(f"  Metadatos           : {summary['metadata_path']}")
    print(f"  MAE baseline        : {summary['baseline_mae']:.2f} USD")
    print(f"  MAE modelo          : {summary['model_mae']:.2f} USD")


if __name__ == "__main__":
    main()