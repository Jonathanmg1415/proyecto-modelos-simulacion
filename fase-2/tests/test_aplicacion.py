"""Pruebas mínimas exigidas por la sección 12 de la especificación de fase-2.

Ejecutar con: python -m pytest   (desde la carpeta fase-2/)
Usan datos sintéticos pequeños (conftest.py), sin red, y directorios
temporales para no tocar los modelos ya entregados en models/v1 y models/v2.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from predict import run_prediction
from train import run_training

from conftest import _make_synthetic_df


# 1. Entrenamiento y almacenamiento

def test_training_produces_loadable_artifact_with_metrics_and_metadata(trained_version):
      model = joblib.load(trained_version["model_path"])
      assert hasattr(model, "predict")

      metrics = json.loads(Path(trained_version["metrics_path"]).read_text(encoding="utf-8"))
      assert "baseline" in metrics and "model" in metrics
      assert "mae" in metrics["baseline"] and "mae" in metrics["model"]

      metadata = json.loads(Path(trained_version["metadata_path"]).read_text(encoding="utf-8"))
      assert metadata["version"] == "vtest"
      assert metadata["python_version"]
      assert metadata["git_commit"]


# 2. Carga y predicción

def test_loaded_artifact_predicts_consistently_with_pretraining_results(tmp_path, trained_version):
      model = joblib.load(trained_version["model_path"])

      df_new = _make_synthetic_df(n_rows=6, with_nulls=False).drop(columns=["price"])
      pred_direct = np.asarray(model.predict(df_new))

      input_csv = tmp_path / "input.csv"
      output_csv = tmp_path / "output.csv"
      df_new.to_csv(input_csv, index=False)

      resumen = run_prediction(
            model_path=trained_version["model_path"],
            input_path=str(input_csv),
            output_path=str(output_csv),
      )
      assert resumen["n_filas"] == len(df_new)

      salida = pd.read_csv(output_csv)
      np.testing.assert_allclose(salida["prediccion"].to_numpy(), pred_direct, rtol=1e-6)


# 3. Valores faltantes

def test_predict_handles_missing_values_in_nullable_columns(tmp_path, trained_version):
      df_new = _make_synthetic_df(n_rows=5, with_nulls=True).drop(columns=["price"])
      assert df_new[["accident", "clean_title"]].isna().any().any(), (
            "la muestra sintética debe incluir nulos para que la prueba sea significativa"
      )

      input_csv = tmp_path / "input_nulls.csv"
      output_csv = tmp_path / "output_nulls.csv"
      df_new.to_csv(input_csv, index=False)

      resumen = run_prediction(
            model_path=trained_version["model_path"],
            input_path=str(input_csv),
            output_path=str(output_csv),
      )
      assert resumen["n_filas"] == len(df_new)

      salida = pd.read_csv(output_csv)
      assert salida["prediccion"].notna().all()


# 4. Entrada inválida

def test_predict_missing_required_column_exits_nonzero(tmp_path, trained_version):
      df_new = _make_synthetic_df(n_rows=3, with_nulls=False).drop(columns=["price", "engine"])

      input_csv = tmp_path / "input_incompleto.csv"
      output_csv = tmp_path / "no_deberia_crearse.csv"
      df_new.to_csv(input_csv, index=False)

      with pytest.raises(SystemExit) as exc_info:
            run_prediction(
                  model_path=trained_version["model_path"],
                  input_path=str(input_csv),
                  output_path=str(output_csv),
            )
      assert exc_info.value.code != 0
      assert not output_csv.exists()


# 5. Versionamiento

def test_version_manager_does_not_overwrite_existing_version(tmp_path_factory, synthetic_csv):
      output_dir = tmp_path_factory.mktemp("models_versioning")

      run_training(data=synthetic_csv, target="price", output_dir=str(output_dir),
                  version="v1", seed=42, test_size=0.25)
      run_training(data=synthetic_csv, target="price", output_dir=str(output_dir),
                  version="v2", seed=7, test_size=0.25)

      model_v1 = joblib.load(output_dir / "v1" / "modelo.joblib")
      model_v2 = joblib.load(output_dir / "v2" / "modelo.joblib")
      assert hasattr(model_v1, "predict") and hasattr(model_v2, "predict")

      with pytest.raises(SystemExit) as exc_info:
            run_training(data=synthetic_csv, target="price", output_dir=str(output_dir),
                        version="v1", seed=99, test_size=0.25)
      assert exc_info.value.code != 0

      metadata_v1 = json.loads((output_dir / "v1" / "metadata.json").read_text(encoding="utf-8"))
      assert metadata_v1["model_params"]["random_state"] == 42