from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class PredictionResult:
      """Resultado de una predicción individual"""
      prediccion: float
      version: str

@dataclass
class TrainingResult:
      """Resultado de un entrenamiento: resumen de lo que ya devuelve run_training() de la fase-2/train.py"""
      version: str
      version_dir: str
      model_path: str
      metrics_path: str
      metadata_path: str
      baseline_mae: float
      model_mae: float
      
      @classmethod
      def from_run_training(cls, version: str, summary: dict[str, Any]) -> "TrainingResult":
            return cls(
                  version=version,
                  version_dir=summary["version_dir"],
                  model_path=summary["model_path"],
                  metrics_path=summary["metrics_path"],
                  metadata_path=summary["metadata_path"],
                  baseline_mae=summary["baseline_mae"],
                  model_mae=summary["model_mae"]
            )