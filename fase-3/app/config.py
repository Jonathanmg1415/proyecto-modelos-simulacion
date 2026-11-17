from __future__ import annotations

import sys
from pathlib import Path

# Raíz del repositorio y ruta a fase-2/, para reutilizar train.py, predict.py y src/ sin duplicar código.

REPO_ROOT = Path(__file__).resolve().parents[2]
FASE1_DIR = REPO_ROOT / "fase-1"
FASE2_DIR = REPO_ROOT / "fase-2"

if str(FASE2_DIR) not in sys.path:
      sys.path.insert(0, str(FASE2_DIR))

class Settings:
      """Configuración de la API, leída de variables de entorno
      con valores por defecto para ejecución local"""
      
      def __init__(self) -> None:
            import os
            
            self.models_dir: str = os.environ.get("MODELS_DIR", str(FASE2_DIR / "models"))
            self.active_version: str = os.environ.get("MODEL_VERSION", "v1")
            self.train_data_path: str = os.environ.get("TRAIN_DATA_PATH", str(FASE1_DIR / "data" / "used_cars.csv"))
            self.train_target: str = os.environ.get("TRAIN_TARGET", "price")
            self.train_seed: int = int(os.environ.get("TRAIN_SEED", "42"))
            self.train_test_size: float = float(os.environ.get("TRAIN_TEST_SIZE", "0.20"))


def get_settings() -> Settings:
      return Settings()