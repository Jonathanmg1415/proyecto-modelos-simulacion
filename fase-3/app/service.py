# Capa de servicio: reutiliza el código de fase-2 (carga del modelo con su
# preprocesamiento guardado, esquema de entradas y run_training) y traduce los
# errores de esos scripts (que terminan con sys.exit) a excepciones propias.
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

# config debe importarse ANTES que train/src: es quien agrega fase-2/ al sys.path.
from app.config import Settings
from app.entities import PredictionResult, TrainingResult
from app.exceptions import (
      EntradaInvalidaError,
      EntrenamientoError,
      ModeloNoDisponibleError,
      VersionExistenteError,
)

from src.data import RAW_REQUIRED_COLUMNS          # fase-2: esquema de entradas
from src.versioning import VersionManager, load_model_for_predict  # fase-2
from train import run_training                     # fase-2: entrenamiento

# Caché de modelos ya cargados, por ruta del modelo.joblib. Evita deserializar
# el modelo en cada petición; una versión nueva (ruta nueva) se carga al usarse.
_MODEL_CACHE: dict[str, object] = {}


def _model_path(settings: Settings, version: str) -> Path:
      return Path(settings.models_dir) / version / VersionManager.MODEL_FILENAME


def load_model(settings: Settings, version: str | None = None):
      """Carga (o recupera de caché) el modelo de la versión indicada o de la activa."""
      version = version or settings.active_version
      path = _model_path(settings, version)
      key = str(path)
      if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
      if not path.exists():
            raise ModeloNoDisponibleError(version, f"no existe '{path}'")
      try:
            # load_model_for_predict (fase-2) termina con sys.exit(2) si el archivo
            # no se puede deserializar; se captura para responder 503 en vez de
            # tumbar el servidor.
            modelo = load_model_for_predict(str(path))
      except SystemExit as exc:
            raise ModeloNoDisponibleError(version, "no se pudo cargar el archivo") from exc
      _MODEL_CACHE[key] = modelo
      return modelo


def predict_one(settings: Settings, datos: dict) -> PredictionResult:
      """Predice el precio de un vehículo a partir de un diccionario con las
      columnas de RAW_REQUIRED_COLUMNS. Mismo flujo que predict.py: cargar el
      modelo (pipeline con preprocesamiento guardado) y llamar a .predict()."""
      version = settings.active_version
      modelo = load_model(settings, version)

      faltantes = [c for c in RAW_REQUIRED_COLUMNS if c not in datos]
      if faltantes:
            raise EntradaInvalidaError(f"faltan campos requeridos: {faltantes}")

      df = pd.DataFrame([{c: datos[c] for c in RAW_REQUIRED_COLUMNS}])
      try:
            pred = modelo.predict(df)
      except Exception as exc:
            raise EntradaInvalidaError(
                  f"el modelo no pudo generar una predicción con los datos recibidos ({exc})"
            ) from exc
      return PredictionResult(prediccion=float(pred[0]), version=version)


def generate_version_id(models_dir: str) -> str:
      """Identificador automático v-AAAAMMDD-HHMMSS; si ya existe (dos
      entrenamientos en el mismo segundo) se agrega un sufijo -2, -3, ..."""
      base = datetime.now().strftime("v-%Y%m%d-%H%M%S")
      candidato, n = base, 1
      while (Path(models_dir) / candidato).exists():
            n += 1
            candidato = f"{base}-{n}"
      return candidato


def train_new_version(settings: Settings, version: str | None = None) -> TrainingResult:
      """Entrena y guarda una versión nueva reutilizando run_training() de fase-2."""
      version = version or generate_version_id(settings.models_dir)
      if (Path(settings.models_dir) / version).exists():
            raise VersionExistenteError(version)
      try:
            resumen = run_training(
            data=settings.train_data_path,
            target=settings.train_target,
            output_dir=settings.models_dir,
            version=version,
            seed=settings.train_seed,
            test_size=settings.train_test_size,
            )
      except SystemExit as exc:
            # run_training termina con sys.exit(2) si faltan el CSV o columnas.
            raise EntrenamientoError(
                  f"revisa TRAIN_DATA_PATH='{settings.train_data_path}' y sus columnas"
            ) from exc
      except Exception as exc:
            raise EntrenamientoError(str(exc)) from exc
      return TrainingResult.from_run_training(version, resumen)