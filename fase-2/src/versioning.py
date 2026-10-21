
""" Manejo de versiones: crear la carpeta de una versión nueva, guardarla (modelo.joblib + metrics.json + metadata.json), 
cargarla desde predict.py. No sobrescribe silenciosamente una versión existente."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


class VersionManager:
    MODEL_FILENAME = "modelo.joblib"
    METRICS_FILENAME = "metrics.json"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def resolve_new_version_dir(self, version: str) -> Path:
        """Ruta de la versión nueva, o termina el programa (código != 0) si ya existe."""
        version_dir = self.output_dir / version
        if version_dir.exists():
            print(f"ERROR: la versión '{version}' ya existe en '{self.output_dir}'. "
                    f"No se sobrescribe: usa un identificador distinto.", file=sys.stderr)
            sys.exit(2)
        return version_dir

    def save(self, version_dir: Path, model, metrics: dict, metadata: dict) -> dict:
        version_dir.mkdir(parents=True)
        metadata = {"created_at": datetime.now(timezone.utc).isoformat(), **metadata}

        model_path = version_dir / self.MODEL_FILENAME
        joblib.dump(model, model_path, compress=3)

        metrics_path = version_dir / self.METRICS_FILENAME
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        metadata_path = version_dir / self.METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        return {
            "version_dir": str(version_dir.resolve()),
            "model_path": str(model_path.resolve()),
            "metrics_path": str(metrics_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        }

    def load(self, version: str):
        # Para predict.py (se deja lista para no duplicar esta lógica luego).
        
        version_dir = self.output_dir / version
        model_path = version_dir / self.MODEL_FILENAME
        if not model_path.exists():
            print(f"ERROR: no existe la versión '{version}' en '{self.output_dir}'.", file=sys.stderr)
            sys.exit(2)
        model = joblib.load(model_path)
        metrics = json.loads((version_dir / self.METRICS_FILENAME).read_text())
        metadata = json.loads((version_dir / self.METADATA_FILENAME).read_text())
        return model, metrics, metadata