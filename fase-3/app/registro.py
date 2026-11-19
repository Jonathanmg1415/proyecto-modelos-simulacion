# Registro de solicitudes a /predict para la fase 4 (monitoreo). Se agrega una
# fila por cada llamada, incluidas las rechazadas por validación. El archivo se
# abre en modo de agregar para conservar el historial entre ejecuciones; si no
# existe, se crea con encabezado. Un valor ausente (version/prediccion en un
# error) se deja vacío, no "None" ni "null".
from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

CAMPOS = ["fecha_hora", "estado_http", "version", "prediccion"]

# Varias solicitudes concurrentes podrían intentar crear/escribir el archivo
# al mismo tiempo; un lock por proceso evita filas intercaladas o un
# encabezado duplicado. No protege contra escrituras desde procesos distintos
# (p. ej. varios contenedores compartiendo el mismo volumen), fuera del
# alcance de esta fase.
_lock = threading.Lock()


def registrar_solicitud(
      ruta_log: str,
      estado_http: int,
      version: str | None = None,
      prediccion: float | None = None,
) -> None:
      path = Path(ruta_log)
      with _lock:
            es_nuevo = not path.exists()
            if es_nuevo:
                  path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", newline="", encoding="utf-8") as f:
                  writer = csv.DictWriter(f, fieldnames=CAMPOS)
                  if es_nuevo:
                        writer.writeheader()
                  writer.writerow({
                        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
                        "estado_http": estado_http,
                        "version": version if version is not None else "",
                        "prediccion": prediccion if prediccion is not None else "",
                  })
