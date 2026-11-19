#!/usr/bin/env python3
"""Script de monitoreo de la fase 4.

Lee el registro de solicitudes a POST /predict (fase-3/app/registro.py lo
escribe en cada llamada, incluidas las rechazadas por validación) y muestra
dos indicadores de operación: el total de solicitudes y el porcentaje de
solicitudes fallidas (HTTP 4xx o 5xx).

Uso (desde fase-4/):
    python monitoreo.py
    python monitoreo.py --log ../fase-4/data/solicitudes.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent / "data" / "solicitudes.csv"


def leer_registro(ruta: Path) -> list[dict]:
    """Devuelve las filas del registro, o una lista vacía si el archivo no
    existe todavía (ninguna solicitud se ha atendido aún)."""
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def calcular_metricas(filas: list[dict]) -> tuple[int, int]:
    """(total de solicitudes, solicitudes con estado_http 4xx o 5xx)."""
    total = len(filas)
    fallidas = sum(1 for f in filas if 400 <= int(f["estado_http"]) <= 599)
    return total, fallidas


def formatear_reporte(total: int, fallidas: int) -> str:
    if total == 0:
        return "Total de solicitudes : 0\nPorcentaje de fallos  : sin solicitudes"
    porcentaje = fallidas / total * 100
    # Formato local (coma decimal), igual que el ejemplo de la especificación: "33,3 %".
    porcentaje_txt = f"{porcentaje:.1f}".replace(".", ",")
    return (
        f"Total de solicitudes : {total}\n"
        f"Porcentaje de fallos : {fallidas}/{total} ({porcentaje_txt} %)"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula el total de solicitudes y el porcentaje de "
                     "fallos (HTTP 4xx/5xx) a partir del registro de /predict."
    )
    parser.add_argument(
        "--log", default=str(DEFAULT_LOG),
        help=f"Ruta al registro CSV de /predict (por defecto: {DEFAULT_LOG}).",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    filas = leer_registro(Path(args.log))
    total, fallidas = calcular_metricas(filas)
    print(formatear_reporte(total, fallidas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
