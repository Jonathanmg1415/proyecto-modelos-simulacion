# Pruebas del script de monitoreo (fase 4). Se ejecutan con:
#   python -m pytest   (desde fase-4/)
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoreo import calcular_metricas, formatear_reporte, leer_registro


def _escribir_registro(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fecha_hora", "estado_http", "version", "prediccion"])
        writer.writeheader()
        writer.writerows(filas)


def test_registro_inexistente_se_lee_como_vacio(tmp_path):
    filas = leer_registro(tmp_path / "no_existe.csv")
    assert filas == []


def test_metricas_con_registro_vacio(tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    _escribir_registro(ruta, [])
    total, fallidas = calcular_metricas(leer_registro(ruta))
    assert (total, fallidas) == (0, 0)
    assert "sin solicitudes" in formatear_reporte(total, fallidas)


def test_metricas_siguen_el_ejemplo_de_la_especificacion(tmp_path):
    # Dos solicitudes válidas y una rechazada: 3 totales, 1 fallida, 33,3 %.
    ruta = tmp_path / "solicitudes.csv"
    _escribir_registro(ruta, [
        {"fecha_hora": "2026-11-06T10:00:00", "estado_http": "200", "version": "v1", "prediccion": "18.7"},
        {"fecha_hora": "2026-11-06T10:01:00", "estado_http": "200", "version": "v1", "prediccion": "19.2"},
        {"fecha_hora": "2026-11-06T10:02:00", "estado_http": "422", "version": "", "prediccion": ""},
    ])
    total, fallidas = calcular_metricas(leer_registro(ruta))
    assert (total, fallidas) == (3, 1)
    assert formatear_reporte(total, fallidas) == (
        "Total de solicitudes : 3\nPorcentaje de fallos : 1/3 (33,3 %)"
    )


def test_estados_5xx_tambien_cuentan_como_fallidos(tmp_path):
    ruta = tmp_path / "solicitudes.csv"
    _escribir_registro(ruta, [
        {"fecha_hora": "2026-11-06T10:00:00", "estado_http": "200", "version": "v1", "prediccion": "18.7"},
        {"fecha_hora": "2026-11-06T10:01:00", "estado_http": "503", "version": "", "prediccion": ""},
    ])
    total, fallidas = calcular_metricas(leer_registro(ruta))
    assert (total, fallidas) == (2, 1)
