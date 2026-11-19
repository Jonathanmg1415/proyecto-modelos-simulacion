#!/usr/bin/env python3
# Cliente de prueba simple de la API (fase 3). Solo usa la biblioteca estándar.
#
# Uso (con la API corriendo, local o en Docker):
#   python client/client.py                       # /health + /predict con el ejemplo
#   python client/client.py --input vehiculo.json # /predict con un JSON propio
#   python client/client.py --url http://127.0.0.1:8000 --sin-campo milage
#                                                 # prueba de rechazo (campo faltante)
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Primera fila de fase-2/data/ejemplo_prediccion.csv.
EJEMPLO = {
      "model_year": 2013,
      "milage": "106,700 mi.",
      "engine": "290.0HP 3.6L V6 Cylinder Engine Flex Fuel Capability",
      "transmission": "6-Speed A/T",
      "brand": "Jeep",
      "fuel_type": "E85 Flex Fuel",
      "accident": "None reported",
      "clean_title": "Yes",
      "ext_col": "Black",
      "int_col": "Black",
}


def llamar(url: str, metodo: str = "GET", cuerpo: dict | None = None) -> tuple[int, dict]:
      """Hace la petición y devuelve (código HTTP, cuerpo JSON), también en errores 4xx/5xx."""
      datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
      peticion = urllib.request.Request(
            url, data=datos, method=metodo, headers={"Content-Type": "application/json"}
      )
      try:
            with urllib.request.urlopen(peticion, timeout=30) as respuesta:
                  return respuesta.status, json.loads(respuesta.read().decode("utf-8"))
      except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))


def mostrar(titulo: str, codigo: int, cuerpo: dict) -> None:
      print(f"\n== {titulo} -> HTTP {codigo}")
      print(json.dumps(cuerpo, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
      parser = argparse.ArgumentParser(description="Cliente de prueba de la API de predicción.")
      parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL base de la API.")
      parser.add_argument("--input", help="Archivo JSON con los campos del vehículo (por defecto: ejemplo).")
      parser.add_argument("--sin-campo", metavar="CAMPO",
                              help="Envía la solicitud omitiendo este campo, para probar el rechazo.")
      args = parser.parse_args(argv)

      if args.input:
            with open(args.input, encoding="utf-8") as f:
                  vehiculo = json.load(f)
      else:
            vehiculo = dict(EJEMPLO)
      if args.sin_campo:
            vehiculo.pop(args.sin_campo, None)

      try:
            codigo, cuerpo = llamar(f"{args.url}/health")
      except (urllib.error.URLError, OSError) as exc:
            print(f"ERROR: no se pudo conectar con {args.url} ({exc}). ¿Está corriendo la API?",
                  file=sys.stderr)
            return 1
      mostrar("GET /health", codigo, cuerpo)

      codigo, cuerpo = llamar(f"{args.url}/predict", "POST", vehiculo)
      mostrar("POST /predict", codigo, cuerpo)
      return 0 if codigo == 200 else 2


if __name__ == "__main__":
      sys.exit(main())