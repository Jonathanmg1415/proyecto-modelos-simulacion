# Excepciones propias de la API. Se traducen a códigos HTTP en main.py
# mediante exception handlers registrados allí.
from __future__ import annotations


class ModeloNoDisponibleError(Exception):
      """La versión activa del modelo no existe o no se pudo cargar. -> 503."""

      def __init__(self, version: str, detalle: str = ""):
            self.version = version
            self.detalle = detalle
            super().__init__(f"Modelo no disponible para la versión '{version}': {detalle}")


class EntradaInvalidaError(Exception):
      """Los datos de entrada no son compatibles con el esquema esperado. -> 400."""

      def __init__(self, mensaje: str):
            self.mensaje = mensaje
            super().__init__(mensaje)


class VersionExistenteError(Exception):
      """/train intentó reutilizar un identificador de versión ya existente. -> 409."""

      def __init__(self, version: str):
            self.version = version
            super().__init__(f"La versión '{version}' ya existe.")


class EntrenamientoError(Exception):
      """El entrenamiento no pudo completarse (p. ej. CSV de datos ausente). -> 500."""

      def __init__(self, detalle: str):
            self.detalle = detalle
            super().__init__(f"El entrenamiento falló: {detalle}")