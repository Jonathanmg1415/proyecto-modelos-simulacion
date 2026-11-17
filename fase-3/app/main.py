from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import (
      EntradaInvalidaError,
      EntrenamientoError,
      ModeloNoDisponibleError,
      VersionExistenteError,
)
from app.routers import health, predict, train

app = FastAPI(
      title="API de predicción de precio de vehículos usados",
      description="Fase 3 del Proyecto Integrador. Reutiliza el modelo y los scripts de fase-2 (train.py / predict.py).",
      version="1.0.0",
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(train.router)


@app.exception_handler(ModeloNoDisponibleError)
def modelo_no_disponible(_: Request, exc: ModeloNoDisponibleError) -> JSONResponse:
      return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(EntradaInvalidaError)
def entrada_invalida(_: Request, exc: EntradaInvalidaError) -> JSONResponse:
      return JSONResponse(status_code=400, content={"detail": exc.mensaje})


@app.exception_handler(VersionExistenteError)
def version_existente(_: Request, exc: VersionExistenteError) -> JSONResponse:
      return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(EntrenamientoError)
def entrenamiento_error(_: Request, exc: EntrenamientoError) -> JSONResponse:
      return JSONResponse(status_code=500, content={"detail": str(exc)})