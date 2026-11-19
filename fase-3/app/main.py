from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import (
      EntradaInvalidaError,
      EntrenamientoError,
      ModeloNoDisponibleError,
      VersionExistenteError,
)
from app.registro import registrar_solicitud
from app.routers import health, predict, train

app = FastAPI(
      title="API de predicción de precio de vehículos usados",
      description="Fase 3 del Proyecto Integrador. Reutiliza el modelo y los scripts de fase-2 (train.py / predict.py).",
      version="1.0.0",
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(train.router)


def _registrar_si_es_predict(request: Request, estado_http: int) -> None:
      # Solo se audita /predict (fase 4); /train y /health no forman parte del
      # registro que pide la especificación.
      if request.url.path != "/predict":
            return
      registrar_solicitud(get_settings().request_log_path, estado_http)


@app.exception_handler(RequestValidationError)
def validacion_fastapi(request: Request, exc: RequestValidationError) -> JSONResponse:
      # Pydantic/FastAPI rechazan aquí un campo faltante o de tipo incompatible
      # ANTES de que el router de /predict se ejecute (ver app/routers/predict.py).
      # Se reproduce el formato de error por defecto de FastAPI; solo se agrega
      # el registro para la fase 4.
      _registrar_si_es_predict(request, status.HTTP_422_UNPROCESSABLE_ENTITY)
      return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
      )


@app.exception_handler(ModeloNoDisponibleError)
def modelo_no_disponible(request: Request, exc: ModeloNoDisponibleError) -> JSONResponse:
      _registrar_si_es_predict(request, status.HTTP_503_SERVICE_UNAVAILABLE)
      return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(EntradaInvalidaError)
def entrada_invalida(request: Request, exc: EntradaInvalidaError) -> JSONResponse:
      _registrar_si_es_predict(request, status.HTTP_400_BAD_REQUEST)
      return JSONResponse(status_code=400, content={"detail": exc.mensaje})


@app.exception_handler(VersionExistenteError)
def version_existente(_: Request, exc: VersionExistenteError) -> JSONResponse:
      # /train, no /predict: no forma parte del registro de fase 4.
      return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(EntrenamientoError)
def entrenamiento_error(_: Request, exc: EntrenamientoError) -> JSONResponse:
      # /train, no /predict: no forma parte del registro de fase 4.
      return JSONResponse(status_code=500, content={"detail": str(exc)})