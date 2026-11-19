# POST /predict: recibe un vehículo en JSON y responde la predicción de precio
# con la versión del modelo utilizada.
from __future__ import annotations

from fastapi import APIRouter, status

from app import service
from app.dependencies import SettingsDep
from app.registro import registrar_solicitud
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, settings: SettingsDep) -> PredictResponse:
      # FastAPI/Pydantic ya rechazó con 422 los campos faltantes o de tipo
      # incorrecto antes de llegar aquí (registrado en app/main.py). Los
      # errores del modelo (versión no disponible, datos incompatibles) los
      # traduce main.py a 503/400, y también se registran allí.
      resultado = service.predict_one(settings, body.model_dump())
      registrar_solicitud(
            settings.request_log_path,
            status.HTTP_200_OK,
            version=resultado.version,
            prediccion=resultado.prediccion,
      )
      return PredictResponse(prediccion=resultado.prediccion, version=resultado.version)