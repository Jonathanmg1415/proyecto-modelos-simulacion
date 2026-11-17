# POST /predict: recibe un vehículo en JSON y responde la predicción de precio
# con la versión del modelo utilizada.
from __future__ import annotations

from fastapi import APIRouter

from app import service
from app.dependencies import SettingsDep
from app.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, settings: SettingsDep) -> PredictResponse:
      # FastAPI/Pydantic ya rechazó con 422 los campos faltantes o de tipo
      # incorrecto antes de llegar aquí. Los errores del modelo (versión no
      # disponible, datos incompatibles) los traduce main.py a 503/400.
      resultado = service.predict_one(settings, body.model_dump())
      return PredictResponse(prediccion=resultado.prediccion, version=resultado.version)