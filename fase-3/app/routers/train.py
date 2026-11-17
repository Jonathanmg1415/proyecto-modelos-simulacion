# POST /train: entrena una versión nueva del modelo y responde su identificador.
from __future__ import annotations

from fastapi import APIRouter, status

from app import service
from app.dependencies import SettingsDep
from app.schemas import TrainRequest, TrainResponse

router = APIRouter(tags=["train"])


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
def train(settings: SettingsDep, body: TrainRequest | None = None) -> TrainResponse:
      # El cuerpo es opcional: sin cuerpo (o sin "version") se genera un
      # identificador automático. 201 porque se crea un recurso (la versión).
      version = body.version if body is not None else None
      resultado = service.train_new_version(settings, version)
      return TrainResponse(
            version=resultado.version,
            model_mae=resultado.model_mae,
            baseline_mae=resultado.baseline_mae,
      )