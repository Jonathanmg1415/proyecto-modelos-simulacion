from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
      status: str = "ok"
      
class PredictRequest(BaseModel):
      model_year: int = Field(..., description="Año del modelo, p. ej. 2018")
      milage: str = Field(..., description="Texto numérico, p. ej. '51,000 mi.'")
      engine: str = Field(..., description="p. ej. '300.0HP 2.0L 4 Cylinder Engine'")
      transmission: str = Field(..., description="p. ej. '7-Speed A/T'")
      brand: str
      fuel_type: str
      accident: Optional[str] = Field(None, description="Nulo -> 'Unknown'")
      clean_title: Optional[str] = Field(None, description="'Yes' o nulo")
      ext_col: str
      int_col: str
      
class PredictResponse(BaseModel):
      prediccion: float
      version: str

class TrainRequest(BaseModel):
      version: Optional[str] = Field(None, description="Identificador de versión, si se omite, se genera automáticamente")
      
class TrainResponse(BaseModel):
      version: str
      model_mae: float
      baseline_mae: float