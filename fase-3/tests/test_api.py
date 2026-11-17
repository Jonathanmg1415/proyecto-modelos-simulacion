# Pruebas de la API (fase 3). Se ejecutan con:  python -m pytest  (desde fase-3/)
from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import EJEMPLO_VALIDO


def test_health_responde_200(client: TestClient):
      """1) GET /health responde 200."""
      respuesta = client.get("/health")
      assert respuesta.status_code == 200
      assert respuesta.json() == {"status": "ok"}


def test_predict_entrada_valida_devuelve_prediccion_y_version(client: TestClient, trained_env):
      """2) POST /predict con una entrada válida responde 200 con 'prediccion'
      numérica y la 'version' del modelo utilizada."""
      respuesta = client.post("/predict", json=EJEMPLO_VALIDO)
      assert respuesta.status_code == 200
      cuerpo = respuesta.json()
      assert set(cuerpo) == {"prediccion", "version"}
      assert isinstance(cuerpo["prediccion"], float) and cuerpo["prediccion"] > 0
      assert cuerpo["version"] == trained_env["version"]


def test_predict_campo_faltante_es_rechazado(client: TestClient):
      """3) POST /predict sin un campo obligatorio responde 4xx indicando el campo."""
      sin_milage = {k: v for k, v in EJEMPLO_VALIDO.items() if k != "milage"}
      respuesta = client.post("/predict", json=sin_milage)
      assert respuesta.status_code == 422
      campos_con_error = [err["loc"][-1] for err in respuesta.json()["detail"]]
      assert "milage" in campos_con_error


def test_predict_version_inexistente_responde_503(client: TestClient, monkeypatch):
      """Extra: si la versión activa no existe, la API responde 503 (modelo no
      disponible) en vez de un error interno."""
      monkeypatch.setenv("MODEL_VERSION", "v-no-existe")
      respuesta = client.post("/predict", json=EJEMPLO_VALIDO)
      assert respuesta.status_code == 503
      assert "v-no-existe" in respuesta.json()["detail"]