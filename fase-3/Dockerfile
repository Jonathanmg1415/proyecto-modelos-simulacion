# Fase 3 — API REST (FastAPI). La imagen se construye desde la RAÍZ del
# repositorio, no desde fase-3/, porque la API reutiliza el código de fase-2
# (src/, train.py, predict.py) y debe conservarse la misma estructura relativa:
#
#   docker build -f fase-3/Dockerfile -t modelos-fase3:1.0 \
#       --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
FROM python:3.13.2-slim

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas. requirements.txt de
# fase-3 incluye las mismas versiones fijadas en fase-2 más FastAPI/uvicorn.
COPY fase-3/requirements.txt fase-3/requirements.txt
RUN pip install --no-cache-dir -r fase-3/requirements.txt

# Código de fase-2 reutilizado por la API. config.py localiza fase-2/ como
# ../fase-2 respecto a fase-3/, igual que en el repositorio.
COPY fase-2/src/ fase-2/src/
COPY fase-2/train.py fase-2/predict.py fase-2/

# Código de la API.
COPY fase-3/app/ fase-3/app/

# Datos y modelos NO se copian a la imagen: se montan en tiempo de ejecución
# en /app/fase-1/data (CSV de entrenamiento) y /app/fase-2/models (versiones),
# para que las versiones creadas con /train persistan fuera del contenedor.

# Commit con el que se construyó la imagen (.git/ no entra en la imagen); lo
# lee get_git_commit() de fase-2 para los metadatos de versiones creadas con /train.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Usuario sin privilegios: la API no debe correr como root dentro del
# contenedor. UID 1000 coincide con el primer usuario de la mayoría de
# distribuciones Linux, así las versiones que /train escriba en el volumen
# montado quedan con el dueño correcto en el equipo anfitrión.
RUN useradd --create-home --uid 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

WORKDIR /app/fase-3
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]