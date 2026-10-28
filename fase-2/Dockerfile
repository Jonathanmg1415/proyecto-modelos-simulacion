# Fase 2 — Scripts y Docker. Imagen para ejecutar train.py y predict.py.
FROM python:3.13.2-slim

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas de Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación. No se copian data/, models/ ni outputs/: se
# proporcionan en tiempo de ejecución mediante montajes (-v), para que los
# datos de entrada y los artefactos generados persistan fuera del contenedor.
COPY src/ src/
COPY train.py predict.py ./

# Referencia al commit de git con el que se construyó la imagen. .git/ se
# excluye de la imagen (ver .dockerignore), así que no se puede leer con
# "git rev-parse HEAD" dentro del contenedor: se recibe como build-arg.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Sin ENTRYPOINT: permite ejecutar "docker run <imagen> python train.py ..."
# o "python predict.py ..." directamente. No se entrena nada durante el build.
CMD ["python", "--version"]
