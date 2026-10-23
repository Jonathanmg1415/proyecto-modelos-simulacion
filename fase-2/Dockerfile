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

# Sin ENTRYPOINT: permite ejecutar "docker run <imagen> python train.py ..."
# o "python predict.py ..." directamente. No se entrena nada durante el build.
CMD ["python", "--version"]
