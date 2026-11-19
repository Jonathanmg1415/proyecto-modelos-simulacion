# Fase 4 — Monitoreo

Proyecto Integrador · Modelos y Simulación de Sistemas I · Grupo B · Universidad de Antioquia
Docente: Andrés Parra

**Integrantes:** Jonathan Mazo González, Juan Carlos Gómez Jaramillo

## Relación con la fase 3

Esta fase agrega un registro de uso a la API REST de [`fase-3/`](../fase-3/README.md)
y un script que calcula indicadores de operación a partir de ese registro. No
se modifica el modelo ni el algoritmo: se conserva el de las fases 1 y 2
(Gradient Boosting).

Los cambios sobre `fase-3/` son mínimos y quedan aislados en:

- `fase-3/app/registro.py` (nuevo): agrega una fila al registro por cada
  llamada a `/predict`.
- `fase-3/app/main.py` y `fase-3/app/routers/predict.py`: cada desenlace de
  `/predict` (200, 400, 422, 503) llama a `registrar_solicitud` antes de
  responder. `/health` y `/train` no se registran: la especificación pide
  auditar únicamente `/predict`.

## 1. Contenido del registro

Cada llamada a `POST /predict` —incluidas las rechazadas por validación—
agrega una fila a `fase-4/data/solicitudes.csv`, con este encabezado:

```
fecha_hora,estado_http,version,prediccion
```

- `fecha_hora`: marca de tiempo local, formato ISO 8601 con segundos.
- `estado_http`: código HTTP de la respuesta (200, 400, 422, 503…).
- `version`: versión del modelo que produjo la predicción; vacío si la
  solicitud no llegó a usar un modelo (por ejemplo, un 422).
- `prediccion`: valor estimado; vacío en un error.

El archivo se abre en modo de agregar y conserva el historial entre
ejecuciones; si no existe, se crea con encabezado en la primera llamada.

La ruta es configurable con la variable de entorno `REQUEST_LOG_PATH` (valor
por defecto: `fase-4/data/solicitudes.csv`, relativo a la raíz del
repositorio).

## 2. Script de monitoreo

`monitoreo.py` no tiene dependencias externas (solo biblioteca estándar:
`argparse`, `csv`, `pathlib`); corre con cualquier Python 3.11–3.13, incluso
sin el entorno de fase-2/fase-3 instalado. Las pruebas en `tests/` sí
necesitan `pytest` (ya está en `fase-3/requirements.txt`).

```bash
# Desde fase-4/
python monitoreo.py
python monitoreo.py --help
python monitoreo.py --log ruta/al/registro.csv

python -m pytest -v
```

Lee el registro y muestra dos indicadores:

- **Total de solicitudes**: número de filas del registro.
- **Porcentaje de fallos**: proporción de filas con `estado_http` entre 400 y
  599. Si el registro no existe o está vacío, muestra `0` e indica
  «sin solicitudes».

## 3. Reporte

*Ensayo real, ejecutado el 18 de noviembre de 2026 contra la API local en
`http://127.0.0.1:8099`, con `fase-2/models/v1` y `fase-2/models/v2` ya
entrenadas en la fase 2.*

Solicitudes enviadas con `fase-3/client/client.py`:

1. `POST /predict` con un vehículo válido (INFINITI FX37, API con
   `MODEL_VERSION=v1`) → 200, predicción 20&nbsp;453,08 USD.
2. `POST /predict` con otro vehículo válido (Ford E350, misma versión activa)
   → 200, predicción 13&nbsp;815,51 USD.
3. `POST /predict` al primer vehículo, sin el campo `milage` → 422, con el
   campo señalado en el mensaje de error.
4. `POST /predict` con un vehículo válido, API reiniciada con
   `MODEL_VERSION=v2` → 200, predicción 22&nbsp;935,60 USD (confirma que la
   API puede servir cualquiera de las dos versiones guardadas).

Registro resultante (incluido en el repositorio como muestra, en
`fase-4/data/solicitudes.csv`):

```
fecha_hora,estado_http,version,prediccion
2026-11-18T21:37:00,200,v1,20453.082267572456
2026-11-18T21:37:00,200,v1,13815.505268955552
2026-11-18T21:37:01,422,,
2026-11-18T21:37:48,200,v2,22935.604276990955
```

Resultado de `python monitoreo.py`:

```
Total de solicitudes : 4
Porcentaje de fallos : 1/4 (25,0 %)
```

**Interpretación.** La API atendió cuatro solicitudes: tres predicciones
válidas (dos con la versión `v1` y una con `v2`, lo que confirma que el
cambio de versión activa funciona) y un rechazo esperado por la falta del
campo `milage`, parte deliberada del ensayo. La tasa de fallos (25,0&nbsp;%)
describe el funcionamiento del servicio en este ensayo puntual, no la calidad
de las predicciones: para evaluar eso se necesitaría conocer el precio real
de venta de esos vehículos, información que no está disponible en el momento
de predecir.

También se ejecutó `POST /train` sin cuerpo durante el ensayo: generó la
versión `v-20261118-213715` (MAE 16&nbsp;195,69 USD, igual al reportado en
fase 1 y fase 2 para el mismo algoritmo y semilla), lo que confirma que
entrenar desde la API produce una versión nueva e identificable sin afectar
`v1` ni `v2`. Esa versión de prueba no se conserva en el repositorio, para no
duplicar artefactos ya documentados en fase 2; el comando y la respuesta
quedan como evidencia en [`fase-3/README.md`](../fase-3/README.md), sección 6.

## 4. Versiones conservadas y su selección

| Versión | Semilla | MAE (test, fase 2) | Activación |
|---|---:|---:|---|
| `v1` (activa por defecto) | 42 | 16&nbsp;195,69 USD | sin configurar nada, o `MODEL_VERSION=v1` |
| `v2` | 7 | 10&nbsp;384,91 USD\* | `MODEL_VERSION=v2` (variable de entorno de la API) |

\* `v1` y `v2` usan particiones train/test distintas (cada una con su propia
semilla), así que sus MAE no son directamente comparables sin tener en cuenta
ese cambio — ver [`fase-2/README.md`](../fase-2/README.md), sección 4.

Para volver a la versión anterior basta con reiniciar la API (o el
contenedor) con `MODEL_VERSION=v1`; no hace falta reentrenar ni modificar
código.

## 5. Política de reentrenamiento

Cuando se acumulen al menos 200 anuncios nuevos con el precio de venta real
conocido (no solo publicado), el equipo ejecutará `POST /train` para generar
una versión candidata con esos datos agregados al conjunto de entrenamiento.
Se comparará el MAE de la candidata y el de la versión activa sobre el mismo
conjunto de prueba reservado (el 20&nbsp;% fijado por `--test-size` en
fase 2), y la candidata se activará, cambiando `MODEL_VERSION`, **solo si su
MAE es menor**; si no mejora, se conserva la versión activa y se documenta el
resultado de la comparación antes de descartar la candidata.

## 6. Reproducción del ensayo

1. Levantar la API de `fase-3/` (local o en Docker; ver
   [`fase-3/README.md`](../fase-3/README.md)), con `fase-4/data/` disponible
   (montada, en el caso de Docker) para que el registro persista.
2. Enviar solicitudes con `fase-3/client/client.py` (incluir al menos una con
   `--sin-campo <campo>` para generar un rechazo).
3. Ejecutar `python monitoreo.py` desde `fase-4/` y comparar el resultado con
   las filas de `fase-4/data/solicitudes.csv`.

## Fuentes

- Especificación técnica de las fases 3 y 4, proporcionada por el docente del curso.
- Reutiliza el código y las versiones del modelo de [`fase-2/`](../fase-2/README.md)
  (ver su README para el detalle del algoritmo, la métrica y las limitaciones
  conocidas).
