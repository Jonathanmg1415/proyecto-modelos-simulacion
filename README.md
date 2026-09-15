# proyecto-modelos-simulacion
## Used Car Price Prediction

## Integrantes del equipo:
  * Jonathan Mazo González
  
  * Juan Carlos Gómez Jaramillo

## Descripción del Problema
El mercado de autos usados funciona con una considerable asimetría de la informción entre el estado real de los vehiculos, estado que es conocido por el vendedor, pero el comprador solo conoce lo que el anuncio de compra - venta muestra (marca, modelo, año, kilometraje, tipo de combustible que usa, historial de accidentes, entre otros), y esto se convierte en la unica manera de verificar si el precio es justo o no. Esta brecha genera dos problemas, uno el comprador puede pagar más de lo justo por un vehiculo por falta de un punto de referencia, y los vendedores no siempre tienen una forma de sugerir un precio competitivo basadis en las caracteristicas reales del vehiculo.
Este dataset representa un problema de `regresion`, la variable objetivo númerica definida `price`, y un conjunto de atributos observables del vehiculo, que explican buena parte de la variabliidad en el precio. Es por esto que consideramos razonable que caracteristicas como el año del modelo, el kilometraje, la marca, el tipo de combustible o el historial de accidentes tengan una relación real y medible con el precio de venta, haciendo viable la construcción de un modelo predictivo a partir de ellas.

## Fuente del conjunto de datos:
Used Car Price Prediction Dataset, disponible en Kaggle, publicado por el usuario taeefnajib bajo licencia CC BY 4.0. Según la descripción del autor, el conjunto de datos fue extraído del sitio cars.com y contiene 4.009 anuncios individuales de vehículos, cada uno descrito por nueve características originales: marca y modelo, año, kilometraje, tipo de combustible, motor, transmisión, colores exterior e interior, historial de accidentes, estado del título ("clean title") y precio.

## objetivo del modelo:
Desarrollar un modelo predictivo capaz de estimar el precio de un vehículo usado a partir de sus atributos (marca, modelo, año, kilometraje, tipo de combustible, motor, transmisión, colores, historial de accidentes y estado del título), siguiendo un proceso reproducible, documentado y libre de fuga de información.

## Algoritmo utilizado
`RandomForestRegressor` de scikit-learn (ensamble de árboles), envuelto en
`TransformedTargetRegressor` para entrenar sobre `log1p(price)` y devolver
predicciones ya en dólares. Se compara además con un `DummyRegressor`
(mediana, modelo base) y con `HistGradientBoostingRegressor` como segundo
punto de referencia. Todo el flujo (limpieza determinista de `engine`,
`transmission`, `milage`, `price` → imputación → codificación one-hot →
modelo) vive dentro de un único `Pipeline`, ajustado solo con el 80 % de
entrenamiento para evitar fuga de información.

## Métrica empleada
**MAE** (error absoluto medio, en USD) como métrica principal, acompañada de
RMSE y R². Se reportan también MAE y R² restringidos al mercado masivo
(`price < 200 000`, ~98 % de las filas) porque unos pocos vehículos de
colección de precio millonario distorsionan cualquier métrica basada en el
error al cuadrado.

## Principales resultados obtenidos
Sobre el 20 % de prueba: el Random Forest reduce el MAE en **~49 %** frente al
baseline (de ~$32.866 a ~$16.631). En el mercado masivo (`price < 200k`) el
MAE es de ~$7.170 con R² ≈ 0.845. El R² global (~0.11) es bajo por el efecto
de los outliers de gama alta, pero sobre `log1p(price)` —lo que el modelo
optimiza— el R² es ≈ 0.82. Las variables más influyentes son kilometraje,
potencia del motor (HP), antigüedad y cilindrada.

## Instrucciones para ejecutar el notebook
Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # abrir fase-1/notebook.ipynb y "Restart & Run All"
```

O de forma no interactiva (reproduce el notebook y regenera `modelo.joblib`):

```bash
cd fase-1
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Requiere Python 3.11–3.13. El notebook fija `RANDOM_STATE = 42` y corre de
principio a fin sin intervención manual (el montaje de Google Drive se omite
automáticamente fuera de Colab).
