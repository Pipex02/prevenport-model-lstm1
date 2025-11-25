# Integración del dataset SCANIA Component X con el pipeline LSTM

Pregunta a responder:  
**“¿Cómo integraron el dataset SCANIA comp X con el pipeline (preprocesos, selección de variables, particiones usadas)?”**

---

## 1. Particiones usadas

Se respetó la partición oficial del dataset:

- **Train**
  - `train_operational_readouts.csv` – lecturas operacionales por vehículo y `time_step`.
  - `train_tte.csv` – etiquetas de tiempo hasta evento (TTE) por vehículo.
  - `train_specifications.csv` – 8 variables categóricas por vehículo (`Spec_0`–`Spec_7`).
- **Validación**
  - `validation_labels.csv` – una etiqueta de proximidad a fallo (`class_label` ∈ {0,1,2,3,4}) por vehículo.
- **Test**
  - Se asume estructura análoga (no integrada aún al pipeline, pero soportada por diseño).

Para entrenar el modelo se construye un conjunto de entrenamiento enriquecido a partir de los archivos de **train**, y se respetan las etiquetas oficiales de validación para evaluar.

---

## 2. Preprocesos sobre train: de TTE a etiquetas 5-clases

### 2.1 Construcción de etiquetas de proximidad (Phase 2)

A partir de `train_tte.csv` y `train_operational_readouts.csv` se generan etiquetas de 5 clases que imitan la lógica de validación/test:

- Se define **time-to-failure (TTF)** por lectura como:
  - `TTF = length_of_study_time_step - time_step`
- Ventanas de TTF → clase:
  - Clase 4: `0 <= TTF <= 6`
  - Clase 3: `6 < TTF <= 12`
  - Clase 2: `12 < TTF <= 24`
  - Clase 1: `24 < TTF <= 48`
  - Clase 0: `TTF > 48` o vehículo censurado (sin fallo observado).

Implementación (archivo `src/data/labels.py`):

- **Vehículos con evento** (`in_study_repair = 1` en `train_tte.csv`):
  - Para cada vehículo se leen sus `time_step` desde `train_operational_readouts.csv`.
  - Se calculan los TTF, se descartan los TTF ≤ 0 (lecturas posteriores al fallo).
  - Para cada clase no cero (1–4) se selecciona aleatoriamente **un** `reference_time_step` real cuya TTF caiga en la ventana correspondiente.
- **Vehículos censurados** (`in_study_repair = 0`):
  - Se genera una sola etiqueta por vehículo:
    - `reference_time_step = length_of_study_time_step`
    - `class_label = 0`

El resultado se guarda en:

- `data/train_proximity_labels.csv` con columnas:  
  `vehicle_id, reference_time_step, time_to_failure, class_label`.

---

## 3. Selección de variables de entrada

### 3.1 Señales operacionales

De `train_operational_readouts.csv` se utilizan:

- Identificadores:
  - `vehicle_id`
  - `time_step`
- **8 contadores** (señales acumulativas de uso):
  - `171_0, 666_0, 427_0, 837_0, 309_0, 835_0, 370_0, 100_0`
- **97 bins de histogramas** (6 variables base: 167, 272, 291, 158, 459, 397):
  - `167_0`–`167_9`
  - `272_0`–`272_9`
  - `291_0`–`291_10`
  - `158_0`–`158_9`
  - `459_0`–`459_19`
  - `397_0`–`397_35`

En total, **105 features** por `time_step`.  
Estas son las entradas dinámicas del LSTM.

### 3.2 Especificaciones (features estáticos)

De `train_specifications.csv` se dispone de:

- `vehicle_id`
- `Spec_0`–`Spec_7` (8 variables categóricas).

En la versión actual del pipeline todavía **no se inyectan** en el modelo, pero el diseño permite añadirlas después como embeddings estáticos concatenados a la representación de secuencia.

---

## 4. Construcción de secuencias temporales (ventanas LSTM)

### 4.1 Definición de ventana (Phase 3)

Para cada fila de `train_proximity_labels.csv` se construye una secuencia:

- Ventana de longitud fija **L = 128** pasos de tiempo.
- Ventana de **pasado solamente**, anclada en `reference_time_step`:
  - Se toman todas las filas de `train_operational_readouts.csv` con:
    - mismo `vehicle_id` y
    - `time_step <= reference_time_step`.
  - Si hay más de 128 lecturas, se conservan **las 128 más recientes**.
  - Si hay menos de 128 lecturas, se toma todo el historial disponible.

### 4.2 Padding y longitud efectiva

Si la secuencia tiene menos de 128 pasos:

- Se hace **padding al inicio** con el valor 0 en todas las features.
- Se almacena la longitud real (`seq_length`) antes de padding.

Salida de esta etapa (implementada en `src/features/windowing.py`):

- Tensores de secuencias: `sequences` de forma `(N, 128, 105)`.
- Etiquetas: `labels` (`class_label` ∈ {0,1,2,3,4}).
- Longitudes reales: `seq_lengths` (`N`, número de pasos no rellenados).
- Metadatos opcionales: `vehicle_ids`, `reference_time_step`.

Ejemplo de comando:

```bash
.\.venv\Scripts\python.exe -m src.features.windowing ^
  --operational data/train_operational_readouts.csv ^
  --labels data/train_proximity_labels.csv ^
  --output data/train_sequences.npz ^
  --window-size 128
```

---

## 5. Preprocesos / normalización

### 5.1 Cálculo de estadísticas en train (Phase 4)

Para evitar fuga de información, las estadísticas se calculan **solo con train**:

- Archivo base: `data/train_operational_readouts.csv`.
- Implementación: `src/features/transformer.py` (CLI incluido).
- Estrategia:
  - Se lee el CSV por chunks (`chunksize=200_000`) para no cargar todo en memoria.
  - Para cada feature:
    - Si es contador:
      - Se aplica `log1p` y se calcula media y desviación estándar en el espacio log.
    - Si es histograma:
      - Se calcula media y desviación estándar sobre el valor original.
  - Se guardan:
    - `feature_order` (orden exacto de columnas de features).
    - `counters`, `histograms`.
    - `per_feature[col] = {transform, mean, std}`.
- Salida: `artifacts/feature_stats.json`.

Comando de ejemplo:

```bash
.\.venv\Scripts\python.exe -m src.features.transformer ^
  --operational data/train_operational_readouts.csv ^
  --output artifacts/feature_stats.json
```

### 5.2 Aplicación de normalización a las secuencias

La clase `FeatureTransformer` aplica estas estadísticas a cualquier tensor de secuencias:

- `FeatureTransformer.from_json("artifacts/feature_stats.json")`
- `transform_sequences(sequences, seq_lengths)`
  - Para contadores:
    - `log1p` → imputación de NaN con la media → z-score.
  - Para histogramas:
    - imputación de NaN con la media → z-score.
  - Finalmente:
    - Para cada secuencia, se pone a 0 la parte de padding (los primeros `L - seq_length` pasos), de forma que:
      - El modelo ve siempre padding igual a **0** en todas las features.

Esto permite:

- Entrenar el LSTM sobre secuencias ya normalizadas y con padding neutro.
- Reutilizar el mismo transformador en validación/test sin recalcular estadísticas.

---

## 6. Resumen

La integración del dataset SCANIA Component X con el pipeline se hace así:

1. **Particiones:** se usan los archivos oficiales de train y validación; train se expande con etiquetas 5-clases derivadas de TTE.
2. **Preprocesos:**
   - De TTE → etiquetas de proximidad (`train_proximity_labels.csv`).
   - Construcción de secuencias de longitud fija L=128 alrededor de cada `reference_time_step`, con padding inicial en 0.
3. **Selección de variables:**
   - Como entradas dinámicas del LSTM se utilizan las 105 columnas de señales de `train_operational_readouts.csv` (8 contadores + 97 bins de histograma).
   - Especificaciones (`Spec_0`–`Spec_7`) se dejan preparadas para integrarse más adelante como features estáticos.
4. **Normalización:**
   - Estadísticas (media/std) y tipo de transformación se calculan solo sobre train.
   - Contadores se tratan como `log1p + z-score`; histogramas con `z-score`.
   - El `FeatureTransformer` aplica estas transformaciones a las secuencias y garantiza padding a 0.

Con esto, el dataset queda completamente alineado con el pipeline de modelado: desde CSV crudo hasta tensores normalizados listos para alimentar un LSTM de clasificación multiclase (5 clases de proximidad a fallo).

