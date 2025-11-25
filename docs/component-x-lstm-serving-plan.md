# Component X LSTM – Inference & Serving Plan

Goal: expose the trained Component X LSTM classifier as a production‑ready HTTP API (`POST /predict`) running on a CPU‑only VPS (Python 3.12.2), using FastAPI and Dokploy + Docker, while reusing the existing training pipeline and feature engineering code.

This document describes how to go from the current repo to a deployable service, without actually implementing it yet.

---

## 1. Current State Recap

Training and preprocessing are already implemented in this repo:

- **Feature statistics and normalization**
  - `src/features/transformer.py`
    - `compute_feature_stats(...)` builds `artifacts/feature_stats.json` from `train_operational_readouts.csv`.
    - `FeatureTransformer` applies normalization and padding cleanup to sequence tensors.

- **Sequence construction for training**
  - `src/features/windowing.py`
    - `SequenceWindowConfig` and `build_sequences_for_training(...)`:
      - Build fixed‑length windows of length `L` (default 128) from operational data + proximity labels.
      - Zero left‑padding, with `seq_lengths` tracking the true length.
      - Save sequences as `data/train_sequences.npz` (and analogous for validation/test if extended).

- **Model architecture**
  - `src/models/lstm_classifier.py`
    - `LSTMClassifierConfig`
    - `LSTMClassifier`:
      - Bidirectional LSTM, configurable hidden size/layers/dropout/pooling.
      - Classification head for 5 proximity classes (0–4).

- **Training + checkpoints**
  - `src/training/train_lstm.py`
    - CLI for training using `.npz` sequence files and `feature_stats.json`.
    - Saves checkpoints (per Phase 7 plan):
      - `artifacts/last.pt` – last epoch state (for resume).
      - `artifacts/best.pt` – best validation macro‑F1 state.
    - Checkpoints include:
      - `model_state`, `optimizer_state`, `scheduler_state`
      - `epoch`, `best_val_f1`
      - `args` dict with hyperparameters (hidden size, layers, pooling, etc.).

This serving plan will **reuse**:

- `FeatureTransformer` + `feature_stats.json` for normalization.
- `LSTMClassifierConfig` + `LSTMClassifier` + `best.pt` for inference.

---

## 2. Target Inference Architecture

High‑level pipeline when a prediction request arrives:

1. **Raw JSON payload** from web app (time‑ordered measurements for one or more vehicles).
2. **Sequence construction**:
   - For each vehicle example:
     - Filter measurements up to `reference_time_step`.
     - Build a `(seq_len, F)` feature matrix using the canonical `feature_order` from `feature_stats.json`.
     - Left‑pad or truncate to `(L, F)` (same `window_size` as training).
     - Track `seq_length = min(seq_len, L)`.
3. **Normalization**:
   - Stack sequences into `(N, L, F)` and `seq_lengths` into `(N,)`.
   - Apply `FeatureTransformer.transform_sequences(...)`:
     - log1p+z‑norm for counter features.
     - z‑norm for histogram features.
     - Impute NaNs and zero out the padded prefix.
4. **Model inference (CPU)**:
   - Run normalized tensor through `LSTMClassifier` with `seq_lengths` (for mean pooling) or just sequences (for last pooling).
   - Compute logits → softmax → class probabilities.
5. **HTTP response**:
   - Return predicted class for each input sequence + probability vector over the 5 classes.

The serving code will be organized into two new modules:

- `src/inference/predict.py` – model + artifacts loading and pure inference utilities.
- `src/service/app.py` – FastAPI application with `/predict` endpoint using `predict.py`.

---

## 3. JSON Interface Design

The JSON contract should be explicit so the web app can construct valid requests.

### 3.1 Request Schema (Raw Measurements → Sequences)

We will design the API so the client sends **raw time‑ordered measurements**; the service is responsible for windowing and feature alignment.

**Per‑sequence item** (single vehicle at a given reference point):

- `vehicle_id` (int) – identifier (optional but useful for traceability).
- `reference_time_step` (number) – time step at which we want the prediction, consistent with training labels.
- `measurements` (list of objects):
  - Each element:
    - `time_step` (number) – time step of this measurement.
    - `features` (object) – mapping `{feature_name: value}`.
      - `feature_name` must belong to `feature_order` from `feature_stats.json`.
      - Missing features are allowed; they will be treated as NaN and imputed downstream.

**Example request (single sequence)**:

```json
{
  "sequences": [
    {
      "vehicle_id": 123,
      "reference_time_step": 8000,
      "measurements": [
        {
          "time_step": 7873,
          "features": { "171_0": 10.0, "666_0": 0.0 }
        },
        {
          "time_step": 7874,
          "features": { "171_0": 11.0, "666_0": 0.0 }
        }
      ]
    }
  ]
}
```

The FastAPI layer will map this JSON into Pydantic models; the inference code will convert it into numpy arrays.

### 3.2 Response Schema

For each input sequence, the service should respond with:

- `vehicle_id` – echoed from request (if present).
- `reference_time_step` – echoed from request.
- `predicted_class` – integer 0–4.
- `class_probs` – list of 5 floats (probabilities for classes 0–4).

**Example response**:

```json
{
  "predictions": [
    {
      "vehicle_id": 123,
      "reference_time_step": 8000,
      "predicted_class": 2,
      "class_probs": [0.01, 0.05, 0.80, 0.10, 0.04]
    }
  ]
}
```

---

## 4. Planned Inference Utilities (`src/inference/predict.py`)

This module will encapsulate:

1. **Artifact loading at import time**
   - `feature_stats_path = Path("artifacts/feature_stats.json")`
   - `checkpoint_path = Path("artifacts/best.pt")`
   - Load `FeatureTransformer` via `FeatureTransformer.from_json(feature_stats_path)` to access:
     - `feature_order` – canonical list of feature names used to order columns.
     - `per_feature` – stats for normalization.
   - Load checkpoint via `torch.load(checkpoint_path, map_location="cpu")`:
     - Extract hyperparameters from `ckpt["args"]` to construct `LSTMClassifierConfig`:
       - `input_size = len(feature_order)`
       - `hidden_size`, `num_layers`, `bidirectional`, `dropout`, `pooling` from `args`.
     - Instantiate `LSTMClassifier(config)` and load `model_state`.
     - Put model in `eval()` mode on CPU.

2. **Sequence building helper (inference side)**

Planned function (signature subject to minor adjustments during implementation):

```python
def build_sequence_for_inference(
    measurements: list[dict],
    reference_time_step: float,
    feature_order: list[str],
    window_size: int,
) -> tuple[np.ndarray, int]:
    """
    Convert raw measurements into a single fixed-length sequence.

    Returns:
        seq: (L, F) float32 array (padded/truncated)
        seq_length: int, number of real (non-padded) steps
    """
```

Responsibilities:

- Sort `measurements` by `time_step` ascending.
- Filter to `time_step <= reference_time_step`.
- For each remaining measurement:
  - Build a row vector of length `F` using `feature_order`:
    - If `feature_name` is missing in `features`, set to `NaN` (to be imputed by `FeatureTransformer`).
- Stack into an array of shape `(seq_len, F)`.
- Pad/truncate to `(L, F)`:
  - If `seq_len >= L`: keep last `L` rows.
  - If `seq_len < L`: create a `(L, F)` array with zeros, then copy `seq_len` rows at the **end** (left‑padding zeros).
- Set `seq_length = min(seq_len, L)`.

3. **Batch preparation helper**

Another planned function:

```python
def prepare_batch(
    sequences_payload: list[SequencePayload],
    feature_order: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build a batch from multiple SequencePayloads.

    Returns:
        sequences: (N, L, F)
        seq_lengths: (N,)
        metadata: list of dicts (vehicle_id, reference_time_step) aligned with batch index
    """
```

Where `SequencePayload` is a simple data structure reflecting the request item (Pydantic or equivalent).

Responsibilities:

- For each payload item:
  - Call `build_sequence_for_inference(...)`.
  - Collect `seq`, `seq_length`, and metadata (`vehicle_id`, `reference_time_step`).
- Stack into `sequences` and `seq_lengths` arrays.

4. **Core prediction function**

Finally, the module will expose a single API for the FastAPI layer:

```python
def predict_sequences(
    sequences: np.ndarray,
    seq_lengths: np.ndarray,
) -> dict:
    """
    Apply FeatureTransformer + model to a batch of sequences.

    Returns:
        {
          "preds": list[int],
          "probs": list[list[float]],
        }
    """
```

Steps:

- Normalize:
  - `sequences_norm = feature_tx.transform_sequences(sequences, seq_lengths)`.
- Convert to PyTorch tensors on CPU:
  - `x = torch.from_numpy(sequences_norm)`
  - `lengths = torch.from_numpy(seq_lengths)`
- Forward pass with `torch.no_grad()`:
  - `logits = model(x, lengths)` if `pooling == "mean"`.
  - `logits = model(x)` if `pooling == "last"` (still passing lengths is fine but not required by the current implementation).
- Softmax to probabilities:
  - `probs = torch.softmax(logits, dim=1)` → shape `(N, 5)`.
- Argmax for class predictions:
  - `preds = probs.argmax(dim=1)`.
- Return CPU numpy → Python lists for JSON serialization.

---

## 5. FastAPI Service Design (`src/service/app.py`)

This module will define the HTTP interface around `predict_sequences`.

### 5.1 Pydantic Models (planned)

```python
class Measurement(BaseModel):
    time_step: float
    features: dict[str, float]

class SequenceItem(BaseModel):
    vehicle_id: int
    reference_time_step: float
    measurements: list[Measurement]

class PredictRequest(BaseModel):
    sequences: list[SequenceItem]

class SequencePrediction(BaseModel):
    vehicle_id: int
    reference_time_step: float
    predicted_class: int
    class_probs: list[float]

class PredictResponse(BaseModel):
    predictions: list[SequencePrediction]
```

### 5.2 Endpoint Behavior (`POST /predict`)

1. Receive `PredictRequest` and validate JSON structure.
2. Extract the list of `SequenceItem`s.
3. Call `prepare_batch(...)` from `predict.py` to obtain:
   - `sequences: (N, L, F)`
   - `seq_lengths: (N,)`
   - `metadata: list[dict]` (vehicle_id, reference_time_step per index)
4. Call `predict_sequences(sequences, seq_lengths)`.
5. Build `PredictResponse` by combining:
   - Metadata (vehicle_id, reference_time_step) from `prepare_batch`.
   - `preds` and `probs` from `predict_sequences`.
6. Return JSON response.

### 5.3 Error Handling & Validation Notes

- If any `SequenceItem` yields no measurements with `time_step <= reference_time_step`, the service should:
  - Either skip that item and return an error entry in the response, or
  - Return an HTTP 400 with a descriptive error.
- If feature vectors are inconsistent with `feature_order` (e.g., completely unknown feature names), the service should:
  - Ignore unknown keys.
  - Treat missing known features as `NaN` (to be imputed by `FeatureTransformer`).

---

## 6. Docker & Dokploy Strategy

### 6.1 CPU‑Only Docker Image

Constraints:

- Python 3.12.2.
- CPU‑only PyTorch.
- FastAPI + Uvicorn.

Key points for the Dockerfile (to be implemented later):

- Base image: `python:3.12-slim` (or similar).
- Install system build tools only if required by dependencies.
- Copy `requirements.txt` and install Python dependencies, including:
  - `torch` (CPU wheel)
  - `fastapi`
  - `uvicorn[standard]`
  - `pydantic`
- Copy the repo into `/app` and set `WORKDIR /app`.
- Expose port `8000` and set entrypoint to:
  - `uvicorn src.service.app:app --host 0.0.0.0 --port 8000`.

### 6.2 Model Artifact Storage Options

Two strategies are considered; both are valid.

#### Option A – Bake artifacts into the image (simple baseline)

- At build time, copy `artifacts/feature_stats.json` and `artifacts/best.pt` into the image.
- `predict.py` expects these paths (e.g., `/app/artifacts/...`).
- Updating the model requires a new image build and redeployment (new tag).
- Pros:
  - Simple Dokploy configuration, no extra volumes.
  - Fully self‑contained image.
- Cons:
  - Each model update requires rebuilding and redeploying the image.

#### Option B – Mount artifacts via Dokploy volume (flexible updates)

- Store artifacts on the host VPS, e.g. `/opt/componentx/artifacts`.
- Configure Dokploy to mount this host directory into the container at `/app/artifacts`.
- `predict.py` reads artifacts from `/app/artifacts`.
- Updating the model becomes:
  - Copy new `best.pt` and `feature_stats.json` to `/opt/componentx/artifacts`.
  - Restart the container via Dokploy (no image rebuild required).
- Pros:
  - Faster iteration on models.
  - Cleaner separation between code image and model data.
- Cons:
  - Slightly more infra complexity (volume configuration).

**Recommended path**:

- Start with **Option A (baked artifacts)** to validate end‑to‑end deployment.
- Once stable, consider migrating to **Option B (volume)** if frequent model updates become necessary.

### 6.3 Dokploy Configuration (High‑Level)

Dokploy configuration (exact syntax depends on Dokploy version) should specify:

- Build:
  - Dockerfile path at the repo root.
  - Build context: this repository directory.
  - Image tag: e.g., `componentx-lstm-api:latest`.
- Runtime:
  - Container port: `8000`.
  - Host port or reverse proxy integration (e.g., via Nginx/Caddy).
  - Environment variables, if any (e.g., `PYTHONUNBUFFERED=1`).
  - Optional volume mapping for artifacts if using Option B.

---

## 7. Integration with the Web App (Same VPS)

The web app and the model service will run on the same VPS, but as separate processes (and likely separate Docker containers):

- The web app calls the model service via HTTP `POST /predict`:
  - Either by IP/port (e.g., `http://model-service:8000/predict` on the Docker network).
  - Or via a reverse‑proxy path (e.g., `https://mydomain.com/componentx/predict`).
- The web app is responsible for:
  - Fetching or constructing time‑ordered measurement data per vehicle.
  - Assembling the JSON payload according to the request schema in Section 3.1.
  - Handling errors (HTTP 4xx/5xx) and displaying/logging them appropriately.

This separation keeps the model lifecycle (retraining, artifact updates) independent from the web app deployment, while still allowing low‑latency intra‑VPS communication.

---

## 8. Future Implementation Checklist

When ready to implement, follow these steps in this repository:

1. **Inference utilities**
   - Implement `src/inference/predict.py` as per Section 4.
   - Ensure it can be imported and used from a simple Python REPL for smoke testing.

2. **FastAPI service**
   - Implement `src/service/app.py` with models and `/predict` endpoint as per Section 5.
   - Test locally with `uvicorn` and sample JSON payloads.

3. **Dependencies**
   - Update `requirements.txt` to include FastAPI, Uvicorn, Pydantic, and CPU‑only PyTorch.

4. **Dockerfile**
   - Create a Dockerfile matching Section 6.1.
   - Build and run locally to verify the containerized service.

5. **Dokploy setup**
   - Add Dokploy app configuration pointing to this repo and Dockerfile.
   - Decide on artifact strategy (baked vs volume) and configure accordingly.

6. **Web app integration**
   - Update the web app to call the `/predict` endpoint with the agreed JSON schema.
   - Implement retries, timeouts, and basic error handling.

Once these steps are complete, the Component X LSTM classifier will be accessible as a robust, HTTP‑based prediction service on the existing VPS.

