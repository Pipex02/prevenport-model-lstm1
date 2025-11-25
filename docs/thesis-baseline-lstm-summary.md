# AI Approach and Methodology (Component X LSTM Baseline)

The goal of this work is to build a predictive maintenance classifier for the SCANIA Component X dataset that estimates how close a vehicle is to a component failure. Rather than predicting a continuous time-to-failure directly, we cast the problem as a 5‑class classification task, where each class corresponds to a time-to-failure window (e.g., 0–6, 6–12, 12–24, 24–48, and more than 48 operational time steps before failure). This framing matches the official validation and test labels provided with the dataset and aligns naturally with cost‑sensitive decision making: the model is explicitly encouraged to recognize imminent failures, while still distinguishing “far from failure” states.

From an AI perspective, the key design choices are:

- **Sequence‑based modelling.** Component X is observed as an irregular, multivariate time series per vehicle. Each record contains counters (cumulative usage measures) and histogram‑like features (summaries over operating conditions). Instead of treating each record as an independent sample, we model each vehicle as a sequence of observations and train a recurrent neural network—specifically, an LSTM—to learn temporal dynamics such as gradual wear and pre‑failure shifts in histogram shapes.

- **Label engineering from time‑to‑event data.** The original training labels are in a time‑to‑event format: for each vehicle we know its observation horizon and whether a repair/failure occurred at that horizon. We combine this survival‑style information with the operational time series to derive multiple labelled points along each vehicle’s trajectory. Time-to-failure is computed relative to selected reference points, and then assigned to one of the five proximity classes using the same windows as the official validation/test splits. This creates richer supervision while preserving label semantics.

- **Use of both event and censored vehicles.** Vehicles with observed failures contribute multiple “near‑failure” examples at different proximity windows. Censored vehicles (no failure observed within the study horizon) are treated as “far from failure” at the end of their observation period, contributing class‑0 examples. This mirrors real operational data where most assets do not fail during the observation window and avoids discarding a large amount of information.

- **Fixed‑length windows around reference points.** For each labelled reference point, we construct a fixed‑length temporal window of preceding operational data (128 time steps). This captures both recent behavior and longer‑term trends, while keeping the model input size tractable. When a vehicle has fewer than 128 prior observations, we pad the window at the start, so the tail of the sequence always corresponds to real data.

- **Feature engineering and normalization focused on stability.** The model sees all 14 anonymized signals: eight counters and six histogram families expanded into bins. Counters are strictly non‑negative and often heavy‑tailed, so we apply a log‑transform followed by standardization to stabilize their scale. Histogram bins are standardized directly. All statistics (means and variances) are estimated on the training split only, then reused for validation and test, to avoid data leakage.

- **Imbalance‑aware training.** Both the original time‑to‑event labels and the 5‑class proximity labels are highly imbalanced, with “far from failure” dominating and truly imminent failures rare. The design anticipates this by:
  - constructing multiple labelled examples around failures to increase coverage of near‑failure classes, and  
  - supporting class‑weighted loss and/or class‑weighted sampling in the training loop to balance each mini‑batch in a principled way.

- **Reproducible, modular architecture.** Each step—label construction, sequence building, feature transformation, model definition, and training—is encapsulated as a separate module. The architecture is parameterised (e.g., window length, hidden size, pooling strategy), making it straightforward to explore alternatives such as different sequence lengths, pooling methods, or model architectures in later experiments.

Overall, the methodology reflects a time‑series classification approach grounded in survival information, tailored to the realities of predictive maintenance: strong class imbalance, censored data, irregular sampling, and the need to exploit sequential context rather than static snapshots.

---

# How This Works (Pipeline Overview)

Conceptually, the system transforms raw SCANIA Component X data into LSTM‑ready training examples through a sequence of stages.

## 1. Data partitioning and inputs

The pipeline respects the official dataset splits. The training split provides:

- **Operational readouts**: per‑vehicle, per‑time‑step measurements (counters and histograms).  
- **Time‑to‑event labels**: for each vehicle, an observation horizon and an indicator of whether a repair occurred at that horizon.  
- **Vehicle specifications**: eight categorical descriptors per vehicle (currently reserved for future use).

The validation split provides one 5‑class proximity label per vehicle, which defines the target semantics.

## 2. Label construction (training proximity labels)

For each vehicle in the training split, we derive one or more 5‑class labels:

- **Vehicles with observed failures**  
  - Treat the reported horizon as the failure time.  
  - For all time steps before failure, compute time‑to‑failure.  
  - Assign a proximity class based on which time window the time‑to‑failure falls into.  
  - For each non‑zero class, select a representative reference time step, creating up to four labelled examples per failing vehicle.

- **Censored vehicles (no failure observed)**  
  - Take the end of the observation window as a reference time.  
  - Assign class 0, representing “far from failure or no failure observed”.

This produces a training label table keyed by vehicle identifier and reference time, with an associated 5‑class label. It substantially increases the number of informative examples around failures without altering the definition of the classes.

## 3. Sequence construction around reference points

For each labelled reference point, we construct a fixed‑length input sequence:

- Collect all operational records for that vehicle up to and including the reference time.  
- If there are more than 128 such records, keep the 128 most recent; if fewer, keep all available records.  
- Left‑pad the sequence with zeros so that every example is represented as a 128×F matrix (F being the number of features), and the end of the sequence always corresponds to a real observation.  
- Record the true (unpadded) sequence length alongside each example.

The result is a set of aligned, fixed‑length multivariate time series windows, each paired with a proximity label and a sequence length. These windows reflect how the system has evolved up to a particular point in time for a specific vehicle.

## 4. Feature transformation and normalization

To ensure stable and comparable inputs across vehicles and time:

- Analyse the training operational data to compute per‑feature statistics:
  - For counter features, work in the log‑transformed space, capturing typical growth rates and variability.  
  - For histogram bins, work in the original scale, capturing typical occupancy levels and variability.
- Using these statistics, define a transformer that:
  - Applies the appropriate transform (log‑plus‑standardisation for counters, standardisation for histograms).  
  - Imputes any remaining missing values using the feature mean.  
  - Restores padding positions to exactly zero, so that padding is neutral in the normalized feature space.
- Apply the same fitted transformer to all training, validation, and test sequences, guaranteeing consistent preprocessing without leakage.

## 5. Dataset and batching abstraction

The normalized sequences and labels are exposed through a dataset abstraction:

- Each dataset item provides:
  - A sequence matrix (time steps × features).  
  - The corresponding class label.  
  - The true sequence length (needed for some pooling strategies).
- A batching component groups items into mini‑batches, optionally using class‑weighted sampling so that rare classes appear more frequently during training. This is particularly important given the severe class imbalance.

## 6. LSTM classifier architecture

The model itself is a sequence classifier:

- A multi‑layer LSTM processes each sequence of feature vectors, capturing temporal dependencies in both counters and histogram shapes.  
- The sequence is summarised into a fixed‑dimensional representation by pooling over time:
  - The default strategy uses the last hidden state (corresponding to the most recent observation), taking advantage of left‑padding.  
  - A mean‑over‑time strategy is also supported, aggregating only over real (unpadded) time steps using the stored sequence lengths.
- A small feed‑forward network maps this sequence representation to a 5‑dimensional logit vector, one dimension per proximity class.

## 7. Training loop, loss and metrics

The training procedure combines the components above into a reproducible loop:

- Each iteration draws a batch of sequences and labels, applies the transformer (if not pre‑applied), and passes the batch through the LSTM classifier.  
- The model is optimised via cross‑entropy loss; class weights or sampling strategies can be used to counterbalance class‑0 dominance.  
- Performance is monitored using:
  - Overall accuracy.  
  - Macro‑averaged F1‑score, which weights each class equally and is more informative in the presence of imbalance.
- The system maintains checkpoints:
  - A “last” checkpoint that reflects the most recent training state.  
  - A “best” checkpoint selected by validation macro F1, suitable for later evaluation or deployment.

Together, these stages form a coherent pipeline: starting from raw, irregular operational data and time‑to‑event labels, the system constructs labelled temporal windows, normalises them in a statistically sound way, and trains an LSTM to classify the proximity to failure. This baseline establishes a clear reference architecture on which more advanced models (e.g., attention mechanisms, time‑gap features, or specification embeddings) can be layered in subsequent work.

