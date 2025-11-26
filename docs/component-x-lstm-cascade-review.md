# Component X LSTM – Cascade Plan Review

This document reviews the two-stage (cascade) proposal from `docs/component-x-lstm-cascade.md`, highlighting strengths, open questions, and suggested clarifications before implementation.

## 1. Summary of the Proposed Cascade

1. **Model A – Binary Detector**
   - Input: current LSTM-ready sequences.
   - Output: probability of “fault within 48 steps” (classes 1–4 collectively).
   - Goal: very high recall (catch every potential fault), even at the expense of false positives.
2. **Model B – Fault Diagnoser**
   - Input: *only the sequences flagged as positive by Model A*.
   - Output: multi-class probability over {1,2,3,4}.
   - Goal: classify specific proximity class under a balanced training distribution (class 0 removed).

## 2. Strengths

| Strength | Details |
| --- | --- |
| **Separated objectives** | Model A optimizes sensitivity without being penalized for confusing minority classes. Model B operates in a balanced regime, avoiding extreme loss weighting. |
| **Simpler tuning** | A single decision threshold on Model A controls alert rate, while Model B can use standard cross-entropy (with or without mild class weights). |
| **Operational diagnostics** | Easier post-mortem: if a failure was missed, inspect Model A; if detected but misclassified, inspect Model B. |

## 3. Questions & Clarifications Needed

| Topic | Open Question | Suggested Clarification |
| --- | --- | --- |
| **Training data for Model A** | Will Model A use the same `train_sequences.npz` distribution (72/7/7/6/6)? | Confirm yes, but consider oversampling classes 1–4 just for Model A if recall targets >99% are required. |
| **Model B training set** | Plan states “remove class 0,” but how are sequences selected? Are we reusing the same event windows or generating new ones? | Document whether Model B reuses the exact sequences from Model A’s training set (filtered) or rebuilds new sequences centered closer to failures. |
| **Inference hand-off** | How are probabilities combined? Do we multiply `P(fault)` from Model A with `P(class/fault)` from Model B, or do we output Model B’s class only when Model A exceeds a threshold? | Clarify inference pipeline, including thresholds (e.g., `if P_A >= 0.3: class = argmax(Model B); else class = 0`). |
| **Evaluation metrics** | Cascade introduces pipeline latency. Are we tracking metrics separately (Model A recall, Model B macro-F1) and combined (overall confusion matrix)? | Define evaluation procedure: e.g., run validation through A → B, produce full 5-class confusion matrix plus A’s ROC/PR curves. |
| **Serving cost** | Two forward passes per sample—acceptable? Need batching strategy? | Add a note on expected throughput and whether Model B is only invoked for ~5% of samples (true positives + false alarms). |

## 4. Suggested Adjustments to the Plan

1. **Explicit Data Splits**
   - Document the exact NPZ creation for both models:
     - `train_sequences_binary.npz` (classes {0, fault}) – optional downsampling/oversampling plan.
     - `train_sequences_diag.npz` (only classes 1–4) – specify whether sequences are the same windows or re-windowed closer to failure.

2. **Threshold Strategy**
   - Define how the detection threshold is chosen. Recommendation: optimize Model A on validation using PR curves, pick threshold for ≥99% recall, then evaluate cascade.

3. **Loss Choices**
   - Model A: consider focal loss or class-weighting for the binary task to keep recall high without saturating.
   - Model B: mention that standard cross-entropy + mild label smoothing might suffice once class 0 is removed.

4. **Logging & Monitoring**
   - Extend the plan to log MLflow runs for both models separately, plus a combined evaluation run that stores:
     - Model A metrics (ROC, PR).
     - Model B metrics (macro-F1 under balanced validation).
     - Cascade confusion matrix (after applying threshold).

5. **Fallback Path**
   - Describe a fallback when Model A is uncertain (e.g., `[0.2, 0.8]` scores). Options:
     - Always run Model B and combine scores.
     - Only run Model B above a lower threshold but still propagate Model A probability downstream (useful for cost-based decisions).

Documenting these clarifications will make it easier to implement and compare cascade experiments with the current single-model baseline. Once the plan is finalized, we can split the codebase into `src/training/train_detector.py` and `train_classifier.py` (or share modules with config flags) and wire both into MLflow + evaluation scripts.
