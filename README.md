# EEG Emotional Memory Classification — TMR Sleep Decoding

A pipeline for decoding emotional vs. neutral memory reactivation from EEG signals recorded during NREM sleep, built for the *EEG Emotional Memory Classification Challenge* (Targeted Memory Reactivation / TMR).

## Problem

Given 1-second EEG epochs (16 channels, 200 Hz, 200 timepoints) locked to an auditory memory-reactivation cue during sleep, predict — at **every timepoint** — the probability that the reactivated memory was emotional rather than neutral. The classifier must generalize to **unseen participants** (zero-shot, no fine-tuning on test subjects).

Scoring is a custom windowed-AUC metric: it finds the longest continuous run of above-chance AUC (≥ 50 ms) and averages that across test participants — not a flat mean AUC across all timepoints. This rewards sustained, localized effects over noisy spikes.

## Approach

**Core features — theta-band covariance (`preprocessor.py`)**
1. Butterworth bandpass filter, theta band (4–8 Hz), zero-phase (SOS + `sosfiltfilt`), applied to the whole `(trials, channels, time)` array at once.
2. Per-participant `RobustScaler` normalization across the channel×time dimensions (fit only on that participant's own data — no cross-participant leakage).
3. Spatial covariance matrix computed over a fixed window (samples 70–130, roughly 350–650 ms post-cue) across all 16 channels.
4. Upper-triangular elements of the covariance matrix extracted as features (136 values from the 16×16 matrix).

**Hemispheric asymmetry features**
Added on top of the 136 covariance features, based on 7 homologous channel pairs (C3/C4, F3/F4, CP3/CP4, CP5/CP6, C5/C6, P7/P8, O1/O2 — Cz and Pz excluded as midline channels with no pair). Two design decisions were tested empirically before settling on the current setup:

- **Computed from raw power, before RobustScaler** — not from the post-scaling covariance diagonal. `RobustScaler`'s shrinkage pushed log-power values close to (and across) zero more often, destabilizing the ratio-based asymmetry feature. Computing from raw per-channel variance in the same window avoids this, and for the difference-based feature this is also the more principled choice: any shared participant-level gain factor cancels out in a log-difference regardless of prior scaling, so scaling before the comparison was unnecessary.
- **DASM (log-power difference) over RASM (log-power ratio)** — DASM is numerically well-behaved by construction (a subtraction never blows up). RASM (a ratio of two log values, each of which can be negative or near zero) needed an epsilon-guarded denominator to avoid division-by-near-zero blow-ups, and empirically contributed less reliably than DASM across LOPO folds.
- **Second band (alpha, 8–13 Hz) for asymmetry only** — theta is the locked band for the main 136 covariance features because that's where the classification signal was validated to live, but hemispheric asymmetry effects in the emotion literature are best established in alpha (and gamma), not theta. Rather than compromise the validated theta features, a second alpha-only bandpass pass is run just to compute asymmetry features, keeping the two roles (classification signal vs. asymmetry signal) in their respective, literature-supported bands.

All asymmetry features are z-scored together with the 136 covariance features in one pass per participant (column-wise: each feature gets its own mean/std across that participant's trials, independent of every other feature — so grouping order before z-scoring has no effect on the resulting values).

**Feature selection** (`preprocessor.select_top_features`)
A first LOPO (leave-one-participant-out) pass fits an LDA per fold and collects the coefficient vectors across the full feature set (covariance + asymmetry). Features are ranked by a stability score combining sign agreement across folds, mean absolute coefficient magnitude, and coefficient variability across folds (penalized). The top-K most stable features are kept for the final model — this step is what lets the asymmetry features earn their place (or get dropped) based on evidence rather than assumption.

**Classification** (`model.py`)
Shrinkage LDA (`solver="lsqr"`, `shrinkage="auto"`), trained on the selected features. A second LOPO pass evaluates generalization AUC per participant using only the selected features. The final submission model is retrained on **all** training participants with the same feature set.

**Inference** (`predict.py`)
Test-participant trials go through the identical preprocessing pipeline (same theta covariance, same raw-power asymmetry computation, same feature selection mask), then the final model predicts one probability per trial, tiled across all 200 timepoints, and formatted into the competition's `{subject}_{trial}_{timepoint}` submission format.

## Results

| Configuration | Leaderboard AUC |
|---|---|
| + asymmetry (raw power, pre-scaling), RobustScaler | 0.551 |
| + asymmetry (post-scaling covariance diagonal), RobustScaler | 0.545 |
| + asymmetry (post-scaling covariance diagonal), z-score | 0.542 |
| Baseline (starter notebook, Hilbert power + plain LDA) | ~0.516 |

The raw-power asymmetry computation closed most of the gap versus the post-scaling version, confirming the hypothesis that `RobustScaler`'s shrinkage was destabilizing the log-power asymmetry features rather than the underlying idea being unhelpful. DASM-only and alpha-band asymmetry variants are still being evaluated via LOPO before further leaderboard submissions.

## Project structure

```
.
├── main.py              # Entry point: runs training then inference
├── train.py              # Data loading, LOPO feature selection + validation, final model fit
├── predict.py            # Test-set preprocessing, prediction, submission.csv generation
├── data_loader.py         # HDF5 (.mat v7.3) loading utilities
├── preprocessor.py        # Bandpass filter, scaling, covariance + asymmetry features, feature selection
├── model.py               # LDA model factory + LOSO evaluation
└── config/
    └── config.yaml        # All paths, signal-processing, and model hyperparameters
```

## Running it

```bash
python main.py
```

Reads `config/config.yaml`, expects training data under `paths.train_path` (with `sleep_neu` / `sleep_emo` subfolders, one `.mat` file per participant) and test data under `paths.test_path` (`test_subject_*.mat` files). Produces `submission.csv` in the format required by the competition.

## Key design decisions worth knowing

- **Per-participant scaling everywhere**: `RobustScaler` and the final z-score are fit independently on each participant's own trials. Cross-participant EEG variability is the central challenge of this task — mixing scaling statistics across participants would leak information and hurt generalization.
- **Covariance window (70:130)**: restricts the covariance computation to a mid-epoch window rather than the full 200 samples, reducing edge artifacts and focusing on the period most discriminative for theta. This window has not yet been separately validated for the alpha-band asymmetry features — it's currently reused as a starting assumption, not a confirmed choice for that band.
- **Coefficient-stability feature selection**: rewards features that are both large in magnitude *and* consistent in direction across LOPO folds — a heuristic for picking features likely to generalize to new participants rather than overfitting to training-set idiosyncrasies. This is also the mechanism relied on to judge whether new feature additions (like asymmetry) are actually earning their place, rather than deciding by assumption.
- **Vectorized filtering and dtype management**: bandpass filtering is applied to the full `(trials, channels, time)` array in one `sosfiltfilt` call (via its `axis` parameter) rather than looping per trial/channel in Python, and arrays are kept in `float32`. Both changes were made to manage memory and runtime in a RAM-constrained environment (Google Colab) once a second (alpha) filtering pass was added for asymmetry features.

## Credits

The the fixed analysis window `[70:130]`, and the coefficient-stability feature-selection method were adapted from **Eng. Mohamed Samy**'s pipeline for this same competition (2nd place), shared here:
[MohamedQiqa/eeg-emotion-classification-pipeline](https://github.com/MohamedQiqa/eeg-emotion-classification-pipeline/blob/master/_research_architecture/01_pipeline_blueprint.md)

The hemispheric asymmetry features (DASM/RASM, raw-power computation, alpha-band extension) are an independent addition on top of that base, motivated by the frontal-asymmetry literature in EEG emotion research. All other pipeline engineering — data loading, LOPO validation structure, per-participant scaling strategy, memory optimization, training/inference orchestration, and configuration — was implemented independently. Code was refactored for clarity with the help of an LLM assistant, but every part of the pipeline is understood and was validated by the author.

## Competition

Based on the *EEG Emotional Memory Classification Challenge* — decoding memory reactivation during NREM sleep via Targeted Memory Reactivation (TMR). See the competition's starter notebook and dataset description for full experimental background.#   E E G - e m o t i o n - c l a s s i f i c a t i o n 
 
 