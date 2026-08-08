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

**Feature selection (`preprocessor.select_top_features`)**

A first LOPO (leave-one-participant-out) pass fits an LDA per fold and collects the coefficient vectors across the full feature set (covariance + asymmetry). Features are ranked by a stability score combining sign agreement across folds, mean absolute coefficient magnitude, and coefficient variability across folds (penalized). The top-K most stable features are kept for the final model — this step is what lets the asymmetry features earn their place (or get dropped) based on evidence rather than assumption.

**Classification (`model.py`)**

Shrinkage LDA (`solver="lsqr"`, `shrinkage="auto"`), trained on the selected features. A second LOPO pass evaluates generalization AUC per participant using only the selected features. The final submission model is retrained on **all** training participants with the same feature set.

**Inference (`predict.py`)**

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
