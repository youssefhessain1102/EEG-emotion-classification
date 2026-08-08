# EEG Emotional Memory Classification — TMR Sleep Decoding

> **Targeted Memory Reactivation (TMR) Challenge**  
> A machine learning pipeline designed to decode emotional vs. neutral memory reactivation from NREM sleep EEG signals.

---

## 📌 Problem Overview

Given 1-second EEG epochs (**16 channels, 200 Hz, 200 timepoints**) locked to an auditory memory-reactivation cue during sleep, predict—at **every timepoint**—the probability that the reactivated memory was emotional rather than neutral.

* **Zero-Shot Generalization:** The classifier must generalize to unseen test participants without fine-tuning.
* **Scoring Metric:** A custom **Windowed-AUC** metric that identifies the longest continuous run of above-chance AUC ($\ge 50\text{ ms}$) averaged across test participants. This rewards sustained, localized neural signatures over noisy spikes.

---

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    A[Raw EEG Signal] --> B[Bandpass Filter: Theta 4-8 Hz & Alpha 8-13 Hz]
    B --> C[Per-Participant Robust Scaling]
    C --> D[Spatial Covariance Matrix 350-650ms Window]
    C --> E[Hemispheric Asymmetry Features: DASM]
    D --> F[Combined Feature Set 136 Covariance + 14 Asymmetry]
    E --> F
    F --> G[LOPO Stability-Based Feature Selection]
    G --> H[Shrinkage LDA Classifier]
    H --> I[Timepoint-Tiled Probability Predictions]
