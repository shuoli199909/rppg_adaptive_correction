# Robust rPPG Pipeline with Motion-Tolerant Correction (AMTC) and RF-based Gating (HMA_RF)

> A concise, high-level description of the project. This repository implements a robust remote photoplethysmography (rPPG) pipeline that converts videos into heart-rate (HR) trajectories and includes motion-aware post-processing designed for challenging, real-world settings. Two new components—**AMTC** and **HMA_RF**—are integrated alongside the baseline **FactorizePhys** implementation to provide strong performance under motion and illumination variability.

---

## Key Ideas

- **End-to-end rPPG flow**: video → face/ROI RGB → BVP windowing → HR estimation → post-processing → evaluation.
- **Motion-tolerant post-processing**: mitigate motion artifacts and temporal spikes without over-smoothing physiologic dynamics.
- **Modular design**: plug-and-play filters and gates under `filters/`, interoperable with the existing pipeline and with FactorizePhys as a baseline reference.
- **Practical robustness**: emphasize stability in medium/high-motion segments, illumination changes, and imperfect face tracking.

---

## What's New

- **AMTC (`filters/AMTC.py`)**  
  Adaptive Motion‑Tolerant Correction that adjusts smoothing strength and state-space parameters based on displacement statistics and band-limited motion energy. The goal is to suppress motion‑induced HR spikes while preserving valid cardiac variability.

- **HMA_RF (`filters/HMA_RF.py`)**  
  A Random‑Forest–based (HMRF‑ready) confidence gate that classifies each window as **trustworthy** or **untrustworthy** using features such as displacement statistics, band energy, and BVP SNR. Untrustworthy windows are corrected via nearest‑valid temporal imputation, reducing large transient errors.

- **FactorizePhys (`FactorizePhys/`)**  
  A reproducible baseline implementation added for apples‑to‑apples comparison with the same data flow and evaluation metrics.

---

## Repository Structure

```
.
├── arduino/                    # Hardware-side sketches (optional benchmarks)
├── config/                     # Config files
├── data/                       # Raw and intermediate data (ignored in VCS)
├── filters/                    # Filters and artifact-handling modules
│   ├── AMTC.py                 # Adaptive Motion-Tolerant Correction
│   ├── HMA_RF.py               # RF/HMRF confidence gating + nearest-neighbor fill
│   ├── index_filter.py
│   ├── kalman_filter.py
│   ├── moving_average.py
│   ├── outlier_detection.py
│   └── peak_verification.py
├── FactorizePhys/              # Baseline method for comparison
├── main/                       # Entry points for each pipeline stage
│   ├── main_vid2rgb.py         # Video → ROI RGB
│   ├── main_rgb2bvpwin.py      # RGB → BVP windows
│   ├── main_bvpwin2HR.py       # BVP windows → HR + post-processing
│   └── main_gen_gtHR.py        # Ground-truth HR preparation
├── processor/                  # Post-processing orchestration & indices
│   ├── index_processor.py
│   └── post_processor.py
├── result/                     # Outputs & evaluation artifacts
├── util/                       # Utilities
├── LICENSE
└── README.md
```

---

## Goals & Positioning

- **Scientific**: Provide a clean reference for studying how motion-aware post‑processing improves rPPG accuracy beyond algorithmic baselines.
- **Engineering**: Offer a realistic, modular pipeline that can be adapted to embedded or mobile use cases.
- **Reproducibility**: Keep components decoupled, with clear I/O conventions across pipeline stages, so experiments are easy to replicate and extend.

---

## Evaluation Summary (Recommended)

Report at least:
- HR errors: MAE, median AE, RMSE, DTW.
- Robustness: proportion of corrupted windows, precision/recall/F1 of HMA_RF gating.
- Stability deltas: improvements before vs. after AMTC/HMA_RF (ΔMAE, ΔRMSE, ΔDTW).
- Scenario breakdowns: low vs. medium/high motion; static vs. dynamic lighting.

---

## Ethics & Responsible Use

rPPG signals are biometric/health‑adjacent. When using this code or derived models:
- Obtain informed consent and follow local regulations.
- Avoid re‑identification and secondary use beyond the original consent.
- Safeguard data and models to prevent misuse.

---

## License

This project is released under the MIT License (see `LICENSE`).

---

## Acknowledgments

We thank the open‑source rPPG community and prior methods for inspiration and baselines. FactorizePhys is included to support transparent, controlled comparisons.
