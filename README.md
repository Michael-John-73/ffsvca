# FFSVCA — Fixed-FPR Security Verification under Composite Attacks

**PAPER3**: Fixed-FPR Security Verification of ROBIN-Style Diffusion Watermark Detectors under Composite Attacks

---

## Overview

This repository implements a **fixed-FPR security verification protocol** for ROBIN-style inversion-based diffusion watermark detectors. The protocol evaluates whether clean-calibrated thresholds hold their false-positive guarantees under ID-single, ID-composite, and OOD-composite attack conditions.

This paper does **not** propose a new watermark embedding algorithm or detector architecture. The ROBIN substrate is used as-is. The contribution is the **verification-layer protocol** and **failure-condition analysis**.

---

## Repository Structure

```
FFSVCA/
├── ffsvca_bridge.py          # ROBIN score dump wrapper (attack loop + CSV export)
├── ffsvca_verifier.py        # Fixed-FPR threshold computation (M1/M2/M3)
├── ffsvca_runner.py          # E1~E4 analysis pipeline runner
├── ffsvca_result_metrics.py  # FPR/TPR/CE/CI/Failure-condition metrics
├── gen_figures.py            # Figure 1-3 generation
├── gen_flow.py               # Pipeline diagram
├── robin_config.json         # ROBIN parameter mapping (edit paths before use)
├── requirements.txt          # Python dependencies
└── docs/                     # GitHub Pages
    └── assets/
```

---

## Protocol Summary

| Step | Description |
|------|-------------|
| 1 | Security requirement: α = 0.05 (primary), 0.10 / 0.01 (secondary) |
| 2 | Score: z(x) = −d(x), where d(x) is ROBIN inversion distance |
| 3 | Threshold: τ_α = Q_{1−α}(Z_{0,cal}) from clean calibration set only |
| 4 | Decision: φ_α(x) = 1[z(x) > τ_α] |
| 5 | Evaluate empirical FPR, TPR@FPR≤α, CE, violation per attack task |
| 6 | Report operating range and failure conditions by attack group |

---

## Experiments

| Experiment | Research Question | Key Output |
|---|---|---|
| E1 | Does fixed-FPR protocol expose FPR risk hidden by M1/M2? | Table 2 |
| E2 | Where does FPR control hold/fail across attack groups? | Table 3 |
| E3 | What score-distribution shift causes FPR violation? | Figure 2 |
| E4 | How many calibration samples are needed for stability? | Figure 3, Table 4 |

---

## Attack Protocol (11 tasks)

| Group | Attack |
|---|---|
| ID-single | none, jpeg (q=50), cropping, rotation, blurring, noise |
| ID-composite | jpeg+cropping, cropping+jpeg, blurring+noise |
| OOD-composite | rotation+cropping+jpeg, noise+blurring+jpeg |

---

## Setup

```bash
pip install -r requirements.txt
# Edit robin_config.json: set robin_root and wm_path to your local paths
```

**Required checkpoint**: `robin_official/ckpts/optimized_r5_15_step10.pt`  
(w_low_radius=5, w_up_radius=15, w_channel=3, w_pattern=ring)

---

## Reproducibility

- gen_seed: 0 (fixed)
- split_seed: 42 (fixed)
- All severity values fixed before experiments (see robin_config.json)
- Full code, config, and checkpoint path documented for replication

---

## Citation

> [Paper citation to be added upon acceptance]
