# Green-Skills-AI-Internship-Project

#  Path-Dependent Energy Stress Modeling for Lithium-Ion Battery Packs

An **end-to-end, research-grade AI project** that models **battery degradation as a path-dependent functional problem**, moving beyond traditional cycle-counting approaches.

This repository integrates **data science, machine learning, functional deep learning (DeepONet), and AI-driven analytics** to quantify battery stress based on **current waveform shape**, not just energy usage.

---

##  Project Overview

Traditional battery aging models assume degradation depends on *how much* energy flows through a battery. In reality, degradation depends on *how* energy flows **over time**.

>  Same Ah ≠ Same Damage

High-frequency current fluctuations, spikes, and pulsed loads degrade batteries faster—even when total energy is identical.

This project introduces a **Stress Functional Learning Framework** that:

* Learns degradation directly from **I(t)** and **dI/dt**
* Identifies **toxic load shapes**
* Predicts **capacity fade and stress per cycle**
* Creates a **Safe vs Dangerous operating envelope**

---

##  Objectives

* Replace **cycle counting** with **stress-based degradation modeling**
* Learn degradation physics directly from data
* Capture **path dependence** in battery aging
* Combine **ML + Deep Learning + Physics constraints**
* Provide **interpretable insights** for Battery Management Systems (BMS)

---

##  Core Concept

Battery degradation is modeled as a **functional**:

```
D[I(t)] = ∫ f(I(t), dI/dt) dt
```

Where:

* `I(t)` = current profile
* `dI/dt` = current derivative (stress amplifier)
* `D` = learned degradation / stress

This replaces heuristic models with **data-driven physics learning**.

---

##  Repository Structure

```
batteryguard-ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── preprocessing/
│   ├── signal_cleaning.py
│   ├── cycle_extraction.py
│
├── features/
│   ├── load_shape_features.py   # C1–C4
│
├── models/
│   ├── stress_deeponet.py
│   ├── degradation_ml.py
│   ├── anomaly_models.py
│   ├── rul_models.py
│
├── xai/
│   ├── shap_analysis.py
│   ├── stress_attribution.py
│
├── safety/
│   ├── early_warning.py
│   ├── risk_scoring.py
│
├── analytics/
│   ├── cell_pack_loss.py
│   ├── reporting.py
│
├── chatbot/
│   ├── assistant.py
│
├── main.py
├── config.yaml
└── README.md  

```

---

##  Dataset Description

### Real Data (EV Lab)

Collected from battery test bench:

* Current (A)
* Voltage (V)
* Temperature (°C)
* Capacity (Ah)
* Time (timestamps)
* Cycle number
* Charge/Discharge status

### Synthetic Data (Fallback)

Used when real data is incomplete:

* Sinusoidal loads
* Pulse currents
* High-frequency noisy loads
* Realistic fluctuating EV patterns

---

##  Data Preprocessing

* Missing value handling
* Time-series resampling
* Cycle segmentation
* Numerical differentiation (`dI/dt`)
* Normalization & scaling
* Capacity drop calculation

---

##  Feature Engineering (Load Shape Toxicity)

Each cycle is summarized using **four physically meaningful coefficients**:

| Feature | Description | Interpretation         |   |                            |
| ------- | ----------- | ---------------------- | - | -------------------------- |
| C1      | mean(       | I                      | ) | Average stress level       |
| C2      | mean(       | dI/dt                  | ) | Spike / fluctuation damage |
| C3      | std(I)      | Oscillation amplitude  |   |                            |
| C4      | sum(I²)/N   | Energy-weighted stress |   |                            |

These features feed ML models and support interpretability.

---

##  Exploratory Data Analysis (EDA)

Performed analyses include:

* Correlation heatmaps
* Stress vs capacity plots
* Cycle-wise degradation trends
* Cluster visualization

### Key Insight:

> `dI/dt` correlates **more strongly** with degradation than energy throughput alone.

---

##  Machine Learning Models

### Regression

Used to predict capacity loss (`ΔCapacity`):

* Linear Regression
* Random Forest Regressor
* Support Vector Regressor (SVR)

### Unsupervised Learning

* KMeans clustering
* Pattern discovery in load shapes

---

##  Deep Learning: Neural Operator (DeepONet)

### Why DeepONet?

DeepONet is designed for:

> **Function → Scalar mappings**

Exactly matching our problem:

```
I(t) → Stress / Degradation
```

### Architecture

* **Branch Network**: Learns current & derivative behavior
* **Trunk Network**: Learns time dependence
* **Elementwise product** + summation

### Physics-Informed Loss

```
Loss = MSE + λ · mean((dI/dt)²)
```

Ensures:

* Smoothness
* Physical plausibility
* Stability

---

##  Stress & Safety Modeling

For each cycle:

* Stress value is predicted
* Equivalent Stress Cycle (ESC) is computed
* Cycles are classified as:

  *  Safe
  *  Dangerous

Thresholding uses percentile-based stress limits.

---

##  AI Chatbot Interface

A lightweight analytics chatbot enables natural queries:

Examples:

* `What is mean stress?`
* `Which cycles are dangerous?`
* `Show stress plot`
* `Show cluster-wise statistics`

Demonstrates AI-powered battery analytics.

---

##  Tech Stack

### Programming & Data

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* SVR
* KMeans

### Deep Learning

* PyTorch
* Neural Operators (DeepONet)

### AI / NLP

* Rule-based chatbot
* GPT-style extensibility

---

##  Key Findings

1. Battery stress is **path-dependent**, not cycle-dependent
2. High `dI/dt` causes disproportionate damage
3. ESC is more accurate than raw cycle count
4. Neural Operators successfully learn degradation functionals
5. ML models identify toxic load shapes

---

##  Applications

* Electric Vehicle Battery Management Systems (BMS)
* Smart charging optimization
* Energy storage systems
* Predictive maintenance
* Warranty & lifecycle estimation
* Safety envelope monitoring

---

##  Final Statement

This project demonstrates a **next-generation AI framework** for lithium-ion battery degradation modeling by:

* Learning degradation physics from data
* Modeling stress as a functional
* Integrating ML, DL, and AI analytics

> **A practical step toward intelligent, safer, and longer-lasting battery systems.**

---

##  Future Work

* Multi-physics coupling (thermal + electrochemical)
* Transformer-based sequence models
* Online BMS deployment
* Real-time stress-aware charging control

---

 If you find this project useful, consider starring the repository.
