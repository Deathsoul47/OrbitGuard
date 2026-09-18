# OrbitGuard 🛰️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview
**OrbitGuard** is an advanced Anti-Gravity Space Debris Trajectory Prediction & Collision Avoidance System. With Low-Earth Orbit (LEO) becoming increasingly congested, traditional physics-based models struggle to maintain accuracy due to unpredictable atmospheric drag and micro-gravitational anomalies. 

OrbitGuard employs an LSTM-based Deep Learning pipeline to ingest raw Two-Line Element (TLE) tracking data, learn temporal orbital anomalies, and forecast future spatial coordinates ($x, y, z$) to predict and prevent catastrophic collisions.

## Key Features
* **Physics-Informed Ingestion (`ingest.py`):** Utilizes the `sgp4` physical propagator to parse TLE telemetry into true Earth-Centered Inertial (ECI) coordinates.
* **Leak-Proof Time-Series Processing (`preprocess.py`):** Group-aware imputation, Z-score normalization, and temporal sliding-window generation.
* **Deep Sequence Forecasting (`model.py` & `train.py`):** An LSTM regression network tailored for continuous trajectory mapping, complete with gradient clipping, dynamic learning rate scheduling, and early stopping.
* **Vectorized Risk Engine (`evaluate.py`):** Highly scalable `scipy` Euclidean distance matrix calculation capable of identifying collision threshold breaches across thousands of debris objects instantaneously.

## Technology Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Orbital Mechanics:** SGP4
* **Testing:** Pytest

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/OrbitGuard.git
   cd OrbitGuard
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Testing Guidelines
OrbitGuard includes a comprehensive `pytest` suite ensuring robust tensor dimensionality and leak-free preprocessing.

To execute the test suite:
```bash
pytest tests/test_pipeline.py -v
```

**Expected output:**
```text
tests/test_pipeline.py::TestPreprocess::test_impute_missing_values PASSED
tests/test_pipeline.py::TestPreprocess::test_normalize_features PASSED
tests/test_pipeline.py::TestPreprocess::test_create_sequences PASSED
tests/test_pipeline.py::TestModel::test_orbital_lstm_tensor_shapes PASSED
```
