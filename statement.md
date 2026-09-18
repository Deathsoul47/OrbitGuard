# Project Statement: OrbitGuard

## Problem Statement
The operational environment in Low-Earth Orbit (LEO) is experiencing exponential congestion. Satellites and millions of pieces of space debris travel at lethal velocities, where even millimeter-sized fragments can cause catastrophic damage. While current orbital propagators (like SGP4) effectively model standard Keplerian motion, they suffer from degradation over time due to micro-gravitational anomalies, solar radiation pressure, and variable atmospheric drag. 

There is a critical need for an intelligent predictive system that can learn from historical tracking data to forecast trajectories with higher long-term accuracy, ultimately preventing orbital collisions.

## Scope of the Project
**In-Scope:**
* Parsing and ingesting raw Two-Line Element (TLE) satellite telemetry data.
* Structuring historical trajectory data into continuous, gap-free time-series sequences.
* Training an LSTM (Long Short-Term Memory) neural network to predict future $(x, y, z)$ orbital coordinates.
* Evaluating model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* Calculating pairwise Euclidean distances across predicted trajectories to flag immediate collision risks based on a proximity threshold.

**Out-of-Scope:**
* Physical execution of automated avoidance maneuvers (station-keeping).
* Real-time hardware integration with ground-based radar arrays.
* Visual 3D simulation rendering (beyond basic mathematical dashboard analytics).

## Target Users
* **Space Traffic Management (STM) Operators:** Organizations responsible for tracking space debris and maintaining secure operational orbits.
* **Satellite Constellation Operators:** Companies (e.g., Starlink, Planet Labs, Spire) managing large fleets of satellites requiring automated collision alerts.
* **Space Agencies:** Government entities (NASA, ESA, ISRO) conducting orbital risk assessments.

## High-Level Features
1. **Automated TLE Ingestion:** Robust API for transforming standard textual tracking formats into analytical DataFrames using physical orbital math.
2. **Deep Trajectory Forecasting:** An advanced Recurrent Neural Network (RNN/LSTM) pipeline designed specifically for spatial time-series coordinate prediction.
3. **Scalable Risk Engine:** Vectorized mathematical matrices capable of checking collision proximities simultaneously across thousands of bodies.
4. **Data Reliability Engine:** Built-in safeguards that interpolate missing radar pings while preventing data-leakage across different satellite tracks.
