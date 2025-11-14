# Apex Telemetry Analysis

End-to-end gameplay analytics project using **Apex Legends** telemetry data:  
data collection, feature engineering, clustering, survival modeling, and visualization of player behavior.

---

## 🎯 Project Goals

- Use Apex Legends telemetry logs (movement, combat, weapon, survival events) to understand player behavior.
- Engineer meaningful features such as movement speed, engagement rate, weapon switch frequency, and squad spacing.
- Predict **survival time** using gradient-boosted tree models (XGBoost).
- Visualize hotspots, correlations, and feature importance to support game design and balance decisions.

---

## 📂 Project Structure

```text
apex-telemetry-analysis/
│
├── src/
│   ├── data_collection.py      # Load & clean raw telemetry JSON
│   ├── feature_engineering.py  # Create gameplay features from events
│   ├── modeling.py             # Train survival-time prediction model
│   └── visualization.py        # Plots: distributions, correlations, feature importance
│
├── data/
│   ├── raw/        # Raw telemetry files (not tracked in git)
│   └── processed/  # Cleaned & feature-engineered datasets
│
├── notebooks/
│   └── 01_data_collection.ipynb   # (optional) interactive exploration
│
├── reports/
│   └── figures/   # Saved plots and figures
│
├── README.md
├── LICENSE
└── .gitignore
