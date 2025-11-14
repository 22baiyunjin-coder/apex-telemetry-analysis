from pathlib import Path

import pandas as pd

from .data_collection import load_raw_telemetry, basic_cleaning
from .feature_engineering import engineer_features, select_model_features
from .modeling import train_survival_model


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def run_pipeline(raw_filename: str = "sample_telemetry.json"):
    """
    End-to-end pipeline:

    1. Load raw telemetry JSON
    2. Basic cleaning
    3. Feature engineering
    4. Select modeling features + target
    5. Train survival model and print metrics
    """

    raw_path = RAW_DIR / raw_filename

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw telemetry file not found at {raw_path}. "
            f"Place a JSON file under data/raw/ first."
        )

    # 1. load & clean
    df_raw = load_raw_telemetry(str(raw_path))
    df_clean = basic_cleaning(df_raw)

    # 2. feature engineering
    df_feat = engineer_features(df_clean)

    # 3. select features for modeling
    feature_df = select_model_features(df_feat)

    # 这里假设原始数据里已经有 survival_time 这一列
    if "survival_time" not in df_feat.columns:
        raise KeyError(
            "Column 'survival_time' not found in data. "
            "Make sure your processed data contains this target."
        )

    model_input = feature_df.copy()
    model_input["survival_time"] = df_feat["survival_time"]

    # 4. train model
    model, metrics = train_survival_model(model_input, target_col="survival_time")

    # 5. save processed data (可选)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "features_for_model.csv"
    model_input.to_csv(out_path, index=False)

    print("=== Training finished ===")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R^2 : {metrics['r2']:.4f}")
    print(f"Saved processed features to: {out_path}")

    return model, metrics


if __name__ == "__main__":
    # Example: run the full pipeline on a sample file
    run_pipeline("sample_telemetry.json")
