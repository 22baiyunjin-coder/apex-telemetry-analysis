from pathlib import Path

import pandas as pd

from .data_collection import load_and_prepare_apex_history
from .feature_engineering import engineer_features, select_model_features
from .modeling import train_survival_model

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def run_pipeline(raw_filename: str = "Apex_Game_History_Season15S1.csv"):
    """
    End-to-end pipeline for the real Apex Season 15 game history dataset.

    1. Load and map raw game history to telemetry-like columns
    2. Feature engineering
    3. Select modeling features + target
    4. Train survival model and print metrics
    """

    # 1. 读取并转换这份真实数据
    df_base = load_and_prepare_apex_history(raw_filename)

    # 2. 特征工程
    df_feat = engineer_features(df_base)

    # 3. 选择特征 + 目标变量
    #    engineer_features 里会基于 combat_events/time 计算 engagement_rate 等特征
    feature_df = select_model_features(df_feat)

    if "survival_time" not in df_feat.columns:
        raise KeyError(
            "Column 'survival_time' not found in data. "
            "Make sure load_and_prepare_apex_history created it correctly."
        )

    model_input = feature_df.copy()
    model_input["survival_time"] = df_feat["survival_time"]

    # 4. 训练模型
    model, metrics = train_survival_model(model_input, target_col="survival_time")

    # 5. 可选：保存用于建模的数据
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "apex_season15_features_for_model.csv"
    model_input.to_csv(out_path, index=False)

    print("=== Training finished on Apex Season 15 dataset ===")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R^2 : {metrics['r2']:.4f}")
    print(f"Saved processed features to: {out_path}")

    return model, metrics

if __name__ == "__main__":
    # Example: run the full pipeline on a sample file
    run_pipeline("sample_telemetry.json")
