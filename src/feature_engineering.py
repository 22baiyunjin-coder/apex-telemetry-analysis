import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core feature engineering for Apex Legends telemetry data.

    Features include:
    - movement_speed: distance traveled / time
    - engagement_rate: number of combat events per minute
    - weapon_switch_freq: number of weapon swap events per minute
    - damage_efficiency: total damage / shots fired
    - teammate_distance: average distance to squadmates
    """

    df = df.copy()

    # movement speed
    if {"distance", "time"}.issubset(df.columns):
        df["movement_speed"] = df["distance"] / df["time"].replace(0, np.nan)

    # engagement rate
    if "combat_events" in df.columns and "time" in df.columns:
        df["engagement_rate"] = df["combat_events"] / df["time"].replace(0, np.nan)

    # weapon switch frequency
    if "weapon_swaps" in df.columns and "time" in df.columns:
        df["weapon_switch_freq"] = df["weapon_swaps"] / df["time"].replace(0, np.nan)

    # damage efficiency
    if {"damage", "shots_fired"}.issubset(df.columns):
        df["damage_efficiency"] = df["damage"] / df["shots_fired"].replace(0, np.nan)

    # teammate distance
    if {"teammate_distances"}.issubset(df.columns):
        df["avg_teammate_distance"] = df["teammate_distances"].apply(
            lambda x: np.mean(x) if isinstance(x, list) else np.nan
        )

    df = df.dropna(axis=1, how="all")  # remove empty columns

    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only useful features for modeling.
    """
    target_columns = [
        "movement_speed",
        "engagement_rate",
        "weapon_switch_freq",
        "damage_efficiency",
        "avg_teammate_distance",
    ]

    return df[[col for col in target_columns if col in df.columns]]


if __name__ == "__main__":
    print("Feature engineering module loaded.")
