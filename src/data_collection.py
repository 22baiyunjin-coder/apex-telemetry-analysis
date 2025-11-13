import pandas as pd
from pathlib import Path


def load_raw_telemetry(filepath: str) -> pd.DataFrame:
    """
    Load a raw Apex Legends telemetry JSON file exported from the API.

    Parameters
    ----------
    filepath : str
        Path to a .json file containing telemetry events.

    Returns
    -------
    pd.DataFrame
        Raw telemetry events as a DataFrame.
    """
    path = Path(filepath)

    if path.suffix != ".json":
        raise ValueError("Expected a .json file, got: {}".format(path.suffix))

    df = pd.read_json(path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning steps shared by all analyses.

    - Drop completely empty rows
    - Reset the index

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Example usage (you can change this later when you have real data)
    raw_path = "data/raw/sample_telemetry.json"

    try:
        df_raw = load_raw_telemetry(raw_path)
        df_clean = basic_cleaning(df_raw)
        df_clean.to_csv("data/processed/sample_telemetry_clean.csv", index=False)
        print("Saved cleaned telemetry to data/processed/")
    except FileNotFoundError:
        print("Example file not found. Make sure data/raw/sample_telemetry.json exists.")
