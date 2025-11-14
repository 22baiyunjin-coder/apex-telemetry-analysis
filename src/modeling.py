import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np


def train_survival_model(df: pd.DataFrame, target_col: str = "survival_time"):
    """
    Train a gradient-boosted regression model (XGBoost) to predict survival time.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with target column.
    target_col : str
        Name of the survival time column.

    Returns
    -------
    model : XGBRegressor
    metrics : dict
    """

    df = df.copy()
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": rmse,
        "r2": r2,
    }

    return model, metrics


if __name__ == "__main__":
    print("Modeling module ready.")
