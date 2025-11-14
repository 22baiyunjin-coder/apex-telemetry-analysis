import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distributions(df: pd.DataFrame, cols=None):
    """
    Plot histograms for key numeric features.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    df[cols].hist(bins=30, figsize=(14, 8))
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plot a correlation heatmap for numeric features.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="viridis")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from a fitted tree-based model.
    """
    importances = model.feature_importances_

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("Model Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Visualization module ready.")
