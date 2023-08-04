import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from cyano.data.utils import add_unique_identifier


def generate_and_plot_crosstab(y_true, y_pred, normalize=False):
    to_plot = pd.crosstab(y_pred, y_true)

    # make sure crosstab is 1-5 on both axes
    for i in np.arange(1, 6):
        if i not in to_plot.index:
            to_plot.loc[i, :] = 0

        if i not in to_plot.columns:
            to_plot[i] = 0

    # reverse index order for plotting
    to_plot = to_plot.loc[::-1, :].astype(int)
    fmt = ",.0f"

    if normalize:
        to_plot = to_plot / to_plot.sum()
        fmt = ".0%"

    fig, ax = plt.subplots()

    sns.heatmap(to_plot, cmap="Blues", annot=True, fmt=fmt, cbar=False, ax=ax)

    ax.set_xlabel("Actual severity")
    ax.set_ylabel("Predicted severity")
    return ax


class Evaluate:
    def __init__(self, y_true_csv, y_pred_csv, save_dir):
        y_true_df = pd.read_csv(y_true_csv)

        if "severity" not in y_true_df.columns:
            raise ValueError("Evaluation data must include a `severity` column to evaluate.")
        
        self.y_true = add_unique_identifier(y_true_df)["severity"]
        
        y_pred_df = pd.read_csv(y_pred_csv).set_index("uid")
        
        try:
            self.y_pred = y_pred_df.loc[self.y_true.index]["severity"]
        except:
            raise IndexError(
                "UIDs for points (lat, lon, date) in evaluation_csv do not align with UIDs in prediction_csv."
            )
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def calculate_metrics(self):
        results = dict()
        results["overall_rmse"] = mean_squared_error(self.y_true, self.y_pred, squared=False)
        results["overall_mae"] = mean_absolute_error(self.y_true, self.y_pred)
        results["overall_mape"] = mean_absolute_percentage_error(self.y_true, self.y_pred)

        results["regional_rmse"] = dict()
        results["regional_mae"] = dict()

        results["classification_report"] = classification_report(
                self.y_true, self.y_pred, labels=np.arange(1, 6), output_dict=True, zero_division=False
        )

        return results
    
    def calculate_all_and_save(self):
        results = self.calculate_metrics()
        with (self.save_dir / "results.json").open("w") as f:
            json.dump(results, f)

        crosstab_plot = generate_and_plot_crosstab(self.y_true, self.y_pred)
        crosstab_plot.figure.savefig(self.save_dir / "crosstab.png")


    # plot regional RMSE
    # plot precision and recall by severity level