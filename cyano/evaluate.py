import json
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
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

    _, ax = plt.subplots()

    sns.heatmap(to_plot, cmap="Blues", annot=True, fmt=fmt, cbar=False, ax=ax)

    ax.set_xlabel("Actual severity")
    ax.set_ylabel("Predicted severity")
    return ax


def generate_actual_density_boxplot(y_true_density, y_pred):
    df = pd.concat(
        [
            y_true_density,
            y_pred.loc[y_true_density.index],
        ],
        axis=1,
    )
    df.columns = ["density_cells_per_ml", "y_pred"]

    _, ax = plt.subplots()

    sns.boxplot(
        data=df,
        y="density_cells_per_ml",
        x="y_pred",
        ax=ax,
        order=list(range(1, 6)),
        showfliers=False,
    )
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("Actual density (cells/mL)")

    return ax


def generate_regional_barplot(regional_rmse):
    to_plot = pd.DataFrame({"regional_rmse": regional_rmse}).sort_values("regional_rmse")

    _, ax = plt.subplots()
    sns.barplot(to_plot.T, ax=ax)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSE")

    return ax


class EvaluatePreds:
    def __init__(self, y_true_csv: Path, y_pred_csv: Path, save_dir: Path, model: lgb.Booster):
        """Instantate EvaluatePreds class. To automatically generate all key visualizations, run
        cls.calculate_all_and_save() after instantiation.
        """
        self.model = model

        all_preds = pd.read_csv(y_pred_csv).set_index("sample_id")

        self.missing_predictions_mask = all_preds.severity.isna()
        self.y_pred_df = all_preds[~self.missing_predictions_mask].copy()
        self.y_pred_df["severity"] = self.y_pred_df.severity.astype(int)
        logger.info(f"Evaluating on {len(self.y_pred_df):,} samples (of {len(all_preds):,})")

        y_true_df = pd.read_csv(y_true_csv)

        if "severity" not in y_true_df.columns:
            raise ValueError("Evaluation data must include a `severity` column to evaluate.")

        y_true_df = add_unique_identifier(y_true_df)

        try:
            self.y_true_df = y_true_df.loc[self.y_pred_df.index]
        except KeyError:
            raise IndexError(
                "Sample IDs for points (lat, lon, date) in y_pred_csv do not align with sample IDs in y_true_csv."
            )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def calculate_severity_metrics(y_true, y_pred, region=None):
        results = dict()
        results["overall_rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        results["overall_mae"] = mean_absolute_error(y_true, y_pred)
        results["overall_mape"] = mean_absolute_percentage_error(y_true, y_pred)

        if region is not None:
            df = pd.concat([y_true, y_pred, region], axis=1)
            df.columns = ["y_true", "y_pred", "region"]
            results["regional_rmse"] = (
                df.groupby("region")
                .apply(lambda x: mean_squared_error(x.y_true, x.y_pred, squared=False))
                .to_dict()
            )
            results["region_averaged_rmse"] = np.mean(
                [val for val in results["regional_rmse"].values()]
            )
            results["regional_mae"] = (
                df.groupby("region")
                .apply(lambda x: mean_absolute_error(x.y_true, x.y_pred))
                .to_dict()
            )

        results["classification_report"] = classification_report(
            y_true, y_pred, labels=np.arange(1, 6), output_dict=True, zero_division=False
        )
        return results

    @staticmethod
    def calculate_density_metrics(y_true, y_pred, region=None):
        results = dict()
        results["overall_r_squared"] = r2_score(y_true, y_pred)
        results["overall_mape"] = mean_absolute_percentage_error(y_true, y_pred)

        if region is not None:
            df = pd.concat([y_true, y_pred, region], axis=1)
            df.columns = ["y_true", "y_pred", "region"]
            results["regional_r_squared"] = (
                df.groupby("region")
                .apply(lambda x: r2_score(x.y_true, x.y_pred, squared=False))
                .to_dict()
            )

        return results

    def calculate_feature_importance(self):
        feature_importances = pd.DataFrame(
            {
                "feature": self.model.feature_name(),
                "importance_gain": self.model.feature_importance(importance_type="gain"),
                "importance_split": self.model.feature_importance(importance_type="split"),
            }
        ).sort_values(by="importance_gain", ascending=False)

        return feature_importances

    def calculate_all_and_save(self):
        results = dict()

        # calculate severity metrics
        results["severity"] = self.calculate_severity_metrics(
            self.y_true_df["severity"], self.y_pred_df["severity"]
        )

        # calculate missing predictions
        results["samples_missing_predictions"] = {
            "count": float(self.missing_predictions_mask.sum()),
            "percent": float(self.missing_predictions_mask.mean()),
        }

        # add density metrics
        for density_var in ["log_density", "density_cells_per_ml"]:
            if density_var in self.y_pred_df.columns:
                results[density_var] = self.calculate_density_metrics(
                    self.y_true_df[density_var], self.y_pred_df[density_var]
                )

                # TODO: plot scatter plot

        # save out metrics
        with (self.save_dir / "results.json").open("w") as f:
            json.dump(results, f, indent=4)

        feature_importance = self.calculate_feature_importance()
        feature_importance.to_csv(self.save_dir / "feature_importance.csv", index=False)

        crosstab_plot = generate_and_plot_crosstab(
            self.y_true_df.severity, self.y_pred_df.severity
        )
        crosstab_plot.figure.savefig(self.save_dir / "crosstab.png")

        actual_density_boxplot = generate_actual_density_boxplot(
            self.y_true_df.density_cells_per_ml, self.y_pred_df.severity
        )
        actual_density_boxplot.figure.savefig(self.save_dir / "actual_density_boxplot.png")

        if "regional_rmse" in results:
            regional_barplot = generate_regional_barplot(results["regional_rmse"])
            regional_barplot.figure.savefig(self.save_dir / "regional_rmse.png")
