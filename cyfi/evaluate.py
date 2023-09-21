import json
from pathlib import Path
from typing import Optional

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
    r2_score,
)
from zipfile import ZipFile

from cyfi.data.utils import (
    add_unique_identifier,
    convert_density_to_log_density,
    convert_density_to_severity,
    SEVERITY_LEFT_EDGES,
)

SEVERITY_LEVEL_NAMES = list(SEVERITY_LEFT_EDGES.keys())
FIGSIZE = (6, 6)

# map severity levels to integers to we can calculate numerical metrics
SEVERITY_INT_MAPPING = dict(zip(SEVERITY_LEVEL_NAMES, range(len(SEVERITY_LEVEL_NAMES))))
# inverse mapping
SEVERITY_CAT_MAPPING = {v: k for k, v in SEVERITY_INT_MAPPING.items()}


def generate_and_plot_severity_crosstab(y_true, y_pred, normalize=False):
    to_plot = pd.crosstab(y_pred, y_true)

    # make sure crosstab is even on both axes
    for level in SEVERITY_LEVEL_NAMES:
        if level not in to_plot.index:
            to_plot.loc[level, :] = 0

        if level not in to_plot.columns:
            to_plot[level] = 0

    # reverse index order for plotting
    to_plot = to_plot.loc[SEVERITY_LEVEL_NAMES[::-1], SEVERITY_LEVEL_NAMES]
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
        order=SEVERITY_LEVEL_NAMES,
        showfliers=False,
    )
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("Actual density (cells/mL)")

    return ax


def generate_density_scatterplot(y_true, y_pred):
    _, ax = plt.subplots(figsize=(FIGSIZE))
    ax.set_xscale("log")
    ax.set_yscale("log")
    max_value = max(y_true.max(), y_pred.max()) * 1.1
    ax.axline(
        (0, 0),
        (max_value, max_value),
        color="black",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label="Predicted = Actual",
    )
    ax.scatter(x=y_true, y=y_pred, s=3, alpha=0.8)

    ax.set_xlim(10, max_value)
    ax.set_ylim(10, max_value)

    ax.set_xlabel(f"Actual {y_true.name}")
    ax.set_ylabel(f"Predicted {y_pred.name}")

    ax.legend()

    return ax


def generate_density_kdeplot(y_true, y_pred):
    to_plot = pd.concat([y_true, y_pred.loc[y_true.index]], axis=1)
    to_plot.columns = ["y_true", "y_pred"]

    _, ax = plt.subplots(figsize=FIGSIZE)
    fig = sns.kdeplot(
        # add 1 so we can do log scale if there are zero values
        data=to_plot + 1,
        y="y_pred",
        x="y_true",
        warn_singular=False,
        log_scale=True,
        ax=ax,
    )

    max_value = max(y_true.max(), y_pred.max()) * 1.1
    ax.set_xlim(10, max_value)
    ax.set_ylim(10, max_value)

    ax.set_xlabel(f"Actual {y_true.name}")
    ax.set_ylabel(f"Predicted {y_pred.name}")

    return fig


def generate_regional_barplot(regional_rmse):
    to_plot = pd.DataFrame({"regional_rmse": regional_rmse}).sort_values("regional_rmse")

    _, ax = plt.subplots()
    sns.barplot(to_plot.T, ax=ax)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSE")

    return ax


class EvaluatePreds:
    def __init__(
        self, y_true_csv: Path, y_pred_csv: Path, save_dir: Path, model_path: Optional[Path] = None
    ):
        """Instantate EvaluatePreds class. To automatically generate all key visualizations, run
        cls.calculate_all_and_save() after instantiation.
        """
        self.model_path = model_path

        # Load preds
        all_preds = pd.read_csv(y_pred_csv).set_index("sample_id")

        self.missing_predictions_mask = all_preds.severity.isna()
        self.y_pred_df = all_preds[~self.missing_predictions_mask].copy()
        logger.info(f"Evaluating on {len(self.y_pred_df):,} sample points (of {len(all_preds):,})")

        # Load ground truth
        y_true_df = pd.read_csv(y_true_csv)
        if "density_cells_per_ml" not in y_true_df.columns:
            raise ValueError("Evaluation data must include a `density_cells_per_ml` column")
        y_true_df = add_unique_identifier(y_true_df)

        try:
            self.y_true_df = y_true_df.loc[self.y_pred_df.index]
        except KeyError:
            raise IndexError(
                "Sample IDs for points (lat, lon, date) in y_pred_csv do not align with sample IDs in y_true_csv."
            )

        # Calculate severity from density
        self.y_true_df["severity"] = convert_density_to_severity(
            self.y_true_df.density_cells_per_ml
        )

        if "region" in self.y_true_df.columns:
            self.region = self.y_true_df.region
        else:
            self.region = None

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def calculate_severity_metrics(y_true, y_pred, region=None):
        results = dict()
        results["overall_rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        results["overall_mae"] = mean_absolute_error(y_true, y_pred)

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
            y_true.map(SEVERITY_CAT_MAPPING),
            y_pred.map(SEVERITY_CAT_MAPPING),
            output_dict=True,
            zero_division=False,
        )
        return results

    @staticmethod
    def calculate_log_density_metrics(y_true_df, y_pred_df, region=None):
        # Get log density
        y_true = convert_density_to_log_density(y_true_df.density_cells_per_ml)
        y_pred = convert_density_to_log_density(y_pred_df.density_cells_per_ml)

        results = dict()
        results["overall_r_squared"] = r2_score(y_true, y_pred)

        if region is not None:
            df = pd.concat([y_true, y_pred, region], axis=1)
            df.columns = ["y_true", "y_pred", "region"]
            results["regional_r_squared"] = (
                df.groupby("region").apply(lambda x: r2_score(x.y_true, x.y_pred)).to_dict()
            )

        return results

    def calculate_feature_importances(self):
        """Calculate feature importances for each model. Feature importances are saved
        out with the same index as the original model file. Eg. Importances for
        `lgb_model_0.txt` will be saved to `self.save_dir / feature_importance_0.txt`
        """
        archive = ZipFile(self.model_path, "r")
        model_files = sorted([name for name in archive.namelist() if "lgb_model" in name])
        logger.info(f"Calculating feature importances for {len(model_files)} model(s)")

        for idx, model_file in enumerate(model_files):
            # Load model
            model = lgb.Booster(model_str=archive.read(model_file).decode())

            # Calculate and save feature importance
            feature_importance = pd.DataFrame(
                {
                    "feature": model.feature_name(),
                    "importance_gain": model.feature_importance(importance_type="gain"),
                    "importance_split": model.feature_importance(importance_type="split"),
                }
            ).sort_values(by="importance_gain", ascending=False)
            feature_importance.to_csv(
                self.save_dir / f"feature_importance_model_{idx}.csv", index=False
            )

    def calculate_all_and_save(self):
        results = dict()

        # calculate severity metrics
        results["severity"] = self.calculate_severity_metrics(
            y_true=self.y_true_df["severity"].map(SEVERITY_INT_MAPPING),
            y_pred=self.y_pred_df["severity"].map(SEVERITY_INT_MAPPING),
            region=self.region,
        )

        # calculate missing predictions
        results["samples_missing_predictions"] = {
            "count": float(self.missing_predictions_mask.sum()),
            "percent": float(self.missing_predictions_mask.mean()),
        }

        # calculate log density metrics
        results["log_density"] = self.calculate_log_density_metrics(
            y_true_df=self.y_true_df,
            y_pred_df=self.y_pred_df,
            region=self.region,
        )

        # add plots
        density_scatter = generate_density_scatterplot(
            self.y_true_df.density_cells_per_ml,
            self.y_pred_df.density_cells_per_ml,
        )
        density_scatter.figure.savefig(self.save_dir / "density_scatterplot.png")

        density_kde = generate_density_kdeplot(
            self.y_true_df.density_cells_per_ml,
            self.y_pred_df.density_cells_per_ml,
        )
        density_kde.figure.savefig(self.save_dir / "density_kde.png")

        # save out metrics
        with (self.save_dir / "results.json").open("w") as f:
            json.dump(results, f, indent=4)

        if self.model_path is not None:
            self.calculate_feature_importances()

        crosstab_plot = generate_and_plot_severity_crosstab(
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
