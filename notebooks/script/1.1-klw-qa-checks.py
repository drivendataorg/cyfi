#!/usr/bin/env python
# coding: utf-8

# Temporal QA checks. See the results for prediction on the same location multiple times over the course of 4 weeks.
# 
# Before this notebook can be run:
# 1. Run notebook 1.0 to generate `experiments/results/temporal_qa_checks/samples.csv`
# 2. Run `python cyano/cli.py predict experiments/results/temporal_qa_checks/samples.csv --output-path experiments/results/temporal_qa_checks/preds.csv` to generate predictions

get_ipython().run_line_magic('load_ext', 'lab_black')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

from cyano.data.utils import add_unique_identifier


EXPERIMENT_DIR = Path("../experiments/results/temporal_qa_checks")


true = pd.read_csv(EXPERIMENT_DIR / "samples.csv")
true = add_unique_identifier(true)
print(true.shape)
true.head()


preds = pd.read_csv(EXPERIMENT_DIR / "preds.csv")

preds = preds.sort_values(by=["latitude", "longitude", "date"])
preds["original_sample_id"] = true.loc[preds.sample_id].original_sample_id.values
preds["region"] = true.loc[preds.sample_id].region.values

# add actual predicted density
preds["density"] = np.exp(preds.log_density) - 1

print(preds.shape)
preds.head()


# ### Process prediction data

preds.isna().sum()


# drop rows with no prediction
preds = preds[preds.log_density.notna()]
preds.isna().sum().sum()


# how many locations do we have?
preds[["latitude", "longitude"]].drop_duplicates().shape


# how many original samples?
preds.original_sample_id.nunique()


# add cumulative stats so we can identify the date order of samples within a location
def _add_cumulative_stats(sub_df):
    sub_df["week_at_location"] = range(len(sub_df))

    sub_df["change"] = sub_df.density.diff()
    sub_df["abs_pct_change"] = np.abs(sub_df.density.diff() / sub_df.density.shift(1))

    sub_df["change_log"] = sub_df.log_density.diff()
    sub_df["abs_pct_change_log"] = np.abs(
        sub_df.log_density.diff() / sub_df.log_density.shift(1)
    )

    return sub_df


preds = (
    preds.groupby(["original_sample_id"])
    .apply(_add_cumulative_stats)
    .reset_index(drop=True)
)


preds.head(10)[
    [
        "date",
        "latitude",
        "longitude",
        "log_density",
        "density",
        "change",
        "abs_pct_change",
        "week_at_location",
    ]
]


# ### Evaluate

# **What is the percent change between dates at the same location?**
# 
# - Most are pretty small in log densities. In 75% of cases, the predicted log density changes by less than 6% between weeks
# - With absolute density, the percent change runs the gamut and is large in some cases. It is still generally not huge -- in half of cases, the predicted exact density changes by less than ~30% between weeks.

preds.filter(regex="change").describe()


# See log density over time by region
_, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 7))

for region, ax in zip(preds.region.unique(), axes.flatten()):
    sns.lineplot(
        data=preds[preds.region == region],
        x="week_at_location",
        y="log_density",
        hue="original_sample_id",
        legend=False,
        ax=ax,
    )
    ax.set_title(region.capitalize())

plt.suptitle("Predicted log(density) over time at the same location")
plt.tight_layout()


# Log density is fairly consistent at each location and doesn't wildly oscillate.

# See density over time by region
_, axes = plt.subplots(2, 2, sharex=True, figsize=(9, 7))

for region, ax in zip(preds.region.unique(), axes.flatten()):
    sns.lineplot(
        data=preds[preds.region == region],
        x="week_at_location",
        y="density",
        hue="original_sample_id",
        legend=False,
        ax=ax,
    )
    ax.set_title(region.capitalize())

plt.suptitle("Predicted density over time at the same location")
plt.tight_layout()


# same as above but with a more reasonable scale
_, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 7))

for region, ax in zip(preds.region.unique(), axes.flatten()):
    sns.lineplot(
        data=preds[preds.region == region],
        x="week_at_location",
        y="density",
        hue="original_sample_id",
        legend=False,
        ax=ax,
    )
    ax.set_title(region.capitalize())
    ax.set_ylim([0, 2e5])

plt.suptitle("Predicted density over time at the same location")
plt.tight_layout()


# For exact density, larger swings are more common. However, we do still generally see relative consistency over time.

# what is the severity level range at one location over time?
severity_range = preds.groupby("original_sample_id").agg(
    min_severity=("severity", "min"),
    max_severity=("severity", "max"),
)
severity_range["severity_range"] = (
    severity_range.max_severity - severity_range.min_severity
)
severity_range.severity_range.value_counts().sort_index()


# No sample changes by more than one severity level over time.

severity_range.groupby(["min_severity", "max_severity"]).size().rename(
    "count"
).sort_index().to_frame()







