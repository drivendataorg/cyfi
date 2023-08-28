#!/usr/bin/env python
# coding: utf-8

# Look at the results of adding elevation data. See the impact on performance for samples with satellite imagery vs. samples with only elevation data

# %load_ext lab_black
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import shutil
import yaml

from cloudpathlib import AnyPath
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from zipfile import ZipFile

from cyano.data.utils import add_unique_identifier
from cyano.evaluate import (
    EvaluatePreds,
    generate_and_plot_crosstab,
    generate_actual_density_boxplot,
    generate_regional_barplot,
    generate_density_scatterplot,
    generate_density_kdeplot,
)


s3_dir = AnyPath("s3://drivendata-competition-nasa-cyanobacteria/experiments")


# ## Load data

# load ground truth
true_path = s3_dir / "splits/competition/test.csv"
true = pd.read_csv(true_path, index_col=0)
true.shape


# load predictions
preds = pd.read_csv(
    s3_dir / "results/third_sentinel_and_elevation/preds.csv", index_col=0
)
preds.shape, preds.severity.isna().sum()


# Load test features to see which samples had satellite imagery
test_features = pd.read_csv(
    s3_dir / "results/third_sentinel_and_elevation/features_test.csv"
)
test_features.shape, test_features.sample_id.nunique()


test_features.head(2)


# all satellite features have the same number of missing values
test_features.isna().sum().value_counts()


# no elevation features are missing in this dataframe
test_features.filter(regex='elevation').isna().any()


test_features["has_satellite"] = test_features.AOT_mean.notna()


test = pd.read_csv(s3_dir / "splits/competition/test.csv")
test = add_unique_identifier(test)
test["date"] = pd.to_datetime(test.date)
test.head(2)


test['has_elevation'] = test.index.isin(test_features.sample_id)
test['has_satellite'] = test.index.isin(test_features[test_features.has_satellite].sample_id)


test_features['region'] = test.loc[test_features.sample_id].region.values


# ## EDA

# how many samples have elevation features?
test.has_elevation.value_counts()


# how many samples have satellite features?
test.has_satellite.value_counts(dropna=False)


# ### How does elevation differ by region?

elev_columns = test_features.filter(regex='elevation').columns


fig, axes = plt.subplots(len(elev_columns), 1, sharey=True, figsize=(6,6.5))

for col, ax in zip(elev_columns, axes):
    sns.boxplot(data=test_features, y='region', x=col, 
                flierprops={'markersize':1, 'alpha':0.3},
                ax=ax)
    ax.set_ylabel('')

plt.tight_layout()


# ### How does density differ with elevation?
# 
# Is there a significant correlation beyond region?

# combine elevation features and actual density
elev_density = test_features[
    ['sample_id', 'elevation_max', 'elevation_range', 'elevation_at_sample']
].drop_duplicates().set_index('sample_id')
assert elev_density.index.is_unique
elev_density['log_density'] = test.loc[elev_density.index].log_density.values
elev_density['region'] = test.loc[elev_density.index].region.values


plt.scatter(
    x=elev_density.log_density, y=elev_density.elevation_at_sample,
    s=1,
    alpha=0.2
)
plt.xlabel('Log(density)')
plt.ylabel('Elevation at sample')
plt.tight_layout()


sns.scatterplot(
    data=elev_density,
    x='log_density',
    y='elevation_at_sample',
    hue='region',
    s=5,
    alpha=0.3
)
                


# It makes sense that we don't see elevation as significantly helping identify high-severity samples in the west. While lower-elevation samples in the west are very concentrated at high densities, higher-elevation samples in the west have a much broader range of density.

# ### Which new samples do we get features for?
# 
# Are we only covering additional older samples, or does elevation data increase our coverage on newer samples as well?
# 
# We get elevation data for 100% of samples.

pd.crosstab(test.date.dt.year, test.has_satellite).sort_index()


# There are very few samples post-2017 that we do not get satellite imagery for. It is likely not worth adding elevation data just to help with coverage -- we should only add it if it improves model performance.

# ### Save out subsets with / without satellite imagery
# 
# So that we can instantiate separate `EvaluatePreds` classes, save out two versions of the predictions -- one with only samples that have satellite imagery, one with only samples that *don't* have satellite imagery

tmp_save_dir = AnyPath("tmp_data")
tmp_save_dir.mkdir(exist_ok=True, parents=True)

preds_sat_path = tmp_save_dir / "preds_with_sat.csv"
preds_no_sat_path = tmp_save_dir / "preds_no_sat.csv"


# Save out subset with satellite imagery
preds_sat = preds.loc[test_features[test_features.has_satellite].sample_id.unique()]
print(preds_sat.shape)
preds_sat.to_csv(preds_sat_path, index=True)


# Save out subset without satellite imagery
preds_no_sat = preds.loc[test_features[~test_features.has_satellite].sample_id]
print(preds_no_sat.shape)
preds_no_sat.to_csv(preds_no_sat_path, index=True)


# Load model
archive = ZipFile(s3_dir / "results/third_sentinel_and_elevation/model.zip", "r")
model = lgb.Booster(model_str=archive.read("lgb_model.txt").decode())
type(model)


# ### Instantiate `EvaluatePreds` classes

evals = {}
evals["with_satellite"] = EvaluatePreds(
    true_path, preds_sat_path, "tmp/eval_sat", model
)


evals["without_satellite"] = EvaluatePreds(
    true_path, preds_no_sat_path, "tmp/eval_sat", model
)


# ## Evaluate

# #### Severity level

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for i, key in enumerate(evals.keys()):
    generate_and_plot_crosstab(
        evals[key].y_true_df.severity, evals[key].y_pred_df.severity, ax=axes[i]
    )
    axes[i].set_title(key)


fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for i, key in enumerate(evals.keys()):
    generate_actual_density_boxplot(
        evals[key].y_true_df.density_cells_per_ml,
        evals[key].y_pred_df.severity,
        ax=axes[i],
    )
    axes[i].set_title(key)


# get scores on severity predictions
severity_results = {
    key: evalpreds.calculate_severity_metrics(
        y_true=evalpreds.y_true_df.severity,
        y_pred=evalpreds.y_pred_df.severity,
        region=evalpreds.region,
    )
    for key, evalpreds in evals.items()
}

pd.DataFrame(severity_results).loc[
    [
        "overall_rmse",
        "overall_mae",
        "overall_mape",
        "region_averaged_rmse",
    ]
]


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i, metric in enumerate(
    [
        "regional_mae",
        "regional_rmse",
    ]
):
    regional_scores = pd.DataFrame(
        {key: res[metric] for key, res in severity_results.items()}
    )
    regional_scores.plot(kind="bar", ax=axes[i])
    axes[i].set_title(metric)
plt.show()


# See RMSE in detail for regions
regional_scores


# #### Log(density)

# log density metrics for samples with satellite imagery
evals["with_satellite"].calculate_density_metrics(
    y_true=evals["with_satellite"].y_true_df.log_density,
    y_pred=evals["with_satellite"].y_pred_df.log_density,
    region=evals["with_satellite"].region,
)


fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for i, key in enumerate(evals.keys()):
    generate_density_scatterplot(
        evals[key].y_true_df.log_density, evals[key].y_pred_df.log_density, ax=axes[i]
    )
    axes[i].set_title(key)


for i, key in enumerate(evals.keys()):
    generate_density_kdeplot(
        evals[key].y_true_df.log_density, evals[key].y_pred_df.log_density
    )


# #### Feature importance

feature_importance = pd.read_csv(
    s3_dir / "results/third_sentinel_and_elevation/metrics/feature_importance.csv",
    index_col=0,
)

# what are the top features by importance gain?
feature_importance.sort_values(by="importance_gain", ascending=False).head()


# what are the top features by importance split?
feature_importance.sort_values(by="importance_split", ascending=False).head()


# **Takeaways**
# 
# All of our elevation features are in the top most important features.









