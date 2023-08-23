#!/usr/bin/env python
# coding: utf-8

# Look at the results of adding climate data. See the impact on performance for samples with satellite imagery vs. samples with only climate data

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import shutil
import yaml

from cloudpathlib import AnyPath
import lightgbm as lgb
import pandas as pd
from zipfile import ZipFile

from cyano.evaluate import (
    EvaluatePreds,
    generate_and_plot_crosstab,
    generate_actual_density_boxplot,
    generate_regional_barplot,
)


s3_dir = AnyPath("s3://drivendata-competition-nasa-cyanobacteria/experiments")


# load ground truth
true_path = s3_dir / "splits/competition/test.csv"
true = pd.read_csv(true_path, index_col=0)
true.shape


# load predictions
preds = pd.read_csv(
    s3_dir / "results/third_sentinel_and_climate/preds.csv", index_col=0
)
preds.shape


preds.isna().sum()


# ### Save out subsets with / without satellite imagery
# 
# So that we can instantiate separate `EvaluatePreds` classes, save out two versions of the predictions -- one with only samples that have satellite imagery, one with only samples that *don't* have satellite imagery

# Load test features to see which samples had satellite imagery
test_features = pd.read_csv(
    s3_dir / "results/third_sentinel_and_climate/features_test.csv"
)
test_features.shape, test_features.sample_id.nunique()


test_features.head(2)


test_features.isna().sum()


# all satellite features have the same number of missing values
test_features.isna().sum().value_counts()


test_features["has_climate"] = test_features.SPFH_max.notna()
test_features["has_satellite"] = test_features.AOT_mean.notna()

# how many samples have climate features?
test_features[test_features.has_climate].sample_id.nunique()


# how many samples have satellite features?
test_features[test_features.has_satellite].sample_id.nunique()


# all sample have either satellite or climate
test_features.groupby(
    ["has_climate", "has_satellite"], as_index=False
).sample_id.nunique()


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
archive = ZipFile(s3_dir / "results/third_sentinel_and_climate/model.zip", "r")
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

import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for i, key in enumerate(evals.keys()):
    generate_and_plot_crosstab(evals[key].y_true, evals[key].y_pred, ax=axes[i])
    axes[i].set_title(key)


fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
for i, key in enumerate(evals.keys()):
    generate_actual_density_boxplot(
        evals[key].metadata.density_cells_per_ml, evals[key].y_pred, ax=axes[i]
    )
    axes[i].set_title(key)


results = {key: evalpreds.calculate_metrics() for key, evalpreds in evals.items()}


fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i, metric in enumerate(
    [
        "regional_mae",
        "regional_rmse",
    ]
):
    regional_scores = pd.DataFrame({key: res[metric] for key, res in results.items()})
    regional_scores.plot(kind="bar", ax=axes[i])
    axes[i].set_title(metric)
plt.show()


# See RMSE in detail for regions
regional_scores


pd.DataFrame(results).loc[
    [
        "overall_rmse",
        "overall_mae",
        "overall_mape",
        "region_averaged_rmse",
    ]
]


# **Takeaway**
# 
# The difference in performance between samples with and without satellite imagery is not huge. Samples without satellite imagery have noticeably higher RMSE in the northeast and a small amount higher in the midwest and south, but actually lower RMSE in the west.
# 
# The difference in performance is much greater when looking at MAE and MAPE than when looking at RMSE.

feature_importance = pd.read_csv(
    s3_dir / "results/third_sentinel_and_climate/metrics/feature_importance.csv",
    index_col=0,
)

# what are the top features by importance gain?
feature_importance.sort_values(by="importance_gain", ascending=False).head(10)


# what are the top features by importance split?
feature_importance.sort_values(by="importance_split", ascending=False).head(10)


# **Takeaways**
# 
# All of our climate features are in the top most important features. Even though performance is somewhat lower on data with no satellite imagery, this implies that the model is still able to glean useful information with climate data alone.

# delete temporary dir
shutil.rmtree(tmp_save_dir)
















