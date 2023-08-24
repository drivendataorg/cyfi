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
import matplotlib.pyplot as plt
import pandas as pd
from zipfile import ZipFile

from cyano.data.utils import add_unique_identifier
from cyano.evaluate import (
    EvaluatePreds,
    generate_and_plot_crosstab,
    generate_actual_density_boxplot,
    generate_regional_barplot,
)


s3_dir = AnyPath("s3://drivendata-competition-nasa-cyanobacteria/experiments")


tmp_save_dir = AnyPath("tmp_data")
tmp_save_dir.mkdir(exist_ok=True, parents=True)


# load ground truth
true_path = s3_dir / "splits/competition/test.csv"
true = pd.read_csv(true_path, index_col=0)
true.shape


# # With metadata features
# 
# With metadata feature for `rounded_longitude`

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


# ### Evaluate

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

# ### Which new samples do we get features for?
# 
# Are we only covering additional older samples, or does climate data increase our coverage on newer samples as well? this helps determine whether the tradeoff of slight decrease in performance is worth the additional coverage moving forward. 

test = pd.read_csv(s3_dir / "splits/competition/test.csv")
test = add_unique_identifier(test)
test["date"] = pd.to_datetime(test.date)
test.head(2)


new_coverage = test.loc[test_features[~test_features.has_satellite].sample_id]
new_coverage.shape


new_coverage.date.hist()
plt.title("Dates for samples with climate data but no satellite imagery")
plt.show()


new_coverage.date.dt.year.value_counts().sort_index()


# **Takeaway**
# 
# Most of the samples climate data enables us to cover are on the older side. There are almost none post-2017 that climate covers but satellite does not.

# # Without metadata feature
# 
# Look into performance without the metadata feature of `rounded_longitude`

preds = pd.read_csv(
    s3_dir / "results/third_sentinel_and_climate_no_meta/preds.csv", index_col=0
)
preds.shape


preds.isna().sum()


# ### Save out subsets
# 
# So that we can instantiate separate `EvaluatePreds` classes, save out two versions of the predictions -- one with only samples that have satellite imagery, one with only samples that *don't* have satellite imagery

# Load test features to see which samples had satellite imagery
test_features = pd.read_csv(
    s3_dir / "results/third_sentinel_and_climate_no_meta/features_test.csv"
)
test_features.shape, test_features.sample_id.nunique()


# how many have climate features?
# checks out, same as experiment with metadata
test_features[test_features.SPFH_max.notna()].sample_id.nunique()


# same number have satellite features as well
test_features["has_satellite"] = test_features.AOT_mean.notna()
test_features[test_features.has_satellite].sample_id.nunique()


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
archive = ZipFile(s3_dir / "results/third_sentinel_and_climate_no_meta/model.zip", "r")
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


# ### Evaluate

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


feature_importance = pd.read_csv(
    s3_dir
    / "results/third_sentinel_and_climate_no_meta/metrics/feature_importance.csv",
    index_col=0,
)

# what are the top features by importance gain?
feature_importance.sort_values(by="importance_gain", ascending=False).head(10)


# what are the top features by importance split?
feature_importance.sort_values(by="importance_split", ascending=False).head(10)








# delete temporary dir
shutil.rmtree(tmp_save_dir)





# ***
# 
# # scrap

from cyano.settings import REPO_ROOT
from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.data.climate_data import download_climate_data
from cyano.data.features import generate_features
import numpy as np


train = pd.read_csv(REPO_ROOT.parent / "tests/assets/train_data.csv")
train.shape


train = add_unique_identifier(train)


train.head()


eval = pd.read_csv(REPO_ROOT.parent / "tests/assets/evaluate_data.csv")
eval = add_unique_identifier(eval)

predict = pd.read_csv(REPO_ROOT.parent / "tests/assets/predict_data.csv")
predict = add_unique_identifier(predict)


meta = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/data/final/combined_final_release.csv"
    )
)
meta = add_unique_identifier(meta)
meta.head(2)





predict


pipe = CyanoModelPipeline(
    features_config=FeaturesConfig(
        use_sentinel_bands=["B02"],
        image_feature_meter_window=500,
        satellite_image_features=["B02_mean", "B02_min", "B02_max"],
        climate_features=["TMP_min", "SPFH_mean"],
        climate_variables=["TMP", "SPFH"],
        climate_level="2 m above ground",
    ),
    model_training_config=ModelTrainingConfig(),
    cache_dir=REPO_ROOT.parent / "tests/assets/feature_cache",
)


train_path = REPO_ROOT.parent / "tests/assets/train_data.csv"
pipe._prep_train_data(train_path, False)


download_climate_data(pipe.train_samples, pipe.features_config, pipe.cache_dir)


pipe.train_samples


sat_meta = pd.read_csv(REPO_ROOT.parent / "tests/assets/satellite_metadata.csv")


fts = generate_features(
    pipe.train_samples, sat_meta, pipe.features_config, pipe.cache_dir
)
fts


fts[["TMP_min", "SPFH_mean"]].notna().all()


np.isclose(fts.loc["9c601f226c2af07d570134127a7fda27", "SPFH_mean"], 0.01440739)


fts.loc["9c601f226c2af07d570134127a7fda27", "SPFH_mean"]




