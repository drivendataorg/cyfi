#!/usr/bin/env python
# coding: utf-8

# Use gridsearch to see if there are more optimal lightGBM parameters we should be using.
# 
# We'll use the features that a have already been generated for our current best experiment (third sentinel + land cover features, trained with folds). Compare the results to that best experiment: `s3://drivendata-competition-nasa-cyanobacteria/experiments/results/filter_water_distance_550
# `

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import yaml

from cloudpathlib import AnyPath
import lightgbm as lgb
from loguru import logger
import pandas as pd
from sklearn.model_selection import GridSearchCV

from cyano.data.utils import add_unique_identifier
from cyano.experiment.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.settings import RANDOM_STATE


# ### Load data

tmp_dir = AnyPath("tmp_dir")
tmp_dir.mkdir(exist_ok=True)


experiment_dir = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/results/filter_water_distance_550"
)


train_features = pd.read_csv(experiment_dir / "features_train.csv", index_col=0)
train_features.head()


# Load train labels
train = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/train.csv"
    )
)
train = add_unique_identifier(train)
train.shape


train.head(2)


# ### Grid search
# 
# Try params from each of the winners

param_grid = {
    "max_depth": [-1, 8],
    "num_leaves": [31],
    "learning_rate": [0.005, 0.1],
    "bagging_fraction": [0.3, 1.0],
    "feature_fraction": [0.3, 1.0],
    "min_split_gain": [0.0, 0.1],
    "n_estimators": [100, 1000, 470],  # same as num_boost_round
}


# Note that this is slightly different than our process because we use LGB.Booster, which we cannot input to the GridSearch. With our grid search, we are not using a valid set or early stopping.

lgb_model = lgb.LGBMModel(objective="regression", metric="rmse")


grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error",
)


# Grid search CV always tries to maximize the score, so root mean squared error has to be negative

# Load past grid search results if we can
grid_search_results_path = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/grid_search.csv"
)
if grid_search_results_path.exists():
    logger.info("Loading existing grid search results")
    results = pd.read_csv(grid_search_results_path)
    results = results.sort_values(by="mean_test_score", ascending=False)

# Otherwise run grid search -- takes ~30 min
else:
    logger.info("Running grid search")
    grid_search.fit(train_features, train.loc[train_features.index].log_density)
    results = pd.DataFrame(grid_search.cv_results_).sort_values(
        by="mean_test_score", ascending=False
    )
    with grid_search_results_path.open("w") as fp:
        results.to_csv(fp, index=False)
    logger.success(f"Grid search results saved to {grid_search_results_path}")


results.shape


results.head()


# do we have multiple tied for first?
# yes, two are tied
results.rank_test_score.value_counts().sort_index().head()


results[results.rank_test_score == 1].filter(regex="param_")


# The only difference is param_bagging_fraction

# **Are there different other top params for different n_estimators?**
# 
# Our n_estimators doesn't exactly match the real process because we don't have a valid set and can't use early stopping. We are more interested in grid search's results for other parameters.

# Are there different other top params for different n_estimators?
by_estimator = []
include_cols = results.filter(regex="param_").columns.tolist() + ["mean_test_score"]

for n_est in results.param_n_estimators.unique():
    sub = results[results.param_n_estimators == n_est]
    sub = sub[sub.rank_test_score == sub.rank_test_score.min()][include_cols]
    by_estimator.append(sub)

pd.concat(by_estimator).set_index("param_n_estimators").T


# how much worse is the 100 estimator model with 0.3 feature fractions?
# not much
results[
    (results.param_n_estimators == 100) & (results.param_feature_fraction == 0.3)
].mean_test_score.max()


# how much worse is the 470 estimator model with 0.3 feature fractions?
# also not much
results[
    (results.param_n_estimators == 470) & (results.param_feature_fraction == 1.0)
].mean_test_score.max()


# n_estimators = 1000 also doesn't change much with feature_fraction
results[
    (results.param_n_estimators == 1000) & (results.param_feature_fraction == 1.0)
].mean_test_score.max()


param_grid


# **Takeaways**
# 
# Best set of LGB params based on grid search:
# 
# - `max_depth` = -1. This is the same as what we're already using (3rd place)
# 
# - `learning_rate` = 0.1. This is the same as what we're already using (3rd place)
# 
# - `bagging_fraction` = 1.0. The bagging fraction does not change the performance, and 1.0 is the default
# 
# - `feature_fraction` = 0.3. Feature fraction of 1.0 is best when we have only 100 boosting iterations, but 0.3 is best with either 470 or 1000. This makes sense because it helps deal with overfitting. When n_estimators is 470 using a feature_fraction of 1.0 instead of 0.3 has a more noticeable impact on the model than using a feature_fraction of 0.3 instead of 1.0 when n_estimators is 100 --> the risk of poor performance is greater is we stick with 1.0. From lightGBM:
#     > LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0. For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree
#     > 
#     > can be used to speed up training
#     > 
#     > can be used to deal with over-fitting
# 
# - `min_split_gain` = 0.0. This is the lightGBM default and the same as what we're already using (3rd place)












