#!/usr/bin/env python
# coding: utf-8

# Use gridsearch to see if there are more optimal lightGBM parameters we should be using.
# 
# We'll use the features that a have already been generated for our current best experiment (third sentinel + land cover features, trained with folds). Compare the results to that [best experiment](https://docs.google.com/presentation/d/1zWrSMSivxylx_iH_aOapJfyziRsDuyuXOELduOn6x3c/edit#slide=id.g278eb39bdd6_0_43)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import yaml

from cloudpathlib import AnyPath
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV

from cyano.data.utils import add_unique_identifier
from cyano.experiment.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.settings import RANDOM_STATE


tmp_dir = AnyPath("tmp_dir")
tmp_dir.mkdir(exist_ok=True)


experiment_dir = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/results/third_sentinel_with_folds"
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


train.loc[train_features.index].log_density


# Try params from each of the winners

param_grid = {
    'max_depth': [-1, 8],
    # 'num_leaves': [31],
    # 'learning_rate': [0.005, 0.1],
    # 'bagging_fraction': [0.3, 1.0],
    # 'feature_fraction': [0.3, 1.0],
    'min_split_gain': [0.0, 0.1],
    # 'n_estimators': [1000, 100000, 470], # same as num_boost_round
}


lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse')


grid_search = GridSearchCV(
    estimator = lgb_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='r2'
)


grid_search.fit(
    train_features,
    train.loc[train_features.index].log_density
)


# If we don't specify 'scoring' in `grid_search`, I think score is the `lgb_model.score` method, which is R^2. Unclear whether specifying `metric="rmse"` makes the score that is returned RMSE.

# with scoring specified in grid_search
pd.DataFrame(grid_search.cv_results_).sort_values(by='mean_test_score')


# without scoring specified in grid_search
pd.DataFrame(grid_search.cv_results_).sort_values(by='mean_test_score')


lgb_model.score


get_ipython().run_line_magic('pinfo', 'lgb_model.score')


grid_search.get_params()


grid_search.best_estimator_.get_params()


lgb_data = lgb.Dataset(
    train_features,
    train.loc[train_features.index].log_density,
)







