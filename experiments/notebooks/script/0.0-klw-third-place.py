#!/usr/bin/env python
# coding: utf-8

# Recreate a model based on the third place solution of the [Tick Tick Bloom: Harmful Algal Bloom Detection Challenge](https://github.com/drivendataorg/tick-tick-bloom/tree/main)

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import yaml

from cloudpathlib import AnyPath

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.experiment import ExperimentConfig
from cyano.settings import REPO_ROOT


DATA_DIR = REPO_ROOT.parent / "data/experiments"
S3_DATA_DIR = AnyPath("s3://drivendata-competition-nasa-cyanobacteria") / "experiments"

SPLITS_DIR = S3_DATA_DIR / "splits"
LOCAL_CACHE_DIR = DATA_DIR / "cache"
EXPERIMENT_SAVE_DIR = DATA_DIR / "rerun_third"
EXPERIMENT_SAVE_DIR.mkdir(exist_ok=True, parents=True)


# ## Settings
# 
# Write config to match third place code

use_sentinel_bands = [
    "AOT",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
    "SCL",
    "WVP",
]
sat_image_fts = [
    f"{band}_{stat}"
    for band in use_sentinel_bands
    for stat in ["mean", "min", "max", "range"]
]
sat_image_fts += ["NDVI_B04", "NDVI_B05", "NDVI_B06", "NDVI_B07"]
sat_image_fts[:6]


len(sat_image_fts)


features_config = FeaturesConfig(
    image_feature_meter_window=200,
    n_sentinel_items=15,
    pc_meters_search_window=5000,
    pc_days_search_window=15,
    use_sentinel_bands=use_sentinel_bands,
    satellite_image_features=sat_image_fts,
    satellite_meta_features=["month", "days_before_sample"],
    metadata_features=["rounded_longitude"],
)


model_config = ModelTrainingConfig(
    num_boost_round=100000,
    params={
        "application": "regression",
        "metric": "rmse",
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
    },
)


experiment_config = ExperimentConfig(
    features_config=features_config,
    train_csv=SPLITS_DIR / "competition/train.csv",
    predict_csv=SPLITS_DIR / "competition/test.csv",
    cache_dir=LOCAL_CACHE_DIR,
    save_dir=EXPERIMENT_SAVE_DIR,
)

with (EXPERIMENT_SAVE_DIR / "experiment_config.yaml").open("w") as fp:
    yaml.dump(experiment_config.model_dump(), fp)


# ## Run experiment

experiment_config.run_experiment()













