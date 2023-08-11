#!/usr/bin/env python
# coding: utf-8

# Recreate a model based on the third place solution of the [Tick Tick Bloom: Harmful Algal Bloom Detection Challenge](https://github.com/drivendataorg/tick-tick-bloom/tree/main)

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import yaml

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.experiment import ExperimentConfig
from cyano.settings import REPO_ROOT


DATA_DIR = REPO_ROOT.parent / "data/experiments"

SPLITS_DIR = DATA_DIR / "splits"
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


# #### Experiment with larger subset of comp data

train = pd.read_csv(SPLITS_DIR / "competition/train.csv")

train_sub_path = "train_sub.csv"
train_sub = train.sample(n=50, random_state=4)
train_sub.to_csv(train_sub_path, index=False)
train_sub.head(2)


test = pd.read_csv(SPLITS_DIR / "competition/test.csv")

test_sub_path = "test_sub.csv"
test_sub = test.sample(n=50, random_state=4)
test_sub.to_csv(test_sub_path, index=False)
test_sub.head(2)


experiment_save_dir = "competition_split_subset"
experiment_config = ExperimentConfig(
    features_config=features_config,
    train_csv=train_sub_path,
    predict_csv=test_sub_path,
    cache_dir=LOCAL_CACHE_DIR,
    save_dir=experiment_save_dir,
    debug=False,
)

experiment_config.run_experiment()


get_ipython().run_cell_magic('time', '', 'experiment_save_dir = "competition_split_subset"\nexperiment_config = ExperimentConfig(\n    features_config=features_config,\n    model_training_config=model_config,\n    train_csv=train_sub_path,\n    predict_csv=test_sub_path,\n    cache_dir=LOCAL_CACHE_DIR,\n    save_dir=experiment_save_dir,\n    debug=False,\n)\n\nexperiment_config.run_experiment()\n')


trained_pipeline = CyanoModelPipeline.from_disk(
    "competition_split_subset/model.zip", cache_dir=LOCAL_CACHE_DIR
)


model = trained_pipeline.model
type(model) == lgb.Booster


ft_importance = pd.DataFrame(
    {
        "feature": model.feature_name(),
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }
)
ft_importance.head()


ft_importance.sort_values(by="importance_gain", ascending=False).head(10)


# load model from disk


trained_pipeline = CyanoModelPipeline(features_config=features_config,
                                      model_training_config=model_config,
                                      cache_dir=LOCAL_CACHE_DIR,
                                      model=
                                      


# ## Training

TEST_ASSETS_DIR = REPO_ROOT.parent / "tests/assets"


pipeline = CyanoModelPipeline(
    features_config=features_config,
    model_training_config=model_config,
    cache_dir=LOCAL_CACHE_DIR,
)
pipeline.run_training(TEST_ASSETS_DIR / "train_data.csv", save_path="/tmp/model.zip")


pipeline.train_samples["region"] = ["south", "south", "south", "south", "south"]
pipeline.train_samples


pipeline.train_features


pipeline.train_labels


lgb_data = lgb.Dataset(
    pipeline.train_features,
    label=pipeline.train_labels.loc[pipeline.train_features.index],
)


model = lgb.train(
    pipeline.model_training_config.params.model_dump(),
    lgb_data,
    num_boost_round=pipeline.model_training_config.num_boost_round,
)


# ## Parallelize downloads

pipeline = CyanoModelPipeline(
    features_config=features_config,
    model_training_config=model_config,
    cache_dir=LOCAL_CACHE_DIR,
)


pipeline._prep_train_data(TEST_ASSETS_DIR / "train_data.csv")


# ***

sat_meta = identify_satellite_data(pipeline.train_samples, pipeline.features_config)
sat_meta.shape


sat_meta.head(2)


from cyano.data.satellite_data import download_satellite_data_new


cache_dir = "image_cache"


download_satellite_data_new(
    sat_meta, pipeline.train_samples, pipeline.features_config, cache_dir
)


from cyano.data.satellite_data import get_bounding_box, download_row
import functools

from cloudpathlib import AnyPath
import geopy.distance as distance
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import planetary_computer as pc
from pystac_client import Client, ItemSearch
import rioxarray
from tqdm import tqdm


NUM_THREADS = 4


cache_dir = "image_cache"
Path(cache_dir).mkdir(exist_ok=True)


imagery_dir = Path(cache_dir) / f"sentinel_{features_config.image_feature_meter_window}"


process_result = process_map(
    functools.partial(
        download_row,
        samples=pipeline.train_samples,
        imagery_dir=imagery_dir,
        config=features_config,
    ),
    sat_meta.iterrows(),
    max_workers=NUM_THREADS,
    chunksize=1,
    total=len(sat_meta),
)


sum(process_result)


for idx, row in tqdm(sat_meta.iterrows(), total=len(sat_meta)):
    download_row(row)


sample_row = pipeline.train_samples.loc[row.sample_id]
sample_row


(minx, miny, maxx, maxy) = get_bounding_box(
    sample_row.latitude,
    sample_row.longitude,
    features_config.image_feature_meter_window,
)


band = features_config.use_sentinel_bands[0]
band


unsigned_href = 
















