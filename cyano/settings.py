from pathlib import Path

REPO_ROOT = Path(__file__).parents[0].resolve()

DEFAULT_LGB_PARAMS = {
    "application": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "learning_rate": 0.005,
    "bagging_fraction": 0.3,
    "feature_fraction": 0.3,
    "min_split_gain": 0.1,
    "verbosity": -1,
    "data_random_seed": 2023,
    "early_stop": 500,
}

DEFAULT_CONFIG = {
    "num_threads": 4,
    "model_dir": "path/to/model/dir",
    "features_dir": "path/to/default/tmp/dir/for/source/data",
    "pc_collections": ["sentinel-2-l2a", "landsat-c2-l2"],
    "pc_days_search_window": 15,
    "pc_meters_search_window": 50000,
    "use_sentinel_bands": ["B03", "B04", "B05"],
}

EXP_CONFIG = {
    "num_threads": 4,
    "model_dir": "tmp/experiments/baseline/model",
    "features_dir": "tmp/experiments/baseline/features",
    "pc_collections": ["sentinel-2-l2a"],
    "pc_days_search_window": 30,
    "pc_meters_search_window": 1000,
    "use_sentinel_bands": ["B02", "B03", "B04"],
    "image_feature_meter_window": 500,
    "satellite_features": [
        "B02_mean",
        "B02_min",
        "B02_max",
        "B03_mean",
        "B03_min",
        "B03_max",
        "B04_mean",
        "B04_min",
        "B04_max",
    ],
    "climate_features": [],
    "elevation_features": [],
    "metadata_features": [],
    "lgb_params": {},
}
