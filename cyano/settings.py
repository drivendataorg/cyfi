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

AVAILABLE_SENTINEL_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B08a",
    "B09",
    "B10",
    "B11",
    "B12",
]
