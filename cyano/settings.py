from pathlib import Path

REPO_ROOT = Path(__file__).parents[0].resolve()

RANDOM_STATE = 40

DEFAULT_LGB_CONFIG = {
    "params": {
        "application": "regression",
        "metric": "rmse",
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbosity": -1,
        "bagging_seed": RANDOM_STATE,
        "seed": RANDOM_STATE,
        # "early_stopping_round": 100, # add back early stopping rounds when we have valid set
    },
    "num_boost_round": 100000,
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
