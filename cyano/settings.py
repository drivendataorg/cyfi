from enum import Enum

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


class FeatureCacheMode(str, Enum):
    """Enumeration of the modes available for for the feature file cache.

    Attributes:
        persistent (str): Cache is not removed.
        tmp_dir (str): Cache is stored in a
            [`TemporaryDirectory`](https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory)
            which is removed after training and after prediction
    """

    persistent = "persistent"  # cache stays as long as dir on OS does
    tmp_dir = "tmp_dir"  # cache is cleared after training and prediction
