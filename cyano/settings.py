from pathlib import Path

REPO_ROOT = Path(__file__).parents[1].resolve()

DEFAULT_CONFIG = {
    "num_threads": 4,
    "model_dir": "path/to/model/dir",
    "features_dir": "path/to/default/tmp/dir/for/source/data",
    "pc_collections": ["sentinel-2-l2a", "landsat-c2-l2"],
    "pc_days_search_window": 15,
    "pc_meters_search_window": 50000,
}
