from pathlib import Path

REPO_ROOT = Path(__file__).parents[0].resolve()

RANDOM_STATE = 40

SEVERITY_LEFT_EDGES = [0, 20000, 100000, 1000000, 10000000]

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
