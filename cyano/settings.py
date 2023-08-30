from pathlib import Path

REPO_ROOT = Path(__file__).parents[0].resolve()

RANDOM_STATE = 40

# Dictionary mapping severity levels to the minimum cells/mL in that level
SEVERITY_LEFT_EDGES = {1: 0, 2: 20000, 3: 100000, 4: 1000000, 5: 10000000}
