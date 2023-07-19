import pandas as pd
from pathlib import Path


def load_sample_list(sample_list_path: Path) -> pd.DataFrame:
    """Load and check dataframe with a list of samples for prediction

    Args:
        sample_list_path (Path): Path to a csv with columns for date,
            longitude, and latitude

    Returns:
        pd.DataFrame: DataFrame where the index is uid and there are
            columns for date, latitude, and longitude
    """

    if not sample_list_path.exists():
        raise FileNotFoundError(f"Path to sample list does not exist: {sample_list_path}")

    samples = pd.read_csv(sample_list_path)

    # Check data
    # - has columns for date, lat, and long
    # - date is within range for satellite imagery
    # - cast columns as correct dtypes and ensure there are no errors

    # Clean data
    # - remove duplicates and log

    # Add uid column with unique identifiers

    return samples


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Load and check dataframe with a list of labels fo training

    Args:
        labels_path (Path): Path to a csv with columns for date,
            longitude, latitude, and severity

    Returns:
        pd.DataFrame: DataFrame where the index is uid and there are
            columns for date, latitude, longitude, and severity
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"Path to labels does not exist: {labels_path}")

    labels = pd.read_csv(labels_path)

    # Check data
    # - has columns for date, lat, long, and severity
    # - date is within range for satellite imagery
    # - check that all severity values are in correct range
    # - cast columns as correct dtypes and ensure there are no errors

    # Clean data
    # - remove duplicates and log

    return labels
