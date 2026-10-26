import hashlib

import numpy as np
import pandas as pd

# Dictionary mapping severity levels to the minimum cells/mL in that level
SEVERITY_LEFT_EDGES = {"low": 0, "moderate": 20000, "high": 100000}


def add_unique_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with the columns []"latitude", "longitude", "date"],
    create a unique identifier for each row and set as the index

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with unique identifiers as the index
    """
    df = df.copy()
    uids = []

    # create UID based on lat/lon and date
    for row in df.itertuples():
        m = hashlib.md5()
        for s in (row.latitude, row.longitude, row.date):
            m.update(str(s).encode())
        uids.append(m.hexdigest())

    df["sample_id"] = uids
    return df.set_index("sample_id")


def calculate_cloud_percent(scl_array: np.ndarray) -> float:
    """Calculate the portion of pixels classified as clouds in a scene classification
    (SCL) band array. Cloud classes are 7 (unclassified or low probability clouds),
    8 (medium probability clouds), 9 (high probability clouds), and 10 (thin cirrus).

    Args:
        scl_array (np.ndarray): SCL band array with dimensions (band, y, x)

    Returns:
        float: Portion of pixels that are clouds, from 0 to 1
    """
    return ((scl_array >= 7) & (scl_array <= 10)).sum() / (scl_array.shape[1] * scl_array.shape[2])


def convert_density_to_severity(density_series: pd.Series) -> pd.Series:
    """Convert exact density to binned severity

    Args:
        density_series (pd.Series): Series containing density values

    Returns:
        pd.Series: Series containing severity buckets
    """
    density = pd.cut(
        density_series,
        list(SEVERITY_LEFT_EDGES.values()) + [np.inf],
        include_lowest=True,
        right=False,
        labels=SEVERITY_LEFT_EDGES.keys(),
    )

    return density


def convert_density_to_log_density(density_series: pd.Series) -> pd.Series:
    """Convert exact density to log density

    Args:
        density_series (pd.Series): Series containing density values

    Returns:
        pd.Series: Series containing log density
    """
    return np.log(density_series + 1).rename("log_density")


def convert_log_density_to_density(log_density_series: pd.Series) -> pd.Series:
    """Convert log density to exact density

    Args:
        log_density_series (pd.Series): Series containing log density values

    Returns:
        pd.Series: Series containing exact density
    """
    return (np.exp(log_density_series) - 1).rename("density_cells_per_ml")
