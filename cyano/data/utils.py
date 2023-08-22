import hashlib

import numpy as np
import pandas as pd

from cyano.settings import SEVERITY_LEFT_EDGES


def add_unique_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with the columns ["latitude", "longitude", "date"],
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


def convert_density_to_severity(density_series: pd.Series) -> pd.Series:
    """Convert exact density to binned severity

    Args:
        density_series (pd.Series): Series containing density values

    Returns:
        pd.Series: Series containing severity buckets
    """
    density = pd.cut(
        density_series,
        SEVERITY_LEFT_EDGES + [np.inf],
        include_lowest=True,
        right=False,
        labels=np.arange(1, 6),
    ).astype(float)

    return density


def handle_missing_train(features: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """For training, drop any samples with no satellite imagery. For all other
    features, fill with the mean of that column

    Args:
        features (pd.DataFrame): Dataframe of training features

    Returns:
        pd.DataFrame
    """
    # Drop samples with no satellite imagery features
    missing_satellite_mask = features[config.satellite_image_features].isna().all(axis=1)
    logger.warning(
        f"Dropping {missing_satellite_mask.sum():,} samples with no satellite imagery from training"
    )

    features = features[~missing_satellite_mask].copy()

    # For remaining samples, fill missing values with the mean
    cols_withna = features.isna().any().loc[lambda x: x].index.tolist()
    num_samples_withna = features.isna().any(axis=1).sum()
    logger.info(
        f"Filling in remaining missing values for {num_samples_withna:,} samples. Columns with missing values: {cols_withna}"
    )
    for col in cols_withna:
        features[col] = features[col].fillna(features[col].mean())

    return features


def handle_missing_predict(features: pd.DataFrame) -> pd.DataFrame:
    """For prediction, do not drop any samples. Fill each missing values with
    the mean of that column

    Args:
        features (pd.DataFrame): Dataframe of features for prediction

    Returns:
        pd.DataFrame
    """
    cols_withna = features.isna().any().loc[lambda x: x].index.tolist()
    num_samples_withna = features.isna().any(axis=1).sum()
    logger.warning(
        f"Filling in missing values for {num_samples_withna:,} samples. Columns with missing values: {cols_withna}"
    )
    for col in features.columns:
        features[col] = features[col].fillna(features[col].mean())

    return features
