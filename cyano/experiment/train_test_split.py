from cloudpathlib import S3Path
from loguru import logger
import numpy as np
import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

S3_COMP_BUCKET = S3Path("s3://drivendata-competition-nasa-cyanobacteria")

# each command will create a child directory in here containing train and test csvs
SPLITS_PARENT_DIR = S3_COMP_BUCKET / "experiments" / "splits"


@app.command()
def make_train_test_competition_split(split_dir="competition"):
    """Write out train and test files using the competition split for use in experiments."""

    logger.info("Loading competition dataset")
    df = pd.read_csv(S3_COMP_BUCKET / "data/final/combined_final_release.csv", index_col=0)

    logger.info("Getting competition split info")
    split_df = pd.read_csv(
        S3_COMP_BUCKET / "data/interim/processed_unified_labels.csv", index_col=0
    )[["split"]]

    # add log density
    df["log_density"] = np.log(df.density_cells_per_ml + 1)

    logger.info("Creating train and test subsets")
    train = df.loc[split_df.split == "train"]
    test = df.loc[split_df.split == "test"]

    logger.info(f"Writing out to {SPLITS_PARENT_DIR}/{split_dir}")
    with (SPLITS_PARENT_DIR / split_dir / "train.csv").open("w") as f:
        train.to_csv(f, index=True)

    with (SPLITS_PARENT_DIR / split_dir / "test.csv").open("w") as f:
        test.to_csv(f, index=True)


@app.command()
def make_train_test_competition_water_distance_split(
    split_dir: str = "competition_train_near_water", filter_distance_m: int = 1000
):
    """Write out train and test files using the competition split for use in experiments. Filter the train set to samples within filter_distance_m
    meters of water based on Google Earth Engine.
    """
    # Update train set
    with (SPLITS_PARENT_DIR / "competition/train.csv").open("r") as f:
        train = pd.read_csv(f)
        logger.info(f"Loaded {train.shape[0]:,} competition train samples")

    train = train[train.distance_to_water_m < filter_distance_m]
    logger.info(f"Filtered to {train.shape[0]:,} samples within {filter_distance_m:,} m of water")

    save_to = SPLITS_PARENT_DIR / f"{split_dir}_{filter_distance_m}m/train.csv"
    logger.info(f"Writing out train samples to {save_to}")
    with (save_to).open("w") as f:
        train.to_csv(f, index=False)


if __name__ == "__main__":
    app()
