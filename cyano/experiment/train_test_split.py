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
    split_dir: str = "competition_near_water",
    filter_distance_m: int = 1000,
    filter_test: bool = False,
):
    """Write out train and test files using the competition split for use in
    experiments. Filter the to samples within filter_distance_m meters of water
    based on Google Earth Engine.

    Args:
        split_dir (str, optional): Directory name to save splits to.
            Defaults to "competition_near_water".
        filter_distance_m (int, optional): Distance from water to include, in meters.
            Defaults to 1000.
        filter_test (bool, optional): Whether to also filter and write out a modified
            test set. Defaults to False.
    """
    if filter_test:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    for split in splits:
        # Load competition data
        with (SPLITS_PARENT_DIR / f"competition/{split}.csv").open("r") as f:
            df = pd.read_csv(f)
            logger.info(f"Loaded {df.shape[0]:,} competition {split} samples")

        # Filter and save
        df = df[df.distance_to_water_m < filter_distance_m]
        logger.info(f"Filtered to {df.shape[0]:,} samples within {filter_distance_m:,} m of water")

        save_to = SPLITS_PARENT_DIR / f"{split_dir}_{filter_distance_m}m/{split}.csv"
        logger.info(f"Saving to {save_to}")
        with (save_to).open("w") as f:
            df.to_csv(f, index=False)


if __name__ == "__main__":
    app()
