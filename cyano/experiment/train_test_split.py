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


@app.command()
def make_train_test_by_provider(split_dir: str = "data_provider"):
    """Write out train and test files split by data provider. A few large
    or key geographically diverse datasets are kept as split between train
    and test using the competition split. The remainder are put entirely
    in train or test.
    """
    train_providers = [
        "EPA Central Data Exchange",
        "Delaware National Resources and the University of Delaware's Citizen Monitoring Program",
        "EPA Water Quality Data Portal",
        "Bureau of Water Kansas Department of Health and Environment",
        "Texas Commission on Environmental Quality",
        "Pennsylvania Department of Environmental Protection",
    ]
    test_providers = [
        "Connecticut State Department of Public Health",
        "Indiana State Department of Health",
        "Wyoming Department of Environmental Quality",
        "New Mexico Environment Department",
        "US Army Corps of Engineers",
    ]

    logger.info("Loading competition dataset")
    df = pd.read_csv(S3_COMP_BUCKET / "data/final/combined_final_release.csv", index_col=0)
    logger.info(f"Loaded {df.shape[0]:,} samples")

    logger.info("Getting competition split info")
    split_df = pd.read_csv(
        S3_COMP_BUCKET / "data/interim/processed_unified_labels.csv", index_col=0
    )
    df["comp_split"] = split_df.loc[df.index].split.values

    df["split"] = np.where(
        df.data_provider.isin(train_providers),
        "train",
        np.where(df.data_provider.isin(test_providers), "test", df.comp_split),
    )

    train = df[df.split == "train"].drop(columns=["comp_split", "split"])
    test = df[df.split == "test"].drop(columns=["comp_split", "split"])
    logger.info(
        f"Generated {train.shape[0]:,} train samples and {test.shape[0]:,} test samples. Writing out to {SPLITS_PARENT_DIR / split_dir}"
    )

    with (SPLITS_PARENT_DIR / f"{split_dir}/train.csv").open("w") as f:
        train.to_csv(f, index=True)

    with (SPLITS_PARENT_DIR / f"{split_dir}/test.csv").open("w") as f:
        test.to_csv(f, index=True)


if __name__ == "__main__":
    app()
