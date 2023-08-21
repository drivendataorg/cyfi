from cloudpathlib import S3Path
from loguru import logger
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

    logger.info("Creating train and test subsets")
    train = df.loc[split_df.split == "train"]
    test = df.loc[split_df.split == "test"]

    logger.info(f"Writing out to {SPLITS_PARENT_DIR}/{split_dir}")
    with (SPLITS_PARENT_DIR / split_dir / "train.csv").open("w") as f:
        train.to_csv(f, index=True)

    with (SPLITS_PARENT_DIR / split_dir / "test.csv").open("w") as f:
        test.to_csv(f, index=True)


@app.command()
def make_train_test_competition_post_2016_split(split_dir="competition_post_2016"):
    for split in ["train", "test"]:
        with (SPLITS_PARENT_DIR / "competition" / f"{split}.csv").open("r") as f:
            df = pd.read_csv(f)
            df = df[pd.to_datetime(df.date).dt.year >= 2016]

            with (SPLITS_PARENT_DIR / split_dir / f"{split}.csv").open("w") as f:
                df.to_csv(f, index=False)


if __name__ == "__main__":
    app()
