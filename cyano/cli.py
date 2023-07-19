from loguru import logger
from pathlib import Path
import typer

from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import download_satellite_data
from cyano.data.utils import load_sample_list, load_labels
from cyano.models.utils import EnsembledModel
from cyano.settings import PRODUCTION_MODEL_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_FEATURE_DIR = "tmp/features"


@app.command()
def train(labels_path: Path):
    df = load_labels()
    # other stuff happens


@app.command()
def predict(sample_list_path: Path, save_path: Path, features_dir: Path = DEFAULT_FEATURE_DIR):
    """_summary_

    Args:
        sample_list_path (Path): _description_
        save_path (Path): _description_
        features_dir (Path): tmp dir by default, will get deleted after
    """
    ## Load data
    df = load_sample_list(sample_list_path)
    logger.info(f"Loaded {df.shape[0]:,} rows of data for prediction")

    ## Query for from data sources and save
    download_satellite_data(df, features_dir=features_dir)
    download_climate_data(df, features_dir=features_dir)
    download_elevation_data(df, features_dir=features_dir)
    logger.success(f"Raw source data saved to {features_dir}")

    ## Generate features
    features = generate_features(df, features_dir)

    ## Load model
    model = EnsembledModel(PRODUCTION_MODEL_DIR)

    ## Predict
    preds = model.predict(features)

    preds.to_csv(save_path)
    logger.success(f"Predictions saved to {save_path}")


if __name__ == "__main__":
    app()
