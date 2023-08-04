import yaml

from pathlib import Path
import typer

from cyano.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def experiment(
    config_path: Path = typer.Argument(exists=True, help="Path to an experiment configuration")
):
    """Run an experiment"""
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config = ExperimentConfig(**config_dict)

    config.run_experiment()


@app.command()
def predict(
    samples_path: Path = typer.Argument(
        exists=True, help="Path to a csv of samples with columns for date, longitude, and latitude"
    ),
    model_zip: Path = typer.Argument(exists=True, help="Path to a trained model zip"),
    output_path: Path = typer.Option(
        default="preds.csv", help="Destination to save predictions csv"
    ),
):
    """Load an existing cyanobacteria prediction model and generate
    severity level predictions for a set of samples.
    """
    pipeline = CyanoModelPipeline.from_disk(model_zip)
    pipeline.run_prediction(samples_path, output_path)


@app.command()
def evaluate(
        y_pred_csv: Path = typer.Argument(exists=True, help="Path to a csv of samples with columns for date, longitude, latitude, and severity"),
        y_true_csv: Path = typer.Argument(exists=True, help="Path to a csv of samples with columns for date, longitude, latitude, and severity, with optional metadata columns"),
        save_dir: Path = typer.Option(default=Path.cwd() / "metrics", help="Folder in which to save out metrics and plots.")
    ):
    EvaluatePreds(y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir).calculate_all_and_save()


if __name__ == "__main__":
    app()
