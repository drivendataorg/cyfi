import pandas as pd
from pathlib import Path
from typer.testing import CliRunner

from cyano.cli import app
from cyano.data.utils import add_unique_identifier

ASSETS_DIR = Path(__file__).parent / "assets"

runner = CliRunner()


def test_cli_predict(tmp_path, predict_data_path, predict_data, ensembled_model_path):
    # Run CLI command
    result = runner.invoke(
        app,
        [
            "predict",
            str(predict_data_path),
            "--model-path",
            str(ensembled_model_path),
            "--output-directory",
            str(tmp_path),
            "--keep-features",
        ],
    )
    assert result.exit_code == 0

    # Check that preds saved out
    assert (tmp_path / "preds.csv").exists()
    preds = pd.read_csv(tmp_path / "preds.csv")
    assert (preds.index == predict_data.index).all()
    assert Path(tmp_path / "sample_features.csv").exists()

    # Check that the missing / non missing values are expected
    missing_sample_mask = preds.sample_id == "e66ea0c31ba500d5d4ac4c610b8cf508"
    assert preds[~missing_sample_mask].severity.notna().all()
    assert preds[missing_sample_mask].severity.isna().all()

    # Check that samples_path is required
    result = runner.invoke(
        app,
        [
            "predict",
            "--model-path",
            str(ensembled_model_path),
            "--output-directory",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_cli_predict_invalid_files(tmp_path):
    # Raises an error when samples_path does not exist
    result = runner.invoke(
        app,
        [
            "predict",
            "not_a_path",
            "--model-path",
            str(ASSETS_DIR / "experiment/model.zip"),
            "--output-directory",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stdout


def test_cli_no_overwrite(tmp_path, train_data, train_data_path, ensembled_model_path):
    # Check that existing files aren't overwritten without the --overwrite flag
    train_data.to_csv(tmp_path / "preds.csv")

    result = runner.invoke(
        app,
        [
            "predict",
            str(train_data_path),
            "--model-path",
            str(ensembled_model_path),
            "--output-directory",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, FileExistsError)


def test_cli_predict_point(tmp_path, ensembled_model_path):
    # Run CLI command
    result = runner.invoke(
        app,
        [
            "predict-point",
            "-dt",
            "2021-05-17",
            "-lat",
            "36.05",
            "-lon",
            "-76.7",
            "--output-directory",
            str(tmp_path),
            "--model-path",
            str(ensembled_model_path),
        ],
    )
    assert result.exit_code == 0

    # Check that preds saved out
    preds_path = tmp_path / "point_pred.csv"
    assert preds_path.exists()
    preds = pd.read_csv(preds_path)
    assert preds.shape[0] == 1
    assert preds.sample_id.iloc[0] == "7284ae28904be4631eabfc4a3acf7872"


def test_cli_evaluate(tmp_path, evaluate_data_path):
    df = pd.read_csv(evaluate_data_path)
    df = add_unique_identifier(df)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=True)

    eval_dir = tmp_path / "evals"

    # Run CLI command
    result = runner.invoke(
        app,
        ["evaluate", str(data_path), str(data_path), "--save-dir", str(eval_dir)],
    )
    assert result.exit_code == 0

    # Check that appropriate files are in the eval_dir
    for file in [
        "actual_density_boxplot.png",
        "crosstab.png",
        "density_kde.png",
        "density_scatterplot.png",
        "results.json",
    ]:
        assert (eval_dir / file).exists()
