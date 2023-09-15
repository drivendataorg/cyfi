import json
from pathlib import Path
import pytest

from cyano.data.utils import add_unique_identifier
from cyano.evaluate import EvaluatePreds


ASSETS_DIR = Path(__file__).parent / "assets"


def test_evaluate_preds(experiment_config, tmp_path):
    ep = EvaluatePreds(
        y_true_csv=experiment_config.predict_csv,
        y_pred_csv=experiment_config.save_dir / "preds.csv",
        save_dir=tmp_path / "metrics",
    )

    assert (ep.y_true_df.index == ep.y_pred_df.index).all()
    assert ep.y_true_df.index.name == "sample_id"
    assert ep.save_dir == tmp_path / "metrics"

    # one missing prediction is expected
    assert ep.missing_predictions_mask.sum() == 1
    assert len(ep.missing_predictions_mask) == 5

    # y_true and y_pred dfs should have four observations
    assert len(ep.y_pred_df) == 4

    assert experiment_config.cyano_model_config.target_col == "log_density"
    # check we have density_cells_per_ml and severity (which are always predicted)
    for col in ["density_cells_per_ml", "severity"]:
        assert (col in ep.y_true_df.columns) & (col in ep.y_pred_df.columns)


def test_evaluate_preds_missing_density(train_data, tmp_path):
    # Check that exact density column is required
    df = add_unique_identifier(train_data)
    df.to_csv(tmp_path / "y_pred.csv", index=True)
    df.drop(columns=["density_cells_per_ml"]).to_csv(tmp_path / "y_true.csv", index=True)

    with pytest.raises(ValueError):
        _ = EvaluatePreds(
            y_true_csv=tmp_path / "y_true.csv",
            y_pred_csv=tmp_path / "y_pred.csv",
            save_dir=tmp_path / "metrics",
        )


def test_calculate_feature_importances(experiment_config, ensembled_model_path, tmp_path):
    ep = EvaluatePreds(
        y_true_csv=experiment_config.predict_csv,
        y_pred_csv=experiment_config.save_dir / "preds.csv",
        model_path=ensembled_model_path,
        save_dir=tmp_path / "metrics",
    )
    ep.calculate_feature_importances()

    # Check that we have feature importance saved out for both ensembled models
    feature_importances = [p for p in (tmp_path / "metrics").iterdir()]
    assert len(feature_importances) == 2


def test_calculate_all_and_save(experiment_config, tmp_path):
    ep = EvaluatePreds(
        y_true_csv=experiment_config.predict_csv,
        y_pred_csv=experiment_config.save_dir / "preds.csv",
        save_dir=tmp_path / "metrics",
    )
    ep.calculate_all_and_save()

    with (tmp_path / "metrics/results.json").open("r") as f:
        results = json.load(f)

    # check we have metrics for both log density and severity in results
    assert list(results.keys()) == ["severity", "samples_missing_predictions", "log_density"]

    # we have region in our ground truth so we expect regional columns
    assert "region" in ep.y_true_df.columns
    assert list(results["severity"].keys()) == [
        "overall_rmse",
        "overall_mae",
        "regional_rmse",
        "region_averaged_rmse",
        "regional_mae",
        "classification_report",
    ]
    assert list(results["log_density"].keys()) == [
        "overall_r_squared",
        "overall_mape",
        "regional_r_squared",
    ]

    # if region is removed, region metrics are not calculated
    ep.region = None
    ep.calculate_all_and_save()
    with (tmp_path / "metrics/results.json").open("r") as f:
        results = json.load(f)

    assert list(results["severity"].keys()) == [
        "overall_rmse",
        "overall_mae",
        "classification_report",
    ]
