import json
from pathlib import Path

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

    # two missing predictions are expected
    assert ep.missing_predictions_mask.sum() == 1
    assert len(ep.missing_predictions_mask) == 5

    # y_true and y_pred dfs should have four observations
    assert len(ep.y_pred_df) == 4

    assert experiment_config.target_col == "log_density"
    # check we have both log density and severity (which is always predicted)
    for col in ["log_density", "severity"]:
        assert (col in ep.y_true_df.columns) & (col in ep.y_pred_df.columns)


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
        "overall_mape",
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
        "overall_mape",
        "classification_report",
    ]
