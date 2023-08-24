from pathlib import Path

from cyano.experiment.experiment import ExperimentConfig
from cyano.evaluate import EvaluatePreds

ASSETS_DIR = Path(__file__).parent / "assets"


def test_evaluate_preds(tmp_path):
    experiment_config = ExperimentConfig.from_file(ASSETS_DIR / "experiment_config.yaml")

    ep = EvaluatePreds(
        y_true_csv=experiment_config.predict_csv,
        y_pred_csv=experiment_config.save_dir / "preds.csv",
        save_dir=tmp_path / "metrics",
    )

    assert (ep.y_true_df.index == ep.y_pred_df.index).all()
    assert ep.y_true_df.index.name == "sample_id"
    assert ep.save_dir == tmp_path / "metrics"

    # two missing predictions are expected
    assert ep.missing_predictions_mask.sum() == 2
    assert len(ep.missing_predictions_mask) == 5

    # y_true and y_pred dfs should have three observations
    assert len(ep.y_pred_df) == 3

    assert experiment_config.target_col == "log_density"
    # check we have both log density and severity (which is always predicted)
    for col in ["log_density", "severity"]:
        assert (col in ep.y_true_df.columns) & (col in ep.y_pred_df.columns)


def test_calculate_all_and_save():
    # ep.clculate_all_and_save()
    # check we have metrics for both log density and severity in results
    # check filepaths are correct
    # check some metrics are correct
    pass


def test_calculate_severity_metrics():
    pass


def calculate_density_metrics():
    pass
