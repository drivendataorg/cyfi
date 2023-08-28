#!/usr/bin/env python
# coding: utf-8

# Experiment with downsampling certain samples we have an even distribution between severity levels 1-4.
# 
# We'll use the features that a have already been generated for our current best experiment (third sentinel + land cover features, trained with folds). Compare the results to that [best experiment](https://docs.google.com/presentation/d/1zWrSMSivxylx_iH_aOapJfyziRsDuyuXOELduOn6x3c/edit#slide=id.g278eb39bdd6_0_43)

# **Takeaway**
# 
# - Performance is better in the west, but significantly worse in other regions
# - Log(density) metrics are worse (R-squared and MAPE)
# - R-squared is better in the west, but much worse in all other regions
# - Based on the crosstab, the model does appear better at identifying severities 3 and 4, and distinguishing between severities 2 and 3. Severity level 2 predictions are also generally higher. However, performance is much worse on severity 1.
# 
# On balance, it does not seem worth it to downsample severities 1 and 4

get_ipython().run_line_magic('load_ext', 'lab_black')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import shutil

from cloudpathlib import AnyPath
import pandas as pd

from cyano.data.utils import add_unique_identifier
from cyano.experiment.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.settings import RANDOM_STATE


tmp_dir = AnyPath("tmp_dir")
tmp_dir.mkdir(exist_ok=True)


experiment_dir = AnyPath(
    "s3://drivendata-competition-nasa-cyanobacteria/experiments/results/third_sentinel_with_folds"
)


train_features = pd.read_csv(experiment_dir / "features_train.csv", index_col=0)
train_features.head()


# add in actual labels
train = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/train.csv"
    )
)
train = add_unique_identifier(train)
train.head(2)


# ## Select samples
# 
# Select which samples we'll keep in our training

train.severity.value_counts().sort_index()


(sample_size := train.severity.value_counts().loc[2])


# shuffle train
train = train.sample(frac=1, random_state=RANDOM_STATE)

# select first n for each severity
downsampled_train = []
for severity in train.severity.unique():
    downsampled_train.append(train[train.severity == severity].head(sample_size))
downsampled_train = pd.concat(downsampled_train)

downsampled_train.severity.value_counts().sort_index()


downsampled_train.shape


train_features = train_features[train_features.index.isin(downsampled_train.index)]
train_features.shape


train_features.index.nunique()


train_features.head()


# ## Train model
# 
# Train a model on the downsampled features

# Load best yaml config
experiment = ExperimentConfig.from_file(experiment_dir / "config_artifact.yaml")


pipe = CyanoModelPipeline(
    features_config=experiment.features_config,
    model_training_config=experiment.model_training_config,
    target_col=experiment.target_col,
)


pipe.model_training_config


pipe.train_features = train_features
pipe.train_labels = downsampled_train["log_density"]
pipe.train_samples = downsampled_train[["date", "latitude", "longitude", "region"]]


pipe._train_model()


pipe._to_disk(tmp_dir / "downsampled_model.zip")


# ## Predict

# read in test features and test samples
test = pd.read_csv(
    AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
    )
)
test = add_unique_identifier(test)
print(test.shape)

test_features = pd.read_csv(experiment_dir / "features_test.csv", index_col=0)
print(test_features.shape)


pipe.predict_samples = test[["date", "latitude", "longitude"]]
pipe.predict_features = test_features


pipe._predict_model()


preds_path = tmp_dir / "preds.csv"
pipe._write_predictions(preds_path)


# ## Evaluate

from cyano.evaluate import (
    EvaluatePreds,
    generate_and_plot_crosstab,
    generate_actual_density_boxplot,
    generate_density_scatterplot,
    generate_density_kdeplot,
)


ep = EvaluatePreds(
    y_true_csv = AnyPath(
        "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
    ),
    y_pred_csv = preds_path,
    save_dir = "eval",
)


# #### Severity predictions

severity_results = ep.calculate_severity_metrics(
    y_true=ep.y_true_df["severity"], y_pred=ep.y_pred_df["severity"], region=ep.region
)

{key: item for key, item in severity_results.items() if key != "classification_report"}


generate_and_plot_crosstab(ep.y_true_df.severity, ep.y_pred_df.severity)


generate_actual_density_boxplot(
    ep.y_true_df.density_cells_per_ml, ep.y_pred_df.severity
)


# #### Log(density) predictions

ep.calculate_density_metrics(
    y_true=ep.y_true_df.log_density, y_pred=ep.y_pred_df.log_density, region=ep.region
)


generate_density_scatterplot(ep.y_true_df["log_density"], ep.y_pred_df["log_density"])


generate_density_kdeplot(ep.y_true_df.log_density, ep.y_pred_df.log_density)





shutil.rmtree(tmp_dir)




