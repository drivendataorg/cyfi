# Advanced Use

CyFi is designed to be highly configurable, allowing advanced users to train their own models, experiment with different features, and use custom-trained models for prediction.

## Training Your Own Model

CyFi uses a configuration-driven approach for model training and experimentation. You can specify a configuration in a YAML file to control how features are generated and how the model is trained.

### Configuration Structure

There are three main configuration classes:

1.  **`ExperimentConfig`**: The top-level configuration that specifies input data paths (`train_csv`, `predict_csv`), the output directory (`save_dir`), and the cache directory.
2.  **`FeaturesConfig`**: Controls the feature generation process, including the satellite search window, the bands used, and the specific statistical features to calculate.
3.  **`CyFiModelConfig`**: Controls the machine learning model hyperparameters (currently using LightGBM).

### Sample YAML Configuration

Create a YAML file (e.g., `retrain.yaml`) to define your experiment:

```yaml
# Input data paths
train_csv: data/my_samples_train.csv
predict_csv: data/my_samples_evaluate.csv
save_dir: outputs/custom_model

# Feature generation settings
features_config:
  pc_days_search_window: 30
  pc_meters_search_window: 2000
  max_cloud_percent: 0.05
  filter_to_water_area: true
  # You can customize which SCL values are considered clouds or water
  scl_cloud_values: [7, 8, 9, 10]
  scl_water_values: [6]

# Model training settings
cyfi_model_config:
  target_col: "log_density"
  n_folds: 5
  params:
    learning_rate: 0.1
    num_leaves: 31
```

## Running an Experiment

To run an experiment using your YAML configuration, use the `cyfi/experiment.py` script:

```bash
python cyfi/experiment.py path/to/retrain.yaml
```

This will:
1.  Download and cache the necessary satellite imagery.
2.  Generate features based on your configuration.
3.  Train a cross-validated model.
4.  Save the model assets and evaluation metrics to your specified `save_dir`.

## Using a Custom Model for Prediction

Once you have trained a custom model, you can use it with the standard `cyfi predict` command by providing the path to your model assets (either a directory or a `.zip` file created during training):

```bash
cyfi predict sample_points.csv --model-path outputs/custom_model/model
```

By default, CyFi uses a pre-trained production model, but the `--model-path` (or `-m`) option allows you to swap it out for any compatible model trained using the CyFi pipeline.
