import tempfile
import yaml
from zipfile import ZipFile

import lightgbm as lgb
from loguru import logger
import pandas as pd
from pathlib import Path

from cyano.config import TrainConfig, PredictConfig, FeaturesConfig
from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import add_unique_identifier
# from cyano.models.cyano_model import CyanoModel


# def run_experiment(experiment_config):
#     # create pipeline from experiment config
#     # train
#     # use that file to predict
#     # write out predictions
#     # write out experiment config / metrics / etc
#     pass



class CyanoModelPipeline:
    def __init__(self, config, cache_dir=None):
        # self.config = config  # is config static metadata?
        self.cache_dir = tempfile.TemporaryDirectory().name if cache_dir is None else cache_dir
        self.features_config = config.features_config
        self.model_config = config.tree_model_config
        # self.predict_config = config.predict_config
        self.samples = None
        self.labels = None

        # make cache dir
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)

    def prep_train_data(self, data, debug=False):
        """Load labels and save out samples with UIDs
        """
        ## Load labels
        labels = pd.read_csv(data)
        labels = labels[["date", "latitude", "longitude", "severity"]]
        labels = add_unique_identifier(labels)
        if debug:
            labels = labels.head(10)
        
        # Save out samples with uids
        labels.to_csv(Path(self.cache_dir) / "train_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {labels.shape[0]:,} samples for training")

        self.samples = labels[["date", "latitude", "longitude"]]
        self.labels = labels["severity"]

        return self.samples, self.labels

    def prepare_features(self):
        if self.samples is None:
            raise ValueError("No samples found")
        
        ## Identify satellite data
        satellite_meta = identify_satellite_data(self.samples, self.features_config, self.cache_dir)
        save_satellite_to = Path(self.cache_dir) / "satellite_metadata_train.csv"
        satellite_meta.to_csv(save_satellite_to, index=False)
        logger.info(
            f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
        )

        ## Download satellite data
        download_satellite_data(satellite_meta, self.samples, self.features_config, self.cache_dir)

        ## Download non-satellite data
        if self.features_config.climate_features:
            download_climate_data(self.samples, self.features_config, self.cache_dir)
        if self.features_config.elevation_features:
            download_elevation_data(self.samples, self.features_config, self.cache_dir)
        logger.success(f"Raw source data saved to {self.cache_dir}")

        ## Generate features
        features = generate_features(self.samples, self.features_config, self.cache_dir)
        save_features_to = Path(self.cache_dir) / "features_train.csv"
        features.to_csv(save_features_to, index=True)
        logger.success(
            f"{features.shape[1]:,} features for {features.shape[0]:,} samples saved to {save_features_to}"
        )

        self.features = features
        return features


    def train_model(self):
        lgb_data = lgb.Dataset(self.features, label=self.labels.loc[self.features.index])
        
        ## Train model
        self.model = lgb.train(
            self.model_config.params.model_dump(),
            lgb_data,
            num_boost_round=self.model_config.num_boost_round,
        )

        return self.model

    def to_disk(self, save_path):
        ## Zip up model config and weights
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving model zip to {save_path}")
        with ZipFile(f"{save_path}", "w") as z:
            z.writestr(f"{save_dir}/config.yaml", yaml.dump(self.features_config.model_dump()))
            z.writestr(f"{save_dir}/lgb_model.txt", self.model.model_to_string())

    def run_training(self, train_csv, save_path="model.zip", debug=False):
        self.prep_train_data(train_csv, debug)
        self.prepare_features()
        self.train_model()
        self.to_disk(save_path)


    @classmethod
    def from_disk(filepath, cache_dir):
        # create instance of this from filepath
        # load model with weights
        # load weights + attach the feature information to pipeline object
        # do the parsing here not in the init
        pass

    # def to_disk():
        # pass

    def run_prediction(input_csv, output_path):
        # download features
        # predict
        pass


# pipeline = CyanoModelPipeline(TrainConfig(), cache_dir)
# pipeline.run_training(data_csv)

# pipeline = CyanoModelPipeline.from_disk(model_zip, cache_dir)
# pipeline.run_prediction(data_csv)

# pipeline = CyanoModelPipeline(TrainConfig(), cache_dir)
# # run_experiment(pipeline, data)