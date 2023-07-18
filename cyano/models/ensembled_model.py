## Define class for ensembled set of models
from pathlib import Path


class EnsembledModel:
    def __init__(self, model_weights_dir: Path):
        self.model_weights_dir = model_weights_dir

    def train(self, features, labels):
        pass

    def predict(self, features):
        pass
