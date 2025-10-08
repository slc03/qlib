import pandas as pd
from typing import Text, Union, Optional

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class MetricBaseModel(Model):
    """Use metrics as the base class for model prediction.    
    This class does not require training and directly uses metrics for prediction."""
    def __init__(self):
        pass

    def fit(self, dataset: Optional[DatasetH] = None):
        """Do nothing"""
        print("No need to train for MetricBaseModel")

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """Process method"""
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        """Predict method"""
        X_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self.process(X_test)
