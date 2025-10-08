import pandas as pd
from typing import List, Text, Tuple, Union, Optional

from .base import MetricBaseModel


class ReturnsModel(MetricBaseModel):
    """A model that directly outputs the return of the next n days."""
    def __init__(self, windows: int=3):
        self.windows = windows

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """Process method"""
        retn = (
            X_test.groupby("instrument")["close"]
            .apply(lambda x: x / x.shift(self.windows) - 1)
        ).droplevel(0)
        retn = retn.reindex(X_test.index)
        return retn
