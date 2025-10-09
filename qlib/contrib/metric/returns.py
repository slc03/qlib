import pandas as pd

from .base import MetricBaseModel


class ReturnsModel(MetricBaseModel):
    """根据过去n日的收益率做决策"""
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
