import re
import hashlib
import numpy as np
import pandas as pd
from typing import Text, Union, Optional, Literal

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class MetricBaseModel(Model):
    """Use metrics as the base class for model prediction.    
    This class does not require training and directly uses metrics for prediction."""
    def __init__(self):
        pass
    
    def _zscore_cs(self, s: pd.Series) -> pd.Series:
        """工具函数：横截面标准化 (Cross-sectional Z-score)"""
        return (s - s.dropna().mean()) / s.dropna().std()

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


class ExpressionFactor(MetricBaseModel):
    """
    通用表达式型因子计算器
    支持表达式字符串或算子树形式。
    
    Example:
        factor = ExpressionFactor(
            expr="zscore(roc(rzmre, 5)) - zscore(roc(close, 5))",
            group_by="instrument",
            normalize_by="datetime"
        )
        factor.process(X_test)
    """
    def __init__(self, expr: str, group_by: str = "instrument", normalize_by: Optional[str] = None, params: Optional[dict] = None):
        self.expr = expr
        self.group_by = group_by
        self.normalize_by = normalize_by
        self.params = params or {}
        
        # 初始化时即生成唯一名称
        self.__name__ = self._generate_name()

    # ======================================================
    # 唯一命名系统
    # ======================================================
    def _generate_name(self) -> str:
        """根据表达式生成安全且唯一的名称"""
        # 替换算子符号为下划线（保持可读性）
        sanitized = self._sanitize_expr(self.expr)

        # 添加哈希后缀以避免重复
        short_hash = self._short_hash(self.expr)
        name = f"Factor_{sanitized}_{short_hash}"

        # 截断过长名称（防止路径过长）
        if len(name) > 100:
            name = name[:90] + "_" + short_hash

        return name

    def _sanitize_expr(self, expr: str) -> str:
        """将表达式转换为文件安全格式"""
        expr = expr.lower()
        expr = re.sub(r"[^a-z0-9_]+", "_", expr)    # 非法字符换成_
        expr = re.sub(r"_+", "_", expr)             # 合并连续的下划线
        expr = expr.strip("_")                      # 去首尾下划线
        return expr

    def _short_hash(self, text: str, length: int = 6) -> str:
        """生成短哈希，用于区分相似表达式"""
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return h[:length]
 
    # --------------------------------------------------------
    # 核心入口
    # --------------------------------------------------------
    def process(self, X: pd.DataFrame) -> pd.Series:
        # 将所有字段注入命名空间
        local_env = {col: X[col] for col in X.columns}

        # 注册常用函数
        local_env.update({
            "roc": self._roc,
            "ma": self._ma,
            "zscore": self._zscore,
            "sigmoid": self._sigmoid,
            "log": np.log,
            "abs": np.abs,
            "np": np,
        })

        # 支持参数化表达式中的变量
        local_env.update(self.params)

        # 分组计算（例如按股票计算时间序列）
        grouped = X.groupby(self.group_by)

        # 对每组计算表达式
        results = []
        for name, group in grouped:
            group_result = self._safe_eval(group, local_env)
            results.append(group_result)

        factor = pd.concat(results).sort_index()

        # 横截面标准化（可选）
        if self.normalize_by is not None:
            factor = factor.groupby(self.normalize_by).transform(self._zscore)

        return factor.reindex(X.index)

    # --------------------------------------------------------
    # 内部函数注册
    # --------------------------------------------------------
    def _zscore_cross_sectional(self, x: pd.Series, error: float=1e-6) -> pd.Series:
        """对同一时点的截面进行标准化"""
        return (x - x.mean()) / (x.std(ddof=0) + error)
    
    def _zscore_rolling(self, x: pd.Series, window: int=252, error: float=1e-6) -> pd.Series:
        """对单支股票按时间序列做滚动标准化"""
        mean = x.rolling(window, min_periods=10).mean()
        std = x.rolling(window, min_periods=10).std(ddof=0)
        return (x - mean) / (std + error)
    
    def _zscore(self, x: pd.Series, mode: Literal['ts', 'cs']='ts', window: int=252, error: float=1e-6) -> pd.Series:
        if mode == 'cs':
            return self._zscore_cross_sectional(x, error)
        elif mode == 'ts':
            return self._zscore_rolling(x, window, error)
        else:
            raise ValueError("mode must be 'cs' or 'ts'")
    
    def _roc(self, series: pd.Series, N: int) -> pd.Series:
        return series / series.shift(N) - 1

    def _ma(self, series: pd.Series, N: int) -> pd.Series:
        return series.rolling(N, min_periods=1).mean()

    def _sigmoid(self, x: pd.Series) -> pd.Series:
        return 1 / (1 + np.exp(-x))

    def _safe_eval(self, group: pd.DataFrame, env: dict) -> pd.Series:
        """安全地按组计算表达式"""
        try:
            # 使用 eval + pandas 支持（比 numexpr 更灵活）
            result = pd.eval(self.expr, local_dict={**env, **group.to_dict("series")}, engine="python")
            result.index = group.index
            return result
        except Exception as e:
            print(f"[EvalError] {e} in group {group[self.group_by].iloc[0]}")
            return pd.Series(np.nan, index=group.index)


if __name__ == "__main__":
    """
    - cd qlib
    - python -m qlib.contrib.metric.base
    """
    import qlib
    from qlib.utils import init_instance_by_config
    
    start_time = "2024-01-01"
    end_time = "2024-12-31"
    
    data_handler_config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": end_time,
        "fields": ["open", "close", "amount", "rzmre", "rzye"],
        "instruments": [f"00000{i}.SZ" for i in [1, 2, 6, 8, 9]],
    }

    task = {
        "model": {
            "class": "ExpressionFactor",
            "module_path": "qlib.contrib.metric",
            # "kwargs": {
            #     "expr": "zscore(roc(rzmre, N)) * sigmoid(-k * roc(close, 1))",
            #     "params": {"N": 5, "k": 4.5},
            #     "group_by": "instrument",
            #     "normalize_by": "datetime",
            # },
            "kwargs": {
                "expr": "zscore(roc(close, N))",
                "params": {"N": 1},          
            }
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "OrdinaryDataHandler",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "test": (start_time, end_time),
                },
            },
        },
    }
    
    # Qlib initialization
    qlib.init(provider_uri="../temp/qlib_data")
    
    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to test model
    ds = dataset.prepare('test')
    ds.to_csv("../temp/ds_origin.csv")
    model.process(ds).to_csv("../temp/ds_processed.csv")
    print(model.__name__)
