import numpy as np
import pandas as pd
from stockstats import wrap

from .base import MetricBaseModel


# ============================================================
# 日内做T因子 (Intraday T Factors)
# ============================================================

class Factor_KDJJ(MetricBaseModel):
    """根据KDJ的J线判断大行情，再在日内做T"""
    def __init__(self, buy_threshold: int=20, sell_threshold: int=70, delta: float=0.005, pull_down: float=0.015, 
                 key: str='kdjj', weight: float=1/3):
        self.buy_threshold = buy_threshold          # 超卖买入阈值
        self.sell_threshold = sell_threshold        # 超买卖出阈值
        self.delta = delta                          # 价格从极值反弹幅度
        self.pulldown = pull_down                   # 高于开盘价幅度
        self.key = key                              # 指标名称
        self.weight = weight                        # 每次开仓的权重
        
    def _compute_daily_metric(self, key: str, X: pd.DataFrame) -> pd.Series:
        """利用 stockstats 计算日线指标"""
        temp_df_day = wrap(X)
        temp_df_day[key]
        return temp_df_day
    
    def _extract_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """从合并的数据中恢复出日线数据"""
        df_grouped = df.groupby("DailyKey").agg({
                        "open_daily": "first",
                        "high_daily": "first",
                        "low_daily": "first",
                        "close_daily": "first",
                    }).rename(columns={
                        "open_daily": "open",
                        "high_daily": "high",
                        "low_daily": "low",
                        "close_daily": "close",
                    }).reset_index()
        df_grouped['DailyKey'] = df_grouped['DailyKey'].astype(int).astype(str)
        df_grouped['DailyKey'] = pd.to_datetime(df_grouped['DailyKey'], format="%y%m%d")
        df_grouped.set_index("DailyKey", inplace=True)
        df_grouped.index.name = "datetime"
        return df_grouped
    
    def _get_signal(self, X: pd.DataFrame) -> pd.DataFrame:
        """从单索引数据中计算得到信号"""
        # 首先计算KDJJ信号
        df_daily = self._extract_daily(X)
        df_daily = self._compute_daily_metric(self.key, df_daily)
        X['DailyKey'] = pd.to_datetime(X['DailyKey'], format="%y%m%d")
        df_merge = pd.merge(X, df_daily[[self.key]], left_on='DailyKey', right_index=True, how='left')
        
        # 接着计算买入和卖出的信号
        df_merge['signal'] = 0.0
        time_str = df_merge.index.strftime('%H:%M')
        df_merge.loc[(df_merge['kdjj'] > self.sell_threshold) & (time_str == '09:30'), 'signal'] = self.weight
        df_merge.loc[(df_merge['kdjj'] < self.buy_threshold) & (time_str == '09:30'), 'signal'] = -self.weight

        return df_merge

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算N周期融资买入额的变化率。 
        因子逻辑: 衡量短期内融资买入额的增长速度。 
        """
        # 检查 instrument 是否唯一
        if X_test.index.get_level_values("instrument").unique().size > 1:
            raise ValueError("暂时仅支持单个股票日内做T！")

        # 删除 instrument 这个索引，只保留 datetime
        instrument_values = X_test.index.get_level_values("instrument")
        df_reset = X_test.droplevel("instrument")

        # ---- 这里写你要执行的函数，比如 ----
        df_reset = self._get_signal(df_reset)

        # 恢复 instrument 索引
        df_final = df_reset.set_index(instrument_values, append=True)
        df_final = df_final.reorder_levels(["datetime", "instrument"])
        
        return df_final['signal']


class Simple_DoT(MetricBaseModel):
    """极简做T模型，每天选择一个时间点（买入/卖出），然后尾盘接回"""
    def __init__(self, sell_time: str = "11:30:00", buy_time: str = "14:57:00"):
        super().__init__()
        self.sell_time = sell_time
        self.buy_time = buy_time

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """
        根据时间生成买卖信号：
        - 每天13:01:00卖出(-1)
        - 每天14:59:00买入(1)
        - 其他时间为0
        """
        # 复制索引
        idx = X_test.index

        # 取 datetime 层
        datetimes = idx.get_level_values('datetime')

        # 提取时间部分字符串
        time_str = datetimes.strftime("%H:%M:%S")

        # 创建信号序列
        signal = pd.Series(0, index=idx, dtype=int)

        # 设置买入/卖出信号
        signal[time_str == self.buy_time] = 1
        signal[time_str == self.sell_time] = -1

        return signal


if __name__ == "__main__":
    """
    - cd qlib
    - python -m qlib.contrib.metric.dot
    """
    import qlib
    from qlib.utils import init_instance_by_config
    
    start_time = "2024-10-18"
    end_time = "2025-09-18"
    freq = "1min"
    benchmark = "000048.SZ"
    
    data_handler_config = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": end_time,
        "fields": ["close", "open_daily", "high_daily", "low_daily", "close_daily", "DailyKey"],
        "instruments": "all",
        "freq": freq,
    }

    task = {
        "model": {
            "class": "Simple_DoT",
            "module_path": "qlib.contrib.metric",
            # "kwargs": {
            #     "xxx": "xxx",
            # },
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
    qlib.init(provider_uri="../temp/qlib_data_minute")
    
    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to test model
    ds = dataset.prepare('test')
    ds.to_csv("../temp/ds_origin.csv")
    model.process(ds).to_csv("../temp/ds_processed.csv")
