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
