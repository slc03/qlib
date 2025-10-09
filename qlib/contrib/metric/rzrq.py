import numpy as np
import pandas as pd

from .base import MetricBaseModel


# ============================================================
# 动量类因子 (Momentum Factors)
# ============================================================

class Factor_FPC_ROC(MetricBaseModel):
    """融资买入额变化率 (Financing Purchase Rate of Change)"""
    def __init__(self, N: int = 5):
        self.N = N

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算N周期融资买入额的变化率。 
        因子逻辑: 衡量短期内融资买入额的增长速度。 
        """
        factor = (
            X_test.groupby("instrument")["rzmre"]
            .apply(lambda x: x / x.shift(self.N) - 1)
        ).droplevel(0)
        return factor.reindex(X_test.index)


class Factor_FPC_MA_Ratio(MetricBaseModel):
    """融资买入额均线比率 (Financing Purchase Moving Average Ratio)"""
    def __init__(self, S: int = 5, L: int = 20):
        self.S = S
        self.L = L

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算融资买入额的短周期(S)与长周期(L)均线比率。 
        因子逻辑: 经典“金叉/死叉”思路，捕捉融资情绪的趋势变化。 
        """
        def calc_ma_ratio(x: pd.Series):
            sma_s = x.rolling(self.S, min_periods=1).mean()
            sma_l = x.rolling(self.L, min_periods=1).mean()
            return sma_s / sma_l.replace(0, np.nan)

        factor = (
            X_test.groupby("instrument")["rzmre"]
            .apply(calc_ma_ratio)
        ).droplevel(0)
        return factor.reindex(X_test.index)


class Factor_FPC_Breakout(MetricBaseModel):
    """融资买入额N日突破 (Financing Purchase N-day High Breakout)"""
    def __init__(self, N: int = 20):
        self.N = N

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算当日融资买入额相对于过去N日最高值的比率。 
        因子逻辑: 衡量融资买入额是否创下近期新高。大于1表示突破。 
        """
        def calc_breakout(x: pd.Series):
            past_max = x.shift(1).rolling(self.N, min_periods=1).max()
            return x / past_max.replace(0, np.nan)

        factor = (
            X_test.groupby("instrument")["rzmre"]
            .apply(calc_breakout)
        ).droplevel(0)
        return factor.reindex(X_test.index)

# ============================================================
# 相对强度类因子 (Relative Strength Factors)
# ============================================================

class Factor_FPC_to_Volume(MetricBaseModel):
    """融资买入占成交额比 (Financing Purchase to Trade Volume Ratio)"""
    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算融资买入额占当日总成交额的比例。 
        因子逻辑: 核心指标，衡量融资交易者在当天交易中的参与度。 
        """
        denominator = X_test["amount"].replace(0, np.nan)
        factor = X_test["rzmre"] / denominator
        return factor.reindex(X_test.index)


class Factor_FPC_to_MCap(MetricBaseModel):
    """融资买入占流通市值比 (Financing Purchase to Market Cap Ratio)"""
    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算融资买入额占当日流通市值的比例。 
        因子逻辑: 衡量新增融资买入资金相对于股票规模的大小。 
        """
        denominator = X_test["circ_mv"].replace(0, np.nan)
        factor = X_test["rzmre"] / denominator
        return factor.reindex(X_test.index)

# ============================================================
# 资金流与存量类因子 (Flow & Balance Factors)
# ============================================================

class Factor_NetFinBuy_Norm(MetricBaseModel):
    """融资净买入额标准化 (Net Financing Purchase Normalized by Volume)"""
    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算融资净买入额，并用成交额进行标准化。 
        因子逻辑: 衡量当天融资做多与做空力量的净方向和强度。 
        """
        net_buy = X_test["rzmre"] - X_test["rzche"]
        denominator = X_test["amount"].replace(0, np.nan)
        factor = net_buy / denominator
        return factor.reindex(X_test.index)


class Factor_FPI(MetricBaseModel):
    """融资买入强度 (Financing Purchase Intensity, [-1, 1]归一化)"""
    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算融资买入强度，归一化到[-1, 1]区间。 
        因子逻辑: 衡量融资盘多空力量的对比强度，排除总量影响。 
        """
        numerator = X_test["rzmre"] - X_test["rzche"]
        denominator = X_test["rzmre"] + X_test["rzche"]
        factor = numerator / denominator.replace(0, np.nan)
        return factor.reindex(X_test.index)


class Factor_FBC(MetricBaseModel):
    """融资余额增长贡献 (Contribution to Financing Balance Growth)"""
    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算当日融资买入额相对于前一日融资余额的比例。 
        因子逻辑: 衡量新增买入额相对于存量融资盘的大小。 
        """
        def calc_fbc(df_group: pd.DataFrame):
            prev_balance = df_group["rzye"].shift(1)
            return df_group["rzmre"] / prev_balance.replace(0, np.nan)

        factor = (
            X_test.groupby("instrument")[["rzmre", "rzye"]]
            .apply(calc_fbc)
        ).droplevel(0)
        return factor.reindex(X_test.index)
