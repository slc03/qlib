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


class Factor_FPC_ROC_Limit(MetricBaseModel):
    """融资买入额变化率 (Financing Purchase Rate of Change)"""
    def __init__(self, N: int = 5, limit: float = 1.5):
        self.N = N
        self.limit = limit

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """ 
        计算N周期融资买入额的变化率，并剔除异常值。
        因子逻辑: 衡量短期内融资买入额的增长速度。
        """
        factor = (
            X_test.groupby("instrument")["rzmre"]
            .apply(lambda x: x / x.shift(self.N) - 1)
        ).droplevel(0)

        # 剔除变化率大于 limit 的部分
        factor[factor > self.limit] = np.nan

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


# ============================================================
# 资金流与存量类因子 (Flow & Balance Factors)
# ============================================================

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


# ============================================================
# 其它综合类因子 (Other Composite Factors)
# ============================================================    

class Factor_Financing_Participation_ROC(MetricBaseModel): 
    """融资参与度变化率因子""" 
    def __init__(self, N: int = 5): 
        self.N = N 

    def process(self, X_test: pd.DataFrame) -> pd.Series: 
        """ 
        因子逻辑: 衡量融资买入额占总成交额比例的变化速度。 
        该比例上升说明融资盘交易活跃度/主导性增强。 
        """ 
        # 防止成交额为0导致计算错误
        X_test['amount'] = X_test['amount'].replace(0, np.nan)
        # 计算每日的融资参与度
        X_test["fin_participation"] = X_test["rzmre"] / X_test["amount"]
        # 对每个instrument分组，计算参与度的变化率
        factor = (
            X_test.groupby("instrument")["fin_participation"]
            .apply(lambda x: x / x.shift(self.N) - 1)
        ).droplevel(0)
        return factor.reindex(X_test.index)
    
    
class Factor_Standardized_Financing_Turnover_Divergence(MetricBaseModel):
    """标准化融资成交背离因子"""
    def __init__(self, N: int = 5):
        self.N = N

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """
        因子逻辑: 衡量融资买入增速与市场整体成交增速的相对强度。
        计算方式: 对二者的变化率进行横截面标准化后求差。
        """
        # 计算融资买入额和成交额的变化率
        roc_rzmre = X_test.groupby("instrument")["rzmre"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_rzmre")
        roc_amount = X_test.groupby("instrument")["amount"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_amount")

        # 合并到一个DataFrame中，方便按日期进行横截面标准化
        combined = pd.concat([roc_rzmre, roc_amount], axis=1).droplevel(0)
        combined_with_date = combined.reindex(X_test.index)

        # 进行横截面标准化
        z_roc_rzmre = combined_with_date.groupby("datetime")["roc_rzmre"].transform(self._zscore_cs)
        z_roc_amount = combined_with_date.groupby("datetime")["roc_amount"].transform(self._zscore_cs)

        # 计算因子值
        factor_values = z_roc_rzmre - z_roc_amount
        factor_values.index = combined_with_date.index # 保持索引一致
        
        # 恢复原始索引并返回
        return factor_values.reindex(X_test.index)
    

class Factor_Financing_Price_Divergence_CS(MetricBaseModel):
    """标准化融资价格背离因子 (Cross-Sectional)"""
    def __init__(self, N: int = 5):
        self.N = N

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """
        因子逻辑: 衡量融资价格背离信号在全市场中的相对强度。
        计算方式: (N周期融资买入额ROC - N周期收盘价ROC) -> 每日横截面Z-Score标准化。
        """
        # 计算融资买入额和股票收盘价的变化率
        roc_rzmre = X_test.groupby("instrument")["rzmre"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_rzmre")
        roc_close = X_test.groupby("instrument")["close"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_close")

        # 合并到一个DataFrame中，方便按日期进行横截面标准化
        combined = pd.concat([roc_rzmre, roc_close], axis=1).droplevel(0)
        combined_with_date = combined.reindex(X_test.index)

        # 进行横截面标准化
        z_roc_rzmre = combined_with_date.groupby("datetime")["roc_rzmre"].transform(self._zscore_cs)
        z_roc_close = combined_with_date.groupby("datetime")["roc_close"].transform(self._zscore_cs)

        # 计算因子值
        factor_values = z_roc_rzmre - z_roc_close
        factor_values.index = combined_with_date.index # 保持索引一致
        
        # 恢复原始索引并返回
        return factor_values.reindex(X_test.index)
    
    
class Factor_Stealth_Accumulation_CS(MetricBaseModel):
    """标准化融资吸筹强度因子 (Cross-Sectional)"""
    def __init__(self, N: int = 5, epsilon: float = 1e-4):
        self.N = N
        self.epsilon = epsilon

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """
        因子逻辑: 衡量静默吸筹行为在全市场中的相对强度。
        计算方式: (N周期融资买入额ROC / |N周期收盘价ROC|) -> 每日横截面Z-Score标准化。
        """
        # 计算融资买入额和股票收盘价的变化率
        roc_rzmre = X_test.groupby("instrument")["rzmre"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_rzmre")
        roc_close = X_test.groupby("instrument")["close"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_close")

        # 合并到一个DataFrame中，方便按日期进行横截面标准化
        combined = pd.concat([roc_rzmre, roc_close], axis=1).droplevel(0)
        combined_with_date = combined.reindex(X_test.index)

        # 进行横截面标准化
        z_roc_rzmre = combined_with_date.groupby("datetime")["roc_rzmre"].transform(self._zscore_cs)
        z_roc_close = combined_with_date.groupby("datetime")["roc_close"].transform(self._zscore_cs)

        # 计算因子值
        factor_values = (z_roc_rzmre / (abs(z_roc_close) + self.epsilon))
        factor_values.index = combined_with_date.index # 保持索引一致
        
        # 恢复原始索引并返回
        return factor_values.reindex(X_test.index)
    

class Factor_Stealth_Accumulation_CS_Plus(MetricBaseModel):
    """标准化融资吸筹强度因子+ (Cross-Sectional)【增加了对涨停带来的突变的过滤】"""
    def __init__(self, N: int = 5, epsilon: float = 1e-4):
        self.N = N
        self.epsilon = epsilon

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        """
        因子逻辑: 衡量静默吸筹行为在全市场中的相对强度。
        计算方式: (N周期融资买入额ROC / |N周期收盘价ROC|) -> 每日横截面Z-Score标准化。
        """
        # 计算融资买入额和股票收盘价的变化率
        roc_rzmre = X_test.groupby("instrument")["rzmre"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_rzmre")
        roc_close = X_test.groupby("instrument")["close"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_close")

        # 合并到一个DataFrame中，方便按日期进行横截面标准化
        combined = pd.concat([roc_rzmre, roc_close], axis=1).droplevel(0)
        combined_with_date = combined.reindex(X_test.index)

        # 进行横截面标准化
        z_roc_rzmre = combined_with_date.groupby("datetime")["roc_rzmre"].transform(self._zscore_cs)
        z_roc_close = combined_with_date.groupby("datetime")["roc_close"].transform(self._zscore_cs)

        # 计算因子值
        factor_values = (z_roc_rzmre / (abs(z_roc_close) + self.epsilon))
        factor_values.index = combined_with_date.index # 保持索引一致
        
        # 计算涨停过滤条件（昨日涨幅 > 9.5%）
        yesterday_up = X_test.groupby("instrument")["close"].apply(
            lambda x: x.shift(1) / x.shift(2) - 1
        ).rename("roc_pre_close").droplevel(0).reindex(factor_values.index)

        # 过滤掉涨停导致的融资异常（设为0或设为NaN）
        factor_values[yesterday_up > 0.095] = np.nan
        
        # 恢复原始索引并返回
        return factor_values.reindex(X_test.index)


class Factor_Financing_Momentum_Weighted(MetricBaseModel):
    """
    V3 = ZScore(Δ融资买入额) × Sigmoid(-k · 当日收益率)
    逻辑：在融资流入显著增加时，如果股价下跌，赋予更高权重 → 捕捉“逆势吸筹/资金抄底”行为。
    """
    def __init__(self, N: float = 5, k: float = 5):
        self.N = N
        self.k = k

    def process(self, X_test: pd.DataFrame) -> pd.Series:
        # 计算融资买入额和股票收盘价的变化率
        roc_rzmre = X_test.groupby("instrument")["rzmre"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_rzmre")
        roc_close = X_test.groupby("instrument")["close"].apply(
            lambda x: x / x.shift(self.N) - 1
        ).rename("roc_close")

        # 合并到一个DataFrame中，方便按日期进行横截面标准化
        combined = pd.concat([roc_rzmre, roc_close], axis=1).droplevel(0)
        combined_with_date = combined.reindex(X_test.index)

        # 进行横截面标准化
        z_roc_rzmre = combined_with_date.groupby("datetime")["roc_rzmre"].transform(self._zscore_cs)
        z_roc_close = combined_with_date.groupby("datetime")["roc_close"].transform(lambda x: x)

        # 计算因子值
        factor_values = z_roc_rzmre * 1 / (1 + np.exp(self.k * z_roc_close))
        factor_values.index = combined_with_date.index # 保持索引一致
        
        # 恢复原始索引并返回
        return factor_values.reindex(X_test.index)
