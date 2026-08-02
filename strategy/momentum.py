"""动量计算方法"""
import numpy as np
from abc import ABC, abstractmethod


class MomentumMethod(ABC):
    @abstractmethod
    def compute(self, prices: np.ndarray) -> np.ndarray:
        """输入价格序列, 返回动量序列 (长度相同, 前window天为NaN)"""
        pass


class PctMomentum(MomentumMethod):
    """传统价差: p[t] / p[t-window] - 1"""
    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: np.ndarray) -> np.ndarray:
        result = np.full(len(prices), np.nan)
        if len(prices) > self.window:
            result[self.window:] = (prices[self.window:] / prices[:-self.window] - 1) * 100
        return result


class OlsMomentum(MomentumMethod):
    """OLS斜率: 对最近window个点做最小二乘, 斜率/均价 * 100"""
    def __init__(self, window: int = 20):
        self.window = window
        self._x = np.arange(window)

    def compute(self, prices: np.ndarray) -> np.ndarray:
        result = np.full(len(prices), np.nan)
        for i in range(self.window, len(prices)):
            y = prices[i - self.window:i]
            if np.any(np.isnan(y)):
                continue
            slope = np.polyfit(self._x, y, 1)[0]
            result[i] = slope / np.mean(y) * 100  # daily % return
        return result


class EwmaMomentum(MomentumMethod):
    """EWMA z-score: (当前值 - EWMA均值) / EWMA标准差"""
    def __init__(self, window: int = 20, halflife: int = 252):
        self.window = window
        self.halflife = halflife

    def compute(self, prices: np.ndarray) -> np.ndarray:
        import pandas as pd
        raw_mom = PctMomentum(self.window).compute(prices)
        s = pd.Series(raw_mom)
        ewm_mean = s.ewm(halflife=self.halflife, min_periods=252).mean()
        ewm_var = s.ewm(halflife=self.halflife, min_periods=252).var()
        ewm_std = np.sqrt(ewm_var)
        z = (raw_mom - ewm_mean.values) / ewm_std.values
        return z
