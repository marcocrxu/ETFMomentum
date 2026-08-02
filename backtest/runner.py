"""回测引擎 — 调用共享 signal 函数"""
import numpy as np
import pandas as pd
from typing import Dict, List

from strategy.signal import SignalConfig, compute_signals, compute_returns
from data.feed import TushareFeed


class BacktestRunner:
    def __init__(self, cfg: SignalConfig, feed: TushareFeed,
                 name_to_code: Dict[str, str]):
        self.cfg = cfg
        self.feed = feed
        self.name_to_code = name_to_code

    def run(self, start: str, end: str) -> dict:
        etf_names = list(self.name_to_code.keys())

        # ── 加载 ETF 价格 ──
        codes = list(self.name_to_code.values()) + ["000001.SH"]
        dfs = self.feed.get_dataframe(codes, start, end)
        all_dates = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
        all_dates = [d for d in all_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        n = len(all_dates)

        prices = {}
        for name, code in self.name_to_code.items():
            s = dfs[code].set_index("trade_date")["close"]
            prices[name] = np.array([s.get(d, np.nan) for d in all_dates])
        sh_s = dfs["000001.SH"].set_index("trade_date")["close"]
        sh_price = np.array([sh_s.get(d, np.nan) for d in all_dates])

        price_mat = np.array([prices[n] for n in etf_names])
        w = 20

        # ── 动量 ──
        def momentum(arr, window):
            m = np.full(n, np.nan)
            m[window:] = (arr[window:] / arr[:-window] - 1) * 100
            return m

        mom20_mat = np.array([momentum(prices[n], 20) for n in etf_names])
        mom10_mat = np.array([momentum(prices[n], 10) for n in etf_names])
        sh_mom = momentum(sh_price, 20)

        # ── 恐慌数据 (399006 + 000001) ──
        extra = self.feed.get_dataframe(["399006.SZ", "000001.SH"], start, end)
        cyb_p = sh_p = np.full(n, np.nan)
        if "399006.SZ" in extra:
            s = extra["399006.SZ"].set_index("trade_date")["close"]
            cyb_p = np.array([s.get(d, np.nan) for d in all_dates])
        if "000001.SH" in extra:
            s = extra["000001.SH"].set_index("trade_date")["close"]
            sh_p = np.array([s.get(d, np.nan) for d in all_dates])

        # ── 计算信号 ──
        actions, final_nav, nav_log_arr = compute_signals(
            self.cfg, etf_names, price_mat, mom20_mat, mom10_mat,
            sh_mom, cyb_p, sh_p, n
        )

        # ── 计算收益（用信号内建的nav_log） ──
        result = compute_returns(actions, price_mat, etf_names, n)
        result["total_ret"] = round((final_nav - 1) * 100, 2)
        # 使用 compute_signals 的 nav_log（含手续费，准确）
        result["nav_log"] = nav_log_arr
        nav_series = np.array(nav_log_arr)
        cm = np.maximum.accumulate(nav_series)
        dd = (nav_series - cm) / cm * 100
        result["max_dd"] = round(float(np.min(dd)), 2) if len(dd) > 0 else 0

        # ── 年度收益 ──
        nav_arr = np.array(result["nav_log"])
        # padding: vs 之前的日期用 1.0
        if len(nav_arr) < n:
            nav_arr = np.concatenate([np.ones(n - len(nav_arr)), nav_arr])
        yrs = {}
        for i, d in enumerate(all_dates):
            yrs[d.year] = nav_arr[i]
        sorted_yrs = sorted(yrs.keys())
        prev = 1.0
        yearly = {}
        for yr in sorted_yrs:
            ret = (yrs[yr] / prev - 1) * 100
            yearly[yr] = round(ret, 2)
            prev = yrs[yr]

        result["yearly"] = yearly
        result["neg_years"] = sum(1 for v in yearly.values() if v < 0)
        result["trade_count"] = len(result["trades"])
        return result
