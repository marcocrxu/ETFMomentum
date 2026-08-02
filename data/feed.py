"""统一数据层 — Tushare历史 + 腾讯实时 + 缓存管理"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class DataFeed(ABC):
    @abstractmethod
    def get_prices(self, codes: List[str], start: str, end: str) -> dict:
        """返回 {code: close_price_array}"""
        pass

    @abstractmethod
    def get_latest_date(self) -> str:
        """数据最新日期"""
        pass

    @abstractmethod
    def is_fresh(self, max_age_days: int = 2) -> bool:
        """数据是否新鲜"""
        pass


class TushareFeed(DataFeed):
    """Tushare 历史数据 + 本地缓存"""
    def __init__(self, token: str, cache_dir: str = "/root/etf_strategy/cache"):
        import tushare as ts
        self.pro = ts.pro_api(token)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_prices(self, codes: List[str], start: str, end: str,
                   adjusted: bool = True) -> Dict[str, np.ndarray]:
        start = start.replace("-", ""); end = end.replace("-", "")
        result = {}
        for code in codes:
            df = self._load_or_fetch(code, start, end, adjusted)
            if df is not None and len(df) > 0:
                result[code] = df.set_index("trade_date")["close"].values
        return result

    def get_dataframe(self, codes: List[str], start: str, end: str,
                      auto_refresh: bool = True) -> Dict[str, pd.DataFrame]:
        """返回完整 DataFrame。auto_refresh: 自动检测并刷新过期数据"""
        start = start.replace("-", ""); end = end.replace("-", "")
        result = {}
        need_refresh = False

        # 第一遍：检查缓存新鲜度
        if auto_refresh:
            for code in codes:
                df = self._load_from_cache(code, start, end)
                if df is None:
                    need_refresh = True
                    break
                cache_max = df["trade_date"].max()
                target_end = pd.Timestamp(end[:4] + "-" + end[4:6] + "-" + end[6:])
                if cache_max < target_end - pd.Timedelta(days=3):
                    need_refresh = True
                    break
                result[code] = df

        if need_refresh or not result:
            # 缓存不够新鲜，全部重新拉取
            for code in codes:
                df = self._load_or_fetch(code, start, end)
                if df is not None and len(df) > 0:
                    result[code] = df

        return result

    def _load_from_cache(self, code: str, start: str, end: str) -> pd.DataFrame:
        """仅从缓存读取，不拉取 Tushare"""
        code_clean = code.replace(".", "_")
        cf = os.path.join(self.cache_dir, "%s_%s_%s.csv" % (code_clean, start, end))
        if not os.path.exists(cf):
            import glob
            pattern = os.path.join(self.cache_dir, "%s_*.csv" % code_clean)
            candidates = sorted(glob.glob(pattern))
            if candidates:
                cf = candidates[-1]
        if os.path.exists(cf):
            df = pd.read_csv(cf, parse_dates=["trade_date"])
            if len(df) > 0:
                cache_max = df["trade_date"].max()
                cache_min = df["trade_date"].min()
                target_end = pd.Timestamp(end[:4] + "-" + end[4:6] + "-" + end[6:])
                target_start = pd.Timestamp(start[:4] + "-" + start[4:6] + "-" + start[6:])
                if cache_max >= target_end - pd.Timedelta(days=3) and cache_min <= target_start + pd.Timedelta(days=3):
                    return df
        return None

    def _load_or_fetch(self, code: str, start: str, end: str,
                       adjusted: bool = True) -> Optional[pd.DataFrame]:
        code_clean = code.replace(".", "_")
        # 尝试多种缓存文件名格式 (日期范围可能不完全匹配)
        cf = os.path.join(self.cache_dir, "%s_%s_%s.csv" % (code_clean, start, end))
        if not os.path.exists(cf):
            # 模糊匹配: 找同名代码的缓存文件
            import glob
            pattern = os.path.join(self.cache_dir, "%s_*.csv" % code_clean)
            candidates = sorted(glob.glob(pattern))
            if candidates:
                cf = candidates[-1]  # 取最新的
        if os.path.exists(cf):
            df = pd.read_csv(cf, parse_dates=["trade_date"])
            if len(df) > 0:
                cache_max = df["trade_date"].max()
                cache_min = df["trade_date"].min()
                target_end = pd.Timestamp(end[:4] + "-" + end[4:6] + "-" + end[6:])
                target_start = pd.Timestamp(start[:4] + "-" + start[4:6] + "-" + start[6:])
                # 缓存必须覆盖目标日期范围
                if cache_max >= target_end - pd.Timedelta(days=3) and cache_min <= target_start + pd.Timedelta(days=3):
                    return df
            # 缓存过期或范围不足，重新拉取
            print("[DataFeed] 缓存过期, 重新拉取 %s (缓存到 %s, 需要到 %s)" % (
                code, str(cache_max)[:10] if cache_max else "无数据", str(target_end)[:10]))
        # 缓存过期或不存在, 重新拉取

        # Fetch from Tushare: 依次尝试 fund_daily → index_daily → daily
        df_new = None
        try:
            if code.endswith("SH") or code.endswith("SZ"):
                # 1) 先尝试 fund_daily（ETF/基金，如 51xxxx, 159915, 513100 等）
                df_new = self.pro.fund_daily(ts_code=code, start_date=start, end_date=end)
                # 2) fund_daily 没数据则尝试 index_daily（指数，如 000001, 399006）
                if df_new is None or len(df_new) == 0:
                    df_new = self.pro.index_daily(ts_code=code, start_date=start, end_date=end)
                # 3) 都没有则尝试 daily（个股）
                if df_new is None or len(df_new) == 0:
                    df_new = self.pro.daily(ts_code=code, start_date=start, end_date=end)
            if df_new is not None and len(df_new) > 0:
                df_new["trade_date"] = pd.to_datetime(df_new["trade_date"], format="%Y%m%d")
                df_new = df_new.sort_values("trade_date").reset_index(drop=True)
                if adjusted and not code.startswith("0"):
                    try:
                        adj = self.pro.fund_adj(ts_code=code, start_date=start, end_date=end)
                        if adj is not None and len(adj) > 0:
                            adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")
                            df_new = df_new.merge(adj, on="trade_date", how="left")
                            df_new["adj_factor"] = df_new["adj_factor"].fillna(1.0)
                            df_new["close"] = df_new["close"] * df_new["adj_factor"] / df_new["adj_factor"].iloc[-1]
                    except Exception: pass
                df_new.to_csv(cf, index=False)
                return df_new
        except Exception as e:
            print("[DataFeed] %s 获取失败: %s" % (code, e))

        # Fallback: return stale cached data if available
        if os.path.exists(cf):
            print("[DataFeed] %s 使用旧缓存" % code)
            return pd.read_csv(cf, parse_dates=["trade_date"])
        return None

    def get_latest_date(self) -> str:
        return datetime.now().strftime("%Y%m%d")

    def is_fresh(self, max_age_days: int = 2) -> bool:
        # 检查缓存中最新的数据日期
        return True  # TODO: 实现数据新鲜度检查


class LiveFeed(DataFeed):
    """腾讯实时行情 + Tushare历史"""
    def __init__(self, tushare_feed: TushareFeed):
        self.tushare = tushare_feed
        self._live_prices: Dict[str, float] = {}

    def get_prices(self, codes: List[str], start: str, end: str) -> dict:
        return self.tushare.get_prices(codes, start, end)

    def get_live_prices(self, tencent_codes: Dict[str, str]) -> Dict[str, float]:
        """tencent_codes: {name: s_sz159915}"""
        import requests
        url = "http://qt.gtimg.cn/q=" + ",".join(tencent_codes.values())
        try:
            resp = requests.get(url, timeout=5)
            resp.encoding = "gbk"
        except Exception:
            return {}

        name_map = {}
        for name, tc in tencent_codes.items():
            name_map[tc] = name
        # Add more name matches
        tc_to_asset = {
            "创业板ETF易方达": "创业", "纳指ETF": "纳指", "纳指ETF国泰": "纳指",
            "黄金ETF华安": "黄金", "上证指数": "上证",
        }

        result = {}
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("~")
            if len(parts) < 10:
                continue
            tc_name = parts[1]
            tag = tc_to_asset.get(tc_name, "")
            if tag:
                try:
                    result[tag] = float(parts[3])
                except ValueError:
                    pass
        return result

    def get_latest_date(self) -> str:
        return datetime.now().strftime("%Y%m%d")

    def is_fresh(self, max_age_days: int = 2) -> bool:
        return True  # Live is always fresh
