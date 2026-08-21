"""统一数据层 — Tushare历史 + 腾讯实时 + 缓存管理"""
import os
import threading
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
    """Tushare 历史数据 + 本地增量缓存

    缓存设计:
    - 每个代码一个规范文件 {code}.csv, 存原始 OHLCV + adj_factor (复权因子)
    - 增量更新: 只拉取缺口日期 (cache_max+1 → end), 合并去重后原子写回
    - 复权价格在读取时现场计算: close_adj = close_raw × adj_factor / 最新adj_factor
      (因子更新时历史价格自动整体重算, 无拼接缝)
    - 旧格式 ({code}_{start}_{end}.csv) 自动迁移清理
    """
    def __init__(self, token: str, cache_dir: str = "/root/etf_strategy/cache"):
        import tushare as ts
        self.pro = ts.pro_api(token)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._migrate_legacy()
        # 每代码一把锁: 并发请求对同一代码的 读-改-写 串行化, 防止 .tmp 竞争
        self._locks = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, path: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(path)
            if lk is None:
                lk = self._locks[path] = threading.Lock()
            return lk

    # ── 缓存文件管理 ──
    def _canonical_path(self, code: str) -> str:
        return os.path.join(self.cache_dir, code.replace(".", "_") + ".csv")

    def _migrate_legacy(self):
        """规范文件已存在的代码, 删除旧日期范围格式的缓存文件"""
        import glob, re
        for f in glob.glob(os.path.join(self.cache_dir, "*.csv")):
            base = os.path.basename(f)
            m = re.match(r"^(.+)_\d{8}_\d{8}\.csv$", base)
            if m and os.path.exists(os.path.join(self.cache_dir, m.group(1) + ".csv")):
                try: os.remove(f)
                except OSError: pass

    def _read_cache(self, code: str) -> Optional[pd.DataFrame]:
        p = self._canonical_path(code)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, parse_dates=["trade_date"])
                if len(df) > 0:
                    return df
            except Exception:
                pass
        return None

    def _atomic_write(self, path: str, df: pd.DataFrame):
        # 临时文件名唯一化: 固定 ".tmp" 会让并发写同一代码的两个请求互相吃掉
        # 对方的临时文件 (os.replace 找不到源 → FileNotFoundError, 2026-08-21 实锤)
        import uuid
        tmp = "%s.tmp.%s.%s.%s" % (path, os.getpid(), threading.get_ident(), uuid.uuid4().hex[:8])
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass

    # ── 远程拉取 ──
    def _fetch_remote(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """拉取 [start, end] 原始行情 + 复权因子"""
        df_new = None
        try:
            if code.endswith("SH") or code.endswith("SZ"):
                df_new = self.pro.fund_daily(ts_code=code, start_date=start, end_date=end)
                if df_new is None or len(df_new) == 0:
                    df_new = self.pro.index_daily(ts_code=code, start_date=start, end_date=end)
                if df_new is None or len(df_new) == 0:
                    df_new = self.pro.daily(ts_code=code, start_date=start, end_date=end)
        except Exception as e:
            print("[DataFeed] %s 行情获取失败: %s" % (code, e))
            return None
        if df_new is None or len(df_new) == 0:
            return None

        df_new["trade_date"] = pd.to_datetime(df_new["trade_date"], format="%Y%m%d")
        df_new = df_new.sort_values("trade_date").reset_index(drop=True)

        # 复权因子: 缺失留 NaN, 由 _load_or_fetch 合并后 ffill 前值
        # (此前缺口行填 1.0 会让 latest_adj 失真, 整段历史被等比放大出现假接缝)
        if not code.startswith("0"):
            try:
                adj = self.pro.fund_adj(ts_code=code, start_date=start, end_date=end)
                if adj is not None and len(adj) > 0:
                    adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")
                    df_new = df_new.merge(adj[["trade_date", "adj_factor"]],
                                          on="trade_date", how="left")
            except Exception:
                pass
        if "adj_factor" not in df_new.columns:
            df_new["adj_factor"] = np.nan
        return df_new

    # ── 主加载逻辑: 增量更新 ──
    def _load_or_fetch(self, code: str, start: str, end: str,
                       adjusted: bool = True) -> Optional[pd.DataFrame]:
        with self._lock_for(self._canonical_path(code)):
            return self._load_or_fetch_impl(code, start, end, adjusted)

    def _load_or_fetch_impl(self, code: str, start: str, end: str,
                            adjusted: bool = True) -> Optional[pd.DataFrame]:
        start_ts = pd.Timestamp(start[:4] + "-" + start[4:6] + "-" + start[6:])
        end_ts = pd.Timestamp(end[:4] + "-" + end[4:6] + "-" + end[6:])

        df = self._read_cache(code)
        fetch_start = None
        if df is None:
            fetch_start = start                     # 无缓存: 全量拉取
        else:
            cache_max = df["trade_date"].max()
            if cache_max < end_ts:
                # 带4天重叠回补: 让上次临时补的因子/价格被官方数据覆盖修正
                fetch_start = (cache_max - pd.Timedelta(days=4)).strftime("%Y%m%d")

        # ── 头部回补: 请求起点早于缓存最早日期时向前拉取 ──
        # (此前只向后增量, 早于 cache_min 的区间被静默截断,
        #   导致 MA200/长预热回测在旧缓存上静默失效)
        if df is not None:
            cache_min = df["trade_date"].min()
            if cache_min > start_ts:
                head_end = min(cache_min, end_ts)
                df_head = self._fetch_remote(code, start, head_end.strftime("%Y%m%d"))
                if df_head is not None and len(df_head) > 0:
                    df = pd.concat([df, df_head], ignore_index=True)
                    df = (df.drop_duplicates("trade_date", keep="last")
                            .sort_values("trade_date").reset_index(drop=True))
                    df["adj_factor"] = df["adj_factor"].ffill().bfill().fillna(1.0)
                    self._atomic_write(self._canonical_path(code), df)
                    print("[DataFeed] %s 头部回补 %s→%s (+%d行, 共%d行)" % (
                        code, start, head_end.strftime("%Y%m%d"),
                        len(df_head), len(df)))

        if fetch_start is not None:
            df_new = self._fetch_remote(code, fetch_start, end)
            if df_new is not None and len(df_new) > 0:
                if df is not None:
                    df = pd.concat([df, df_new], ignore_index=True)
                    df = (df.drop_duplicates("trade_date", keep="last")
                            .sort_values("trade_date").reset_index(drop=True))
                    print("[DataFeed] %s 增量更新 %s→%s (+%d行, 共%d行)" % (
                        code, fetch_start, end, len(df_new), len(df)))
                else:
                    df = df_new
                    print("[DataFeed] %s 全量拉取 %s→%s (%d行)" % (code, fetch_start, end, len(df)))
                # 因子缺口继承前值 (指数全 NaN 则统一 1.0)
                df["adj_factor"] = df["adj_factor"].ffill().bfill().fillna(1.0)
                self._atomic_write(self._canonical_path(code), df)
                self._migrate_legacy()
            elif df is None:
                print("[DataFeed] %s 拉取失败且无缓存" % code)
                return None
            # else: 增量拉取失败, 用旧缓存兜底

        if df is None:
            return None

        # 按请求范围过滤, 现场计算复权价
        out = df[(df["trade_date"] >= start_ts) & (df["trade_date"] <= end_ts)].copy()
        if len(out) == 0:
            return None
        if adjusted:
            latest_adj = df["adj_factor"].iloc[-1]
            out["close"] = out["close"] * out["adj_factor"] / latest_adj
        return out.reset_index(drop=True)

    # ── 对外接口 ──
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
        """返回完整 DataFrame。auto_refresh=True 时按代码独立增量更新"""
        start = start.replace("-", ""); end = end.replace("-", "")
        result = {}
        for code in codes:
            df = self._load_or_fetch(code, start, end)
            if df is not None and len(df) > 0:
                result[code] = df
        return result

    def get_latest_date(self) -> str:
        return datetime.now().strftime("%Y%m%d")

    def is_fresh(self, max_age_days: int = 2) -> bool:
        """缓存中最新数据日期距今天不超过 max_age_days"""
        latest = None
        import glob
        for f in glob.glob(os.path.join(self.cache_dir, "*.csv")):
            if "_" in os.path.basename(f).replace(".csv", "") and \
               os.path.basename(f).count("_") > 1:
                continue  # 跳过旧格式
            try:
                df = pd.read_csv(f, parse_dates=["trade_date"], usecols=["trade_date"])
                if len(df) > 0:
                    mx = df["trade_date"].max()
                    if latest is None or mx > latest:
                        latest = mx
            except Exception:
                pass
        if latest is None:
            return False
        return (pd.Timestamp.now() - latest).days <= max_age_days


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
