"""宽基ETF份额数据层 — 上交所官网接口 + CSV 增量缓存

数据源: query.sse.com.cn ETF规模查询接口, 单次请求返回当日全部沪市ETF份额。
国家队申购主战场(510300/510500/510050)均为沪市, 故无需深交所源,
也不引入 akshare 依赖 (仅 requests)。

T+1 纪律: T 日份额在 T 日晚间发布, 实盘 cron 14:50 决策时当日份额不可得;
build_flow 统一 shift(1), 回测/实盘同口径, 无前视。

缓存: cache_dir/sse_etf_shares.csv 宽表 (date + 每code一列, 单位亿份),
增量更新只拉缺口日期, 原子写回。
"""
import os
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

SSE_URL = "https://query.sse.com.cn/commonQuery.do"
_HEADERS = {
    "Referer": "https://www.sse.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
HTTP_TIMEOUT = 15
FETCH_SLEEP_SEC = 0.15        # 逐日回填限速 (防远端封禁)
MAX_CONSEC_FAIL = 10          # 连续空结果暂停 (长假最多约5个连续非交易工作日)
DEFAULT_START = "2016-01-01"  # SSE ETF 份额披露可得起点 (实测 2016-01-04 有数据)
CHECKPOINT_ROWS = 300         # 每拉 N 天落盘一次, 中断不丢已拉数据


def _norm_code(code: str) -> str:
    """'510300.SH' / '510300' 统一为 '510300'"""
    return code.split(".")[0].strip()


def fetch_sse_shares(date_str: str) -> Dict[str, float]:
    """拉取某日全部沪市ETF份额 -> {code: 亿份}; 失败/非交易日返回 {}"""
    params = {
        "isPagination": "true", "pageHelp.pageSize": "10000",
        "pageHelp.pageNo": "1", "pageHelp.beginPage": "1",
        "pageHelp.cacheSize": "1", "pageHelp.endPage": "1",
        "sqlId": "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L",
        "STAT_DATE": date_str,
    }
    try:
        r = requests.get(SSE_URL, params=params, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        rows = r.json().get("result") or []
    except Exception as e:
        print(f"[SHARE] sse {date_str} failed: {e}")
        return {}
    out = {}
    for row in rows:
        code = str(row.get("SEC_CODE", ""))
        vol = row.get("TOT_VOL")
        if code and vol is not None:
            out[code] = float(vol) / 1e4   # 万份 -> 亿份
    return out


def build_flow_from(shares_df: pd.DataFrame, dates, codes: List[str], window: int) -> np.ndarray:
    """纯函数: 由份额宽表构建对齐 dates 的篮子净申购率序列 (%), 无前视。

    flow[t] = mean_c( s_c[t-1] / s_c[t-1-window] - 1 ) * 100
      - shift(1): 决策日 t 只能用 t-1 及更早的份额 (T+1 发布)
      - 逐 code 计算后等权平均: 单 code 缺失不拖垮篮子, 也不被单只异常主导
      - 数据不足/未覆盖日为 NaN (调用方降级为不拦截)
    """
    n = len(dates)
    if shares_df is None or len(shares_df) == 0 or window <= 0:
        return np.full(n, np.nan)
    idx = pd.to_datetime(pd.Index(dates))
    cols = []
    for c in codes:
        c = _norm_code(c)
        if c in shares_df.columns:
            s = pd.Series(shares_df[c].values, index=pd.to_datetime(shares_df["date"]))
            s = s.reindex(idx)
            shifted = s.shift(1)                      # T+1 发布滞后
            flow = (shifted / shifted.shift(window) - 1) * 100
            cols.append(flow.values)
    if not cols:
        return np.full(n, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.array(cols, dtype=float), axis=0)


class ShareFlowFeed:
    """份额缓存与增量回填 (边拉边写, 重跑只补缺失段)"""

    def __init__(self, cache_dir: str = "/home/ecs-user/backtest/cache/share"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, "sse_etf_shares.csv")

    def load(self) -> pd.DataFrame:
        """读缓存 -> 宽表 DataFrame (date 列 + 每code一列), 无缓存返回空表"""
        if not os.path.exists(self._cache_path):
            return pd.DataFrame(columns=["date"])
        df = pd.read_csv(self._cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _save(self, df: pd.DataFrame) -> None:
        tmp = self._cache_path + ".tmp"
        df.to_csv(tmp, index=False, date_format="%Y-%m-%d")
        os.replace(tmp, self._cache_path)

    def update(self, trade_dates: Optional[List[str]] = None,
               end_date: Optional[str] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> int:
        """增量回填: 从缓存最大日期(或 DEFAULT_START)补到 end_date。

        trade_dates: 已知交易日序列(YYYY-MM-DD); None 则按工作日试探
        (节假日返回空跳过, 由 MAX_CONSEC_FAIL 防远端故障空转)。
        返回新增行数。
        """
        df = self.load()
        start = DEFAULT_START
        if len(df) > 0:
            start = (df["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        if start > end:
            return 0

        if trade_dates is not None:
            days = [d for d in trade_dates if start <= d <= end]
        else:
            days = []
            cur = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            while cur <= end_dt:
                if cur.weekday() < 5:
                    days.append(cur.strftime("%Y-%m-%d"))
                cur += timedelta(days=1)

        batch: List[dict] = []
        added, fails = 0, 0
        for k, d in enumerate(days):
            rows = fetch_sse_shares(d)
            if rows:
                rows["date"] = d
                batch.append(rows)
                added += 1
                fails = 0
            else:
                fails += 1
                if fails >= MAX_CONSEC_FAIL:
                    print(f"[SHARE] 连续 {fails} 天空结果, 暂停回填 (远端故障?)")
                    break
            if progress and (k % 50 == 0):
                progress(k, len(days))
            if len(batch) >= CHECKPOINT_ROWS:
                df = self._merge(df, batch)
                batch = []
            time.sleep(FETCH_SLEEP_SEC)
        if batch:
            df = self._merge(df, batch)
        return added

    def _merge(self, df: pd.DataFrame, batch: List[dict]) -> pd.DataFrame:
        new = pd.DataFrame(batch)
        new["date"] = pd.to_datetime(new["date"])
        df = pd.concat([df, new], ignore_index=True)
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        df = df.reset_index(drop=True)
        self._save(df)
        return df

    def build_flow(self, dates, codes: List[str], window: int) -> np.ndarray:
        return build_flow_from(self.load(), dates, codes, window)
