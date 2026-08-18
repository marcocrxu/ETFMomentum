"""美股信号源 — QQQ(美元价)×USDCNY 的人民币口径日线序列, 用于纳指动量混合 (I 组审计)

数据: data/qqq_usdcny.csv (2013-06 起, Yahoo Finance chart API, 已随仓库提交)
刷新: refresh_qqq_cny() 拉取最新数据原子写回 (可选, 需外网; 失败不影响读取旧缓存)
对齐: CN 交易日 D 的收盘决策, 只用美国日历日期 ≤ D-1 的已收盘交易段 (严格无前视)
"""
import os
import time

import numpy as np
import pandas as pd

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qqq_usdcny.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _fetch(sym, period1="2013-06-01"):
    import requests
    p1 = int(pd.Timestamp(period1).timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%s" % sym,
                     params={"period1": p1, "period2": p2, "interval": "1d"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    ts = d["timestamp"]
    close = d["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s", utc=True).date, sym: close})
    return df.dropna()


def refresh_qqq_cny():
    """拉取 QQQ + USDCNY 最新日线, 原子写回 data/qqq_usdcny.csv"""
    q = _fetch("QQQ")
    time.sleep(1)
    fx = _fetch("USDCNY=X")
    df = q.merge(fx, on="date", how="inner").sort_values("date")
    tmp = CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, CSV)
    return df


def load_qqq_cny():
    """读取本地 CSV, 返回含 qqq_cny 列 (QQQ×USDCNY) 的 DataFrame"""
    df = pd.read_csv(CSV, parse_dates=["date"])
    df["qqq_cny"] = df["QQQ"] * df["USDCNY=X"]
    return df


def align_to_cn(cn_dates, df=None):
    """CN 交易日序列 → QQQ-CNY 收盘数组: 每个 CN 日期取美国日期 ≤ D-1 的最后收盘 (无前视)"""
    if df is None:
        df = load_qqq_cny()
    s = df.set_index("date")["qqq_cny"]
    idx = pd.DatetimeIndex(s.index)
    out = np.full(len(cn_dates), np.nan)
    for i, d in enumerate(cn_dates):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        pos = idx.searchsorted(dt, side="right") - 1
        if pos >= 0:
            out[i] = s.iloc[pos]
    return out
