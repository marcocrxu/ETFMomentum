# -*- coding: utf-8 -*-
"""赚钱效应数据 —— 十年版 + 多价格假设 (流式构建, 内存 O(1))

对每个交易日的四个群体 (lim=涨停 / top=成交额前10 / up=上涨 / dn=下跌),
在【信号日的不同买入价】×【次日的不同卖出价】下分别算平均收益:

  买入价 b ∈ {o=开盘, l=最低, a=均价(VWAP), c=收盘, h=最高}   (取信号日 T)
  卖出价 s ∈ {o=开盘, l=最低, a=均价(VWAP), c=收盘, h=最高}   (取次日 T+1)
  收益率 = 卖出价 / 买入价 - 1  (组内等权平均)

另保留可执行口径 (T+1 开盘买 → T+2 开盘/收盘卖), 字段 `{u}_{g}_o` / `{u}_{g}_c`。
涨停判定: 科创板 20%; 创业板 2020-08-24 前 10%、之后 20%; 主板 10%。

产物: web/static/data/earn_effect.csv
用法: PYTHONPATH=. .venv/bin/python web/earn_build10y.py
"""
import os
import time

import numpy as np
import pandas as pd

from live.daily_run import _tushare_token
import tushare as ts

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "web", "static", "data", "earn_effect.csv")
START, END = "20160101", "20261022"
REG_DATE = "20200824"
BT = "olach"      # 买入价类型
ST = "olach"      # 卖出价类型


def limit_thr(code, d):
    if code.startswith("688"):
        return 19.5
    if code.startswith(("300", "301")):
        return 19.5 if d >= REG_DATE else 9.8
    return 9.8


def px_dict(o, h, l, c, v, amt):
    """五种价格; 均价用成交额/成交量近似 VWAP (amt 千元, v 手)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(v > 0, amt * 10.0 / v, c)
    return {"o": o, "h": h, "l": l, "c": c, "a": a.astype("float32")}


def build(days):
    pro = ts.pro_api(_tushare_token())
    rows = []
    prev2 = prev = None
    t0 = time.time()
    for i, d in enumerate(days):
        df = pro.daily(trade_date=d,
                       fields="ts_code,open,high,low,close,vol,amount,pct_chg")
        if df is None or len(df) < 800:
            continue
        codes = df.ts_code.values.astype("U10")
        pct = df.pct_chg.values.astype("float32")
        P = px_dict(df.open.values.astype("float32"), df.high.values.astype("float32"),
                    df.low.values.astype("float32"), df.close.values.astype("float32"),
                    df.vol.values.astype("float32"), df.amount.values.astype("float32"))
        am = df.amount.values.astype("float32")

        masks = {"all": np.ones(len(codes), bool),
                 "cyb": np.array([c.startswith(("300", "301")) for c in codes])}
        groups = {}
        for u, m in masks.items():
            cs, ps, ams = codes[m], pct[m], am[m]
            lim_mask = (np.array([ps[k] >= limit_thr(cs[k], d) for k in range(len(cs))])
                        if len(cs) else np.zeros(0, bool))
            groups[u] = {"lim": cs[lim_mask],
                         "top": cs[np.argsort(-ams)[:10]] if len(cs) else np.array([], dtype="U10"),
                         "up": cs[ps > 0], "dn": cs[ps < 0]}

        now = {"code": codes, "pct": pct, "P": P, "amt": am,
               "groups": groups, "masks": masks, "date": d}

        if prev is not None:
            r = {"date": pd.to_datetime(str(d), format="%Y%m%d")}
            pos_cur = {c: k for k, c in enumerate(codes)}
            for u in ("all", "cyb"):
                m = masks[u]
                r[f"{u}_breadth"] = float((pct[m] > 0).mean() * 100)
                r[f"{u}_median"] = float(np.median(pct[m]))
                r[f"{u}_lim_n"] = int(len(prev["groups"][u]["lim"]))
                for g in ("lim", "top", "up", "dn"):
                    mem = set(prev["groups"][u][g].tolist())
                    pairs = [(k, pos_cur[c]) for k, c in enumerate(prev["code"])
                             if c in mem and c in pos_cur]
                    if not pairs:
                        for b in BT:
                            for s in ST:
                                r[f"{u}_{g}_{b}{s}"] = np.nan
                        r[f"{u}_{g}_o"] = np.nan
                        r[f"{u}_{g}_c"] = np.nan
                        r[f"{u}_{g}_i"] = np.nan
                        continue
                    i1 = np.array([a for a, _ in pairs], dtype=int)
                    i2 = np.array([b for _, b in pairs], dtype=int)
                    for b in BT:
                        bp = prev["P"][b][i1].astype(float)
                        for s in ST:
                            sp = P[s][i2].astype(float)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                ret = np.where(bp > 0, sp / bp - 1, np.nan)
                            r[f"{u}_{g}_{b}{s}"] = float(np.nanmean(ret)) * 100
                    r[f"{u}_{g}_o"] = np.nan
                    r[f"{u}_{g}_c"] = np.nan
                    r[f"{u}_{g}_i"] = r[f"{u}_{g}_cc"]
            rows.append(r)

        if prev2 is not None:      # 可执行: prev2 选股, prev 开盘买, 今日开盘/收盘卖
            p1 = {c: k for k, c in enumerate(prev["code"])}
            p2 = {c: k for k, c in enumerate(codes)}
            for u in ("all", "cyb"):
                for g in ("lim", "top", "up", "dn"):
                    oo, cc = [], []
                    for c in prev2["groups"][u][g]:
                        if c in p1 and c in p2:
                            oo.append(P["o"][p2[c]] / prev["P"]["o"][p1[c]] - 1)
                            cc.append(P["c"][p2[c]] / prev["P"]["o"][p1[c]] - 1)
                    rows[-1][f"{u}_{g}_o"] = float(np.mean(oo)) * 100 if oo else np.nan
                    rows[-1][f"{u}_{g}_c"] = float(np.mean(cc)) * 100 if cc else np.nan

        prev2, prev = prev, now
        if (i + 1) % 200 == 0:
            print("  %d/%d %.0fs" % (i + 1, len(days), time.time() - t0), flush=True)

    return pd.DataFrame(rows)


def main():
    pro = ts.pro_api(_tushare_token())
    cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END)
    days = sorted(cal[cal.is_open == 1].cal_date.tolist())
    print("交易日", len(days), days[0], "→", days[-1], flush=True)
    D = build(days)
    D.to_csv(OUT, index=False)
    print("saved ->", OUT, D.shape, D.date.min().date(), D.date.max().date())


def refresh(n=6):
    """日常更新: 只重算最近 n 个交易日并合并进 CSV (几分钟)"""
    from datetime import datetime, timedelta
    pro = ts.pro_api(_tushare_token())
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=int(n * 2.5) + 10)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    days = sorted(cal[cal.is_open == 1].cal_date.tolist())[-(n + 2):]
    new = build(days)
    old = pd.read_csv(OUT, parse_dates=["date"])
    df = pd.concat([old[~old.date.isin(new.date)], new]).sort_values("date")
    df.to_csv(OUT, index=False)
    print("refreshed:", [str(d.date()) for d in new.date], "| rows", len(df))


if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv:
        refresh()
    else:
        main()
