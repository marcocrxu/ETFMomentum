# -*- coding: utf-8 -*-
"""N-23: "收盘成交"到底可不可行 —— 用实盘 14:50 报价 vs 当日收盘 直接量化

数据来源 (不是回测, 是实盘留痕):
  - live/notify_log.jsonl: 每个交易日 14:50:xx 推送的**当时报价**(¥) + 动量, 共 N 天
  - web/mytrack_data.json: 16 笔**真实成交**(其中 4 笔带成交时间戳)
  - data/feed.py 缓存: 当日收盘
口径: 复权价 = raw × adj_factor / 最新adj_factor (与 feed.py 一致)

回答三个问题:
  Q1 14:50 报价 → 收盘, 10 分钟漂移多大?
  Q2 这个漂移会不会改变排名第一(即改变信号)?
  Q3 真实成交价 vs 收盘价差多少?

用法: .venv/bin/python audit/n_close_exec_feasibility.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

CACHE = "/home/ecs-user/backtest/cache/etf_momentum"
CODE = {"纳指": "513100_SH", "黄金": "518880_SH", "创业": "159915_SZ"}
PAT = re.compile(r"\*\*(纳指|黄金|创业)\*\*:\s*([+-]?[\d.]+)%\s*\(¥([\d.]+)\)")
DPAT = re.compile(r"⏰\s*(\d{4}-\d{2}-\d{2})")


def adj_close(code, date):
    df = pd.read_csv(os.path.join(CACHE, code + ".csv"), parse_dates=["trade_date"])
    latest = float(df.adj_factor.iloc[-1])
    x = df[df.trade_date == pd.Timestamp(date)]
    if len(x) == 0:
        return None
    x = x.iloc[0]
    return float(x.close) * float(x.adj_factor) / latest


def main():
    rows = []
    for line in open(os.path.join(HERE, "live", "notify_log.jsonl")):
        o = json.loads(line)
        if o.get("kind") != "daily":
            continue
        if not o.get("ts", "")[11:13] == "14":       # 剔除 01:52 的异常测试推送
            continue
        m = DMATCH = DPAT.search(o.get("message", ""))
        if not m:
            continue
        date = m.group(1)
        q = {k: float(v) for k, _, v in PAT.findall(o["message"])}
        if len(q) < 3:
            continue
        ts = o.get("ts", "")[11:]
        mom = (o.get("extra") or {}).get("momentums") or {}
        rows.append(dict(date=date, ts=ts, **{("q_" + k): v for k, v in q.items()}, mom=mom))
    print("=== 实盘 14:5x 报价样本: %d 天 (%s ~ %s) ===" % (len(rows), rows[0]["date"], rows[-1]["date"]))
    print("推送时刻分布: %s" % sorted({r["ts"][:5] for r in rows}))

    out = []
    for r in rows:
        d = r["date"]
        rec = dict(date=d, ts=r["ts"])
        for nm in CODE:
            c = adj_close(CODE[nm], d)
            q = r.get("q_" + nm)
            if c is None or q is None:
                rec["drift_" + nm] = np.nan
                continue
            rec["drift_" + nm] = (c / q - 1) * 1e4
            rec["q_" + nm] = q
            rec["c_" + nm] = c
        # 14:50 与收盘各自的排名第一
        drifts = {nm: rec.get("drift_" + nm) for nm in CODE}
        if all(np.isfinite(v) for v in drifts.values()):
            q_rank = max(CODE, key=lambda nm: r.get("q_" + nm, np.nan))
            c_rank = max(CODE, key=lambda nm: rec.get("c_" + nm, np.nan))
            rec["rank_same_price"] = (q_rank == c_rank)
            # 动量口径下 (14:50 推送的动量 + 漂移) 第一名是否变化
            mm = r.get("mom") or {}
            if len(mm) == 3:
                c_mom = {nm: mm[nm] + drifts[nm] / 100.0 for nm in CODE}
                rec["rank_same_mom"] = (max(CODE, key=lambda n: mm[n]) == max(CODE, key=lambda n: c_mom[n]))
                rec["q_top"] = max(CODE, key=lambda n: mm[n])
                rec["c_top"] = max(CODE, key=lambda n: c_mom[n])
        out.append(rec)

    df = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    cols = ["date", "ts"] + ["drift_" + n for n in CODE]
    print("\n=== Q1 14:50 报价 → 收盘 的漂移 (bp; 正=收盘更高) ===")
    print(df[cols].to_string(index=False, float_format=lambda v: "%+8.1f" % v))
    print()
    for nm in CODE:
        s = df["drift_" + nm].dropna()
        print("  %-4s n=%2d  均值 %+6.1fbp  中位 %+6.1fbp  绝对均值 %5.1fbp  P90|.| %5.1fbp  最大 %+6.1f  最小 %+6.1f"
              % (nm, len(s), s.mean(), s.median(), s.abs().mean(),
                 s.abs().quantile(0.9), s.max(), s.min()))

    print("\n=== Q2 这个漂移会不会改变第一名 (即改变信号) ===")
    if "rank_same_price" in df:
        a = df["rank_same_price"].dropna()
        print("  按**价格**排名第一是否与收盘一致: %d/%d 一致 (%.0f%%)"
              % (a.sum(), len(a), 100 * a.mean()))
    if "rank_same_mom" in df:
        b = df["rank_same_mom"].dropna()
        print("  按**动量**排名第一是否与收盘一致: %d/%d 一致 (%.0f%%)"
              % (b.sum(), len(b), 100 * b.mean()))
        bad = df[df["rank_same_mom"] == False]  # noqa: E712
        if len(bad):
            print("  排名发生变化的日期: %s"
                  % ", ".join("%s(%s→%s)" % (x.date, x.q_top, x.c_top) for x in bad.itertuples()))

    print("\n=== Q3 真实成交价 vs 收盘 ===")
    rec = json.load(open(os.path.join(HERE, "web", "mytrack_data.json")))["records"]
    fills = []
    for r in rec:
        c = adj_close(CODE[r["asset"]], r["date"])
        if c is None:
            continue
        fills.append(dict(date=r["date"], asset=r["asset"], act=r["action"],
                          dev_bp=(float(r["price"]) / c - 1) * 1e4,
                          timed=("(" in r.get("note", "") or r["date"] >= "2026-09-09")))
    fd = pd.DataFrame(fills)
    for tag, g in (("全体", fd), ("策略单(9月后/带时间)", fd[fd.timed]), ("7-8月手工建仓", fd[~fd.timed])):
        print("  %-22s n=%2d  均值 %+7.1fbp  中位 %+6.1fbp  绝对均值 %5.1fbp"
              % (tag, len(g), g.dev_bp.mean(), g.dev_bp.median(), g.dev_bp.abs().mean()))

    p = os.path.join(HERE, "audit", "out", "n_close_exec_feasibility.json")
    json.dump(dict(days=len(df), per_asset={nm: dict(
        mean=float(df["drift_" + nm].mean()), median=float(df["drift_" + nm].median()),
        abs_mean=float(df["drift_" + nm].abs().mean()),
        p90_abs=float(df["drift_" + nm].abs().quantile(0.9)),
        max=float(df["drift_" + nm].max()), min=float(df["drift_" + nm].min()))
        for nm in CODE},
        rank_same_price=float(df["rank_same_price"].mean()) if "rank_same_price" in df else None,
        rank_same_mom=float(df["rank_same_mom"].mean()) if "rank_same_mom" in df else None,
        rows=df.to_dict("records")), open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> %s" % p)


if __name__ == "__main__":
    main()
