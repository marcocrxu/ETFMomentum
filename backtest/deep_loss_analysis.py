# -*- coding: utf-8 -*-
"""深亏交易(≤-5%)成因与规避可行性研究 (2026-08-23)

1) 全窗口逐笔列出 pnl≤-5% 的腿: 出入场原因/持有天数/标的
2) 入场时技术画像: 标的 mom20、价vsMA20/MA60、上证 mom20、入场日跳空
3) 用引擎已有闸门做规避实验: entry_mom_cap / entry_bias_cap / trend_ma /
   entry_decel_on, 看各自能躲掉多少深亏腿、代价多少
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from web.bt_runner import run_backtest

BASE = {"universe": "纳指,黄金,创业", "start_date": "2016-07-07", "end_date": "2026-08-21"}

def legs_of(r):
    legs, entry = [], None
    for d in r["decisions"]:
        a = d["action"]
        if a.startswith("买入"):
            entry = {"asset": a[2:].split("(")[0], "entry": d["date"], "reason": d["reason"]}
        elif a.startswith("卖出") or "→" in a:
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                legs.append(entry)
            entry = None if "→" not in a else {"asset": a.split("→")[1].split("(")[0],
                                               "entry": d["date"], "reason": d["reason"]}
        elif a.startswith("持有"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                legs.append(entry)
            entry = None
    return legs

def main():
    r = run_backtest(dict(BASE))
    dates = r["dates"]
    dix = {d: i for i, d in enumerate(dates)}
    legs = legs_of(r)
    deep = [l for l in legs if l["pnl"] is not None and l["pnl"] <= -5]
    print("全窗口 %s~%s: 共 %d 条腿, 深亏(≤-5%%) %d 条 (%.1f%%)" % (
        dates[0], dates[-1], len(legs), len(deep), 100 * len(deep) / len(legs)))
    print()
    print("== 深亏腿明细 ==")
    print("%-11s %-6s %8s %5s %-10s -> %-10s" % ("入场日", "标的", "pnl", "天数", "入场原因", "出场原因"))
    for l in deep:
        print("%-11s %-6s %+7.2f%% %5s %-10s -> %-10s" % (
            l["entry"], l["asset"], l["pnl"], l["days"], l["reason"], l["exit_reason"]))
    print()
    from collections import Counter
    print("出场原因分布:", dict(Counter(l["exit_reason"] for l in deep)))
    print("入场原因分布:", dict(Counter(l["reason"] for l in deep)))
    print("标的分布:", dict(Counter(l["asset"] for l in deep)))
    print()

    # ── 入场技术画像 ──
    # 用 etf_navs 归一序列计算 mom20 / MA20 / MA60 偏离; 上证 mom20 用缓存读
    def series(nm):
        return np.array([v if v is not None else np.nan for v in r["etf_navs"][nm]])
    feats = {"deep": [], "ok": []}
    for l in legs:
        if l["pnl"] is None:
            continue
        i = dix.get(l["entry"])
        if i is None or i < 60:
            continue
        nm = l["asset"]
        s = series(nm)
        mom20 = (s[i] / s[i - 20] - 1) * 100 if not np.isnan(s[i - 20]) else np.nan
        ma20 = np.nanmean(s[i - 19:i + 1]); ma60 = np.nanmean(s[i - 59:i + 1])
        bias20 = (s[i] / ma20 - 1) * 100; bias60 = (s[i] / ma60 - 1) * 100
        gap = (s[i] / s[i - 1] - 1) * 100 if i > 0 and not np.isnan(s[i - 1]) else np.nan
        rec = {"mom20": mom20, "bias20": bias20, "bias60": bias60, "gap": gap}
        (feats["deep"] if l["pnl"] <= -5 else feats["ok"]).append(rec)
    print("== 入场时技术画像 (深亏腿 vs 其他腿, 均值/中位) ==")
    for k in ["mom20", "bias20", "bias60", "gap"]:
        for grp in ["deep", "ok"]:
            v = [x[k] for x in feats[grp] if x[k] is not None and not np.isnan(x[k])]
            print("  %-6s %-4s n=%3d 均值%+7.2f 中位%+7.2f" % (k, grp, len(v), np.mean(v), np.median(v)))
    print()

    # ── 规避实验: 引擎已有闸门 ──
    print("== 规避实验 (全窗口, 加闸门后: 深亏腿数 / 累计 / 回撤 / Calmar) ==")
    def show(tag, over):
        rr = run_backtest(dict(BASE, **over))
        dd = [l for l in legs_of(rr) if l["pnl"] is not None and l["pnl"] <= -5]
        print("  %-24s 深亏 %2d 条 | %+8.1f%% %6.1f%% %.2f" % (
            tag, len(dd), rr["total_ret"], rr["max_dd"], rr["calmar"]))
    show("基线(无闸门)", {})
    for cap in (5.0, 8.0, 10.0):
        show("entry_mom_cap=%g" % cap, {"entry_mom_cap": cap})
    for bias in (2.0, 3.0, 5.0):
        show("entry_bias_cap=%g" % bias, {"entry_bias_cap": bias})
    for ma in (60, 100, 200):
        show("trend_ma=%d" % ma, {"trend_ma": ma})
    show("entry_decel_on", {"entry_decel_on": True})

if __name__ == "__main__":
    main()
