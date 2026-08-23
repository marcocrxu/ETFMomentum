# -*- coding: utf-8 -*-
"""加仓时机研究: 立即追当前持仓(A) vs 等下一次调仓再进(B) vs 10日均分(D)

复用生产回测引擎 run_backtest (与 web/实盘同口径), 在全历史每一天模拟一笔加仓:
  A 立即:   注入当日收盘按策略当前持仓买入, 此后完全跟随策略
  B 等调仓:  现金等待, 策略持仓第一次变化(切换/买卖)当天收盘入场, 此后跟随策略
  D 分批:   注入后 10 个交易日每日 1/10 按当日策略持仓买入, 此后跟随

比较同一日历窗口 [注入日, 注入日+H] 的收益, 统计:
  胜率 P(A>B)、平均/中位差、各路径平均收益、等待天数分布、
  按注入时持仓标的 / 该标的20日动量正负 分组。

用法: PYTHONPATH=. python3 backtest/add_capital_entry.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from web.bt_runner import run_backtest

UNIVERSE = "纳指,黄金,创业"          # 生产默认池
START, END = "2016-07-07", "2026-08-21"
HORIZONS = [20, 60, 120, 250]        # 交易日 (约 1/3/6/12 个月)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "add_capital_entry_result.json")

def pct(x):
    return round(float(x) * 100, 2)

def main():
    print("跑生产口径回测 %s (%s~%s) ..." % (UNIVERSE, START, END))
    r = run_backtest({"universe": UNIVERSE, "start_date": START, "end_date": END})
    if "error" in r:
        print("回测失败:", r["error"]); sys.exit(1)
    dates, nav = r["dates"], np.array(r["nav"])
    pos = np.array(r["position_series"])
    labels = r["position_labels"]
    n = len(dates)
    assert len(nav) == n and len(pos) == n, "序列长度不一致"
    print("基准校验: total_ret=%.2f%% calmar=%.2f (当前默认基线 3683.00/2.43, 窗口 2016-07-07~2026-08-12)" % (r["total_ret"], r["calmar"]))
    print("样本: %d 交易日 (%s ~ %s), 换仓次数 %d\n" % (n, dates[0], dates[-1], r["trade_count"]))

    # 各标的 20 日动量 (用于分组)
    mom20 = {}
    for nm, s in r["etf_navs"].items():
        arr = np.array([v if v is not None else np.nan for v in s])
        m = np.full(n, np.nan); m[20:] = arr[20:] / arr[:-20] - 1
        mom20[nm] = m

    summary = {}
    PNL_BUCKETS = [("≤-5%", -1e9, -5), ("(-5,-2]%", -5, -2), ("(-2,0]%", -2, 0),
                   ("(0,2]%", 0, 2), ("(2,5]%", 2, 5), ("(5,10]%", 5, 10), (">10%", 10, 1e9)]
    for H in HORIZONS:
        rows = []
        leg_entry = None          # 当前持仓腿的入场日索引
        for i in range(0, n - H - 1):
            p_i = pos[i]
            if i == 0 or pos[i] != pos[i - 1]:
                leg_entry = i if p_i >= 0 else None    # 持仓变化日 = 新腿起点
            retA = nav[i + H] / nav[i] - 1                    # 立即追仓, 跟随策略
            s = None
            for j in range(i + 1, i + H + 1):                  # 第一次持仓变化
                if pos[j] != p_i:
                    s = j; break
            retB = (nav[i + H] / nav[s] - 1) if s is not None else 0.0
            waited = (s - i) if s is not None else H
            # 分批: 10 个切片各在第 i+k 天入场后跟随策略
            ks = min(10, H)
            retD = float(np.mean([nav[i + H] / nav[i + k] - 1 for k in range(1, ks + 1)]))
            nm = labels[p_i + 1]        # position_series: -1=空仓, 0..N-1=标的; labels 前插了"空仓"
            m20 = mom20.get(nm, np.full(n, np.nan))[i]
            # 注入时当前持仓腿的浮盈/浮亏 (净值口径: nav[i]/nav[leg_entry]-1)
            leg_pnl = ((nav[i] / nav[leg_entry] - 1) * 100
                       if p_i >= 0 and leg_entry is not None and nav[leg_entry] > 0 else None)
            rows.append({"i": i, "date": dates[i], "pos_i": int(p_i), "pos": nm,
                         "retA": retA, "retB": retB, "retD": retD, "waited": waited,
                         "mom20": None if (m20 is None or np.isnan(m20)) else round(float(m20), 4),
                         "leg_pnl": round(float(leg_pnl), 2) if leg_pnl is not None else None})

        # 只统计"注入时策略有持仓"的日子 (空仓日 A≡B 无信息量, 单独计数)
        held = [x for x in rows if x["pos_i"] >= 0]
        def agg(rs):
            if not rs: return None
            d = np.array([x["retA"] - x["retB"] for x in rs])  # A-B
            a = np.array([x["retA"] for x in rs]); b = np.array([x["retB"] for x in rs])
            w = np.array([x["waited"] for x in rs])
            return {"n": len(rs),
                    "winA": pct(np.mean(d > 0)), "mean_diff": pct(np.mean(d)),
                    "median_diff": pct(np.median(d)), "meanA": pct(np.mean(a)),
                    "meanB": pct(np.mean(b)), "medianA": pct(np.median(a)),
                    "medianB": pct(np.median(b)), "meanD": pct(np.mean([x["retD"] for x in rs])),
                    "wait_median": float(np.median(w)), "wait_p90": float(np.percentile(w, 90)),
                    "wait_mean": round(float(np.mean(w)), 1),
                    "never_switch_pct": pct(np.mean(w >= H))}

        by_pos = {}
        for nm in sorted(set(x["pos"] for x in held)):
            by_pos[nm] = agg([x for x in held if x["pos"] == nm])
        by_mom = {"mom20_positive": agg([x for x in held if x["mom20"] is not None and x["mom20"] > 0]),
                  "mom20_negative": agg([x for x in held if x["mom20"] is not None and x["mom20"] <= 0])}
        # 按注入时当前持仓腿的浮盈/浮亏分档
        by_pnl = {}
        for label, lo, hi in PNL_BUCKETS:
            by_pnl[label] = agg([x for x in held if x["leg_pnl"] is not None
                                 and lo < x["leg_pnl"] <= hi])
        cash_n = len([x for x in rows if x["pos_i"] < 0])

        # 分布: A-B 的尾部不对称性 (A大胜/B大胜以 ±5pp 为界)
        def dist(rs):
            if not rs: return None
            d = np.array([(x["retA"] - x["retB"]) * 100 for x in rs])
            return {"n": len(d),
                    "A_big_ge5": pct(np.mean(d >= 5)), "A_small_1_5": pct(np.mean((d >= 1) & (d < 5))),
                    "tie_abs1": pct(np.mean(np.abs(d) < 1)),
                    "B_small_1_5": pct(np.mean((d <= -1) & (d > -5))), "B_big_le5": pct(np.mean(d <= -5)),
                    "A_p95": round(float(np.percentile(d, 95)), 1), "B_p5": round(float(np.percentile(d, 5)), 1)}
        by_pos_dist = {nm: dist([x for x in held if x["pos"] == nm]) for nm in by_pos}

        summary[str(H)] = {"overall": agg(held), "overall_dist": dist(held),
                           "by_pos": by_pos, "by_pos_dist": by_pos_dist,
                           "by_mom20": by_mom, "by_leg_pnl": by_pnl, "cash_days": cash_n}

        print("=" * 72)
        print("持有期 H = %d 交易日 (约 %d 个月)" % (H, round(H / 21)))
        print("  注入时持仓 %d 天, 空仓 %d 天(空仓日A≡B, 已剔除)" % (len(held), cash_n))
        o = agg(held)
        print("  【总体】n=%d  胜率A>B=%.1f%%  平均差(A-B)=%+.2f%%  中位差=%+.2f%%" % (o["n"], o["winA"], o["mean_diff"], o["median_diff"]))
        print("            A均收益=%+.2f%%  B均收益=%+.2f%%  D(分批)均=%+.2f%%" % (o["meanA"], o["meanB"], o["meanD"]))
        print("            等待下次调仓: 中位 %d 天 / 均值 %.1f 天 / P90 %d 天 / 期内无调仓 %.1f%%" % (
            o["wait_median"], o["wait_mean"], o["wait_p90"], o["never_switch_pct"]))
        for nm, s2 in by_pos.items():
            if s2 and s2["n"] >= 30:
                print("  [注入时持 %s] n=%d  胜率A>B=%.1f%%  平均差=%+.2f%%  (A均%+.2f%% vs B均%+.2f%%)" % (
                    nm, s2["n"], s2["winA"], s2["mean_diff"], s2["meanA"], s2["meanB"]))
        for k, s2 in by_mom.items():
            if s2 and s2["n"] >= 30:
                print("  [注入时持仓20日动量: %s] n=%d  胜率A>B=%.1f%%  平均差=%+.2f%%" % (
                    "正" if k.endswith("positive") else "负", s2["n"], s2["winA"], s2["mean_diff"]))
        dd = dist(held)
        print("  【分布 A-B】A大胜>=5pp %.1f%% | A小胜1-5 %.1f%% | 打平±1 %.1f%% | B小胜1-5 %.1f%% | B大胜>=5 %.1f%% | A的P95 %+.1f / B的P5 %+.1f" % (
            dd["A_big_ge5"], dd["A_small_1_5"], dd["tie_abs1"], dd["B_small_1_5"], dd["B_big_le5"], dd["A_p95"], dd["B_p5"]))
        for nm, dd2 in sorted(by_pos_dist.items()):
            if dd2 and dd2["n"] >= 30:
                print("    [持%s] A大胜%.1f%% A小胜%.1f%% 打平%.1f%% B小胜%.1f%% B大胜%.1f%% | P95 %+.1f / P5 %+.1f" % (
                    nm, dd2["A_big_ge5"], dd2["A_small_1_5"], dd2["tie_abs1"], dd2["B_small_1_5"], dd2["B_big_le5"], dd2["A_p95"], dd2["B_p5"]))
        print("  【按注入时当前持仓收益率分档】(A=立即追 B=等调仓 D=分批)")
        print("    %-9s %5s | %7s %9s | %8s %8s %8s | 推荐" % ("档位", "n", "A胜率", "A-B均值", "A均收益", "B均收益", "D均收益"))
        for label, _, _ in PNL_BUCKETS:
            s3 = by_pnl[label]
            if s3 and s3["n"] >= 30:
                rec = "立即追" if s3["mean_diff"] >= 1 else ("等调仓" if s3["mean_diff"] <= -1 else "差别小/分批")
                print("    %-9s %5d | %6.1f%% %+8.2fpp | %+7.2f%% %+7.2f%% %+7.2f%% | %s" % (
                    label, s3["n"], s3["winA"], s3["mean_diff"], s3["meanA"], s3["meanB"], s3["meanD"], rec))

    # 最近状态 (回答"现在怎么办")
    print("=" * 72)
    print("最近 5 次调仓:")
    for d in r["decisions"][-5:]:
        print("   %s  %-10s  pnl=%s%%  nav=%.3f  (%s)" % (d["date"], d["action"], d["pnl"], d["nav"], d["reason"]))
    print("最后持仓: %s" % r["last_position"])

    with open(OUT, "w") as f:
        json.dump({"universe": UNIVERSE, "window": [START, END], "baseline": {
            "total_ret": r["total_ret"], "calmar": r["calmar"]},
            "summary": summary}, f, ensure_ascii=False, indent=2)
    print("\n明细已存:", OUT)

if __name__ == "__main__":
    main()
