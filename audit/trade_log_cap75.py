"""交易明细导出: mom250 · top3 · R=10 · cap75% · 等权
窗口 2016-01-01 ~ 2025-12-31, 单边成本 0.05%
输出: audit/out/trades_cap75_equal_R10.csv (逐笔)
"""
import os

import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location("ps", "audit/param_search.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)

COST = 0.0005
TOP_K, R, CAP = 3, 10, 0.75
S0, S1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31")

panel = ps.load_panel()
P, names, n_a, pool_mask, cash_i, fac, vol = ps.build(panel)
dates = panel.index
mom = fac["mom250"]

reb = [i for i in range(250, len(P) - R, R)]
reb = [i for i in reb if S0 <= dates[i] <= S1]

w = np.zeros(n_a)
nav = 1.0
rows = []
rebal_log = []
for i in reb:
    S = mom[i]
    cand = np.where(pool_mask & ~np.isnan(S) & ~np.isnan(P[i]))[0]
    cand = cand[S[cand] <= CAP]
    w_new = np.zeros(n_a)
    if cand.size:
        order = cand[np.argsort(-S[cand])][:TOP_K]
        w_new[order] = 1.0 / len(order)
    else:
        w_new[cash_i] = 1.0
    turn = float(np.abs(w_new - w).sum())
    cost = COST * turn
    nav_pre = nav
    nav *= (1 - cost)
    p0, p1 = P[i], P[i + R]
    pos = []
    for j in np.where(w_new > 0)[0]:
        r = p1[j] / p0[j] - 1 if (p0[j] > 0 and not np.isnan(p1[j])) else 0.0
        pos.append((names[j], p0[j], p1[j], r, w_new[j], w_new[j] * r))
        rows.append(dict(
            rebal_date=dates[i].date(), exit_date=dates[i + R].date(),
            etf=names[j], entry_px=round(float(p0[j]), 4), exit_px=round(float(p1[j]), 4),
            ret_pct=round(float(r) * 100, 3), weight=round(float(w_new[j]), 4),
            contrib_pct=round(float(w_new[j] * r) * 100, 4),
            mom250_at_entry_pct=round(float(S[j]) * 100, 2),
            turnover=round(turn, 4), cost_pct=round(cost * 100, 4),
            nav_after=round(nav, 6)))
    gross = sum(x[5] for x in pos)
    nav *= (1 + gross)
    rebal_log.append(dict(date=dates[i], exit=dates[i + R], pos=pos, turn=turn,
                          cost=cost, gross=gross, net=gross - cost, nav=nav))
    w = w_new

trades = pd.DataFrame(rows)
os.makedirs("audit/out", exist_ok=True)
trades.to_csv("audit/out/trades_cap75_equal_R10.csv", index=False)

# ── 汇总 ──
navs = pd.Series([x["nav"] for x in rebal_log], index=[x["date"] for x in rebal_log])
print("=" * 100)
print("策略: mom250 · top3 · R=10 · cap75%% · 等权   |   窗口 %s ~ %s   |   单边成本 %.2f%%"
      % (dates[reb[0]].date(), dates[reb[-1] + R].date(), COST * 100))
print("=" * 100)
s = ps.stats(navs, np.array([x["turn"] for x in rebal_log]), R)
print("调仓 %d 次, 持仓 %d 笔, 年化 %+.2f%%, 波动 %.1f%%, 夏普 %.2f, 最大回撤 %.1f%%, Calmar %.2f, 年换手 %.0f%%"
      % (len(navs), len(trades), s["cagr"] * 100, s["vol"] * 100, s["sharpe"],
         s["mdd"] * 100, s["calmar"], s["turn"] * 100))
print("最终净值 %.3f (%.1f 倍), 累计成本拖累约 %.2f%%/年"
      % (nav, nav, (1 - (1 - COST * s["turn"])) * 100))

print("\n【逐笔统计】")
r = trades["ret_pct"].values
print("  单笔平均 %+.2f%%  中位数 %+.2f%%  胜率 %.1f%% (%d/%d)"
      % (r.mean(), np.median(r), (r > 0).mean() * 100, (r > 0).sum(), len(r)))
print("  平均盈利 %+.2f%%  平均亏损 %+.2f%%  盈亏比 %.2f"
      % (r[r > 0].mean(), r[r < 0].mean(), abs(r[r > 0].mean() / r[r < 0].mean())))
print("  最好单笔 %+.2f%%  最差单笔 %+.2f%%  标准差 %.1f%%" % (r.max(), r.min(), r.std(ddof=1)))

print("\n【各标的表现 (按贡献排序)】")
g = trades.groupby("etf").agg(次数=("ret_pct", "size"), 平均收益=("ret_pct", "mean"),
                              胜率=("ret_pct", lambda x: (x > 0).mean() * 100),
                              累计贡献=("contrib_pct", "sum"))
g = g.sort_values("累计贡献", ascending=False)
print("  %-10s %5s %9s %8s %10s" % ("标的", "持有次数", "平均收益", "胜率", "累计贡献"))
for k, v in g.iterrows():
    print("  %-10s %5d %+8.2f%% %7.1f%% %+9.1f%%" % (k, v["次数"], v["平均收益"], v["胜率"], v["累计贡献"]))

print("\n【年度收益】")
yr = navs.pct_change().dropna().groupby(navs.pct_change().dropna().index.year).apply(lambda x: (1 + x).prod() - 1)
for y, v in yr.items():
    sub = trades[pd.to_datetime(trades["rebal_date"]).dt.year == y]
    print("  %d: %+7.2f%%   调仓 %2d 次, 单笔胜率 %.0f%%, 最好 %+.1f%%, 最差 %+.1f%%"
          % (y, v * 100, len(sub) // 3, (sub["ret_pct"] > 0).mean() * 100,
             sub["ret_pct"].max(), sub["ret_pct"].min()))

print("\n【最好 / 最差 10 笔】")
t = trades.sort_values("ret_pct", ascending=False)
print("  最好:")
for _, x in t.head(10).iterrows():
    print("    %s → %s  %-8s %+7.2f%%" % (x.rebal_date, x.exit_date, x.etf, x.ret_pct))
print("  最差:")
for _, x in t.tail(10).iloc[::-1].iterrows():
    print("    %s → %s  %-8s %+7.2f%%" % (x.rebal_date, x.exit_date, x.etf, x.ret_pct))

print("\n【逐次调仓明细】")
cur_y = None
for x in rebal_log:
    if x["date"].year != cur_y:
        cur_y = x["date"].year
        print("\n  ── %d 年 " % cur_y + "─" * 60)
    parts = ["%s %+.1f%%" % (p[0], p[3] * 100) for p in x["pos"]]
    print("  %s→%s | %-46s | 组合 %+6.2f%% (费后 %+6.2f%%) | 净值 %.3f"
          % (x["date"].strftime("%m-%d"), x["exit"].strftime("%m-%d"),
             "  ".join(parts), x["gross"] * 100, x["net"] * 100, x["nav"]))
print("\n逐笔明细已导出: audit/out/trades_cap75_equal_R10.csv (%d 行)" % len(trades))
