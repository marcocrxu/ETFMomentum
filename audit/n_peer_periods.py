# -*- coding: utf-8 -*-
"""N-10: 我方策略 vs 同行 V12-R2 —— 分时段对比 (逐年 / 周期分段 / 滚动窗)

口径:
  共同区间 2014-01-02 ~ 2026-09-24 (我方池数据够; 同期双方均按收盘成交、单边万一)。
  我方 = web/bt_runner.py 默认基线 (纳指/黄金/创业, QQQ 0.7 混合开);
  对方 = v12_r2 冻结独立复算逐日路径 (results/audit/reference_paths.npz, 与冻结矩阵逐值一致),
         主口径 close_1bp。对方另有 close_11bp / lag1_* 压力情景。
  期间收益为**累计复利**, 不做年化折算 (避免 244/252 口径混淆); 滚动窗另标年化(244)。

用法: .venv/bin/python audit/n_peer_periods.py
"""
import json
import os
import sys
from bisect import bisect_left, bisect_right

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.bt_runner import run_backtest  # noqa: E402

PEER = "/home/ecs-user/peer_artifacts/v12_r2"
START, END = "2014-01-02", "2026-09-24"
SESSIONS = 244.0

# 与展示页 §2 完全相同的周期分段 (web/static/archive_data.js CYCLE_PHASES), 末段延到 09-24
PHASES = [
    ("2014-01-01", "2014-06-30", "熊末磨底"),
    ("2014-07-01", "2015-06-12", "第一轮牛·杠杆牛"),
    ("2015-06-15", "2016-02-29", "股灾+熔断"),
    ("2016-03-01", "2018-01-31", "慢牛/价值"),
    ("2018-02-01", "2019-01-31", "A股熊"),
    ("2019-02-01", "2021-02-28", "结构牛"),
    ("2021-03-01", "2024-09-30", "长熊"),
    ("2024-10-01", "2026-09-24", "第二轮牛"),
]

ROLES = {
    "V12-R2(主推送)": "v12r2_08b522c2e1aa291f054d",
    "V12-C": "v12r2_55420df78993194e846c",
    "V10-H": "v12r2_2b63e64ab0b9e24b8cab",
    "V9.2+": "v12r2_c03a5dc306d9bf5b3e82",
}


def our_series(slippage=0.0):
    r = run_backtest(dict(universe="纳指,黄金,创业", start_date=START, end_date=END,
                          slippage=slippage))
    dates = list(r["dates"])
    nav = np.array(r["nav_log"], dtype=float)
    assert len(dates) == len(nav), "dates/nav_log 长度不一致 %d/%d" % (len(dates), len(nav))
    assert r["checks"] == ["OK"], r["checks"]
    keep = [i for i, d in enumerate(dates) if START <= d <= END]
    dates = [dates[i] for i in keep]
    nav = nav[keep]
    return dates, nav / nav[0], dict(ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"],
                                     trades=r["trade_count"], range=r["data_range"])


def peer_series(scen="close_1bp"):
    meta = json.load(open(os.path.join(PEER, "results", "path_metadata.json")))
    dates = list(meta["dates"])
    out = {}
    with np.load(os.path.join(PEER, "results", "ref_paths.npz"), allow_pickle=False) as z:
        for role, cid in ROLES.items():
            rets = z["%s__%s__returns" % (cid, scen)]
            out[role] = np.concatenate([[1.0], np.cumprod(1.0 + rets)])
    return dates, out


def compound(nav, dates, a, b):
    """[a,b] 区间累计收益 (小数); 无数据返回 None"""
    lo, hi = bisect_left(dates, a), bisect_right(dates, b)
    if hi - lo < 2:
        return None
    return nav[hi - 1] / nav[lo] - 1.0


def ann(nav, dates, a, b):
    lo, hi = bisect_left(dates, a), bisect_right(dates, b)
    n = hi - lo - 1
    if n < 20:
        return None
    tot = nav[hi - 1] / nav[lo]
    return tot ** (SESSIONS / n) - 1.0


def main():
    od, on, oinfo = our_series()
    pd_, peers = peer_series()
    common = sorted(set(od) & set(pd_))
    print("我方 %s (%d 天) | 对方 %s (%d 天) | 共同 %d 天 (%s ~ %s)"
          % (START, len(od), pd_[0], len(pd_), len(common), common[0], common[-1]))

    # 对齐到共同日期轴
    oi = {d: i for i, d in enumerate(od)}
    pi = {d: i for i, d in enumerate(pd_)}
    # 重采样成对齐序列 (只用共同日期)
    o_nav = np.array([on[oi[d]] for d in common])
    p_nav = {k: np.array([v[pi[d]] for d in common]) for k, v in peers.items()}

    res = {"period": [common[0], common[-1]], "days": len(common),
           "our_range": od[0] + "~" + od[-1],
           "yearly": {}, "phases": {}, "rolling3y": {}, "summary": {}}

    # ── 逐年 ──
    years = sorted({d[:4] for d in common})
    print("\n===== 逐年累计收益 =====")
    hdr = "%-6s %10s %10s %10s %10s %10s" % ("年份", "我方", "V12-R2", "V12-C", "V10-H", "V9.2+")
    print(hdr)
    for y in years:
        a, b = y + "-01-01", y + "-12-31"
        row = {"ours": compound(o_nav, common, a, b)}
        for role in ROLES:
            row[role] = compound(p_nav[role], common, a, b)
        res["yearly"][y] = row
        print("%-6s %9s%% %9s%% %9s%% %9s%% %9s%%" % (
            y, *["%.2f" % (row[k] * 100) if row[k] is not None else "—"
                 for k in ["ours"] + list(ROLES)]))

    # ── 周期分段 ──
    print("\n===== 周期分段累计收益 (与我方 §2 同分段) =====")
    print("%-26s %10s %10s %10s" % ("周期段", "我方", "V12-R2", "V10-H"))
    for a, b, label in PHASES:
        row = {"ours": compound(o_nav, common, a, b)}
        for role in ROLES:
            row[role] = compound(p_nav[role], common, a, b)
        res["phases"][label] = row
        print("%-26s %9s%% %9s%% %9s%%" % (
            label,
            *["%.1f" % (row[k] * 100) if row[k] is not None else "—"
              for k in ("ours", "V12-R2(主推送)", "V10-H")]))

    # ── 滚动窗 (3年 / 1年) ──
    for label, win in (("rolling3y", int(SESSIONS * 3)), ("rolling1y", int(SESSIONS))):
        ours_a, r2_a, ends = [], [], []
        for i in range(0, len(common) - win):
            j = i + win
            ours_a.append(o_nav[j] / o_nav[i] - 1.0)
            r2_a.append(p_nav["V12-R2(主推送)"][j] / p_nav["V12-R2(主推送)"][i] - 1.0)
            ends.append(common[j])
        ours_a, r2_a = np.array(ours_a), np.array(r2_a)
        wins = int((ours_a > r2_a).sum())
        gaps = (ours_a - r2_a) * 100
        res[label] = dict(n=len(ours_a), our_wins=wins, peer_wins=len(ours_a) - wins,
                          our_win_rate=wins / max(len(ours_a), 1),
                          median_gap_pp=float(np.median(gaps)),
                          p10=float(np.percentile(gaps, 10)), p90=float(np.percentile(gaps, 90)),
                          worst=float(gaps.min()), best=float(gaps.max()),
                          our_win_years=sorted({d[:4] for d, w in zip(ends, ours_a > r2_a) if w}),
                          peer_win_years=sorted({d[:4] for d, w in zip(ends, ours_a > r2_a) if not w}))
        print("\n===== 滚动 %s (%d 窗) =====" % (label[-2:], len(ours_a)))
        print("我方胜 %d / 对方胜 %d (我方胜率 %.0f%%)"
              % (wins, len(ours_a) - wins, 100 * wins / max(len(ours_a), 1)))
        print("我方 - V12-R2 超额: 中位 %+.1fpp, P10 %+.1fpp, P90 %+.1fpp, 最差 %+.1fpp, 最好 %+.1fpp"
              % (np.median(gaps), np.percentile(gaps, 10), np.percentile(gaps, 90),
                 gaps.min(), gaps.max()))
        print("我方获胜的窗口终止年份: %s" % res[label]["our_win_years"])

    # ── 同成本口径 (单边 11bp) 逐年对照: 我方 ideal-close/11bp vs 对方 close_11bp ──
    print("\n===== 同成本口径 (单边 11bp, 均为理想同收盘) 逐年 =====")
    _, on11, _ = our_series(slippage=0.0010)
    o11 = np.array([on11[oi[d]] for d in common])
    _, peers11 = peer_series("close_11bp")
    p11 = {k: np.array([v[pi[d]] for d in common]) for k, v in peers11.items()}
    res["yearly_11bp"] = {}
    print("%-6s %11s %11s %8s %11s" % ("年份", "我方(11bp)", "V12-R2(11bp)", "我方-对方", "V10-H(11bp)"))
    for y in years:
        a, b = y + "-01-01", y + "-12-31"
        row = {"ours": compound(o11, common, a, b),
               "V12-R2": compound(p11["V12-R2(主推送)"], common, a, b),
               "V10-H": compound(p11["V10-H"], common, a, b)}
        row["gap"] = (row["ours"] - row["V12-R2"]) if None not in (row["ours"], row["V12-R2"]) else None
        res["yearly_11bp"][y] = row
        print("%-6s %10s%% %10s%% %7spp %10s%%" % (
            y, *["%.2f" % (row[k] * 100) if row[k] is not None else "—"
                 for k in ("ours", "V12-R2")],
            ("%+.1f" % (row["gap"] * 100)) if row["gap"] is not None else "—",
            "%.2f" % (row["V10-H"] * 100) if row["V10-H"] is not None else "—"))
    wins11 = sum(1 for y in years if res["yearly_11bp"][y]["gap"] is not None
                 and res["yearly_11bp"][y]["gap"] > 0)
    print("单边11bp下, 逐年我方胜 %d / %d 年" % (wins11, len(years)))
    n11 = len(years) - years.count(list(years)[-1])  # 2026 为部分年
    fa = ann(o11, common, common[0], common[-1]); fb = ann(p11["V12-R2(主推送)"], common, common[0], common[-1])
    print("全期(11bp): 我方年化 %.2f%% / 对方 %.2f%% (差 %+.2fpp); 我方回撤 %.2f%% / 对方 %.2f%%"
          % (fa * 100, fb * 100, (fa - fb) * 100,
             (o11 / np.maximum.accumulate(o11) - 1).min() * 100,
             (p11["V12-R2(主推送)"] / np.maximum.accumulate(p11["V12-R2(主推送)"]) - 1).min() * 100))
    res["full_11bp"] = dict(ours_ann=float(fa), peer_ann=float(fb),
                            ours_dd=float((o11 / np.maximum.accumulate(o11) - 1).min()),
                            peer_dd=float((p11["V12-R2(主推送)"] /
                                           np.maximum.accumulate(p11["V12-R2(主推送)"]) - 1).min()),
                            our_year_wins=wins11, years=len(years))

    # ── 汇总 ──
    for label, nav, dates in [("ours", o_nav, common)] + \
                             [(k, p_nav[k], common) for k in ROLES]:
        res["summary"][label] = dict(
            total=float(nav[-1] / nav[0] - 1.0),
            ann=float(ann(nav, dates, common[0], common[-1])),
            max_dd=float((nav / np.maximum.accumulate(nav) - 1.0).min()))
    path = "/home/ecs-user/etf_strategy/audit/out/n_peer_periods.json"
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("\n全期 (2014-01-02 ~ %s):" % common[-1])
    for k, v in res["summary"].items():
        print("  %-16s 累计 %+10.1f%%  年化(244) %6.2f%%  回撤 %6.2f%%"
              % (k, v["total"] * 100, v["ann"] * 100, v["max_dd"] * 100))
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
