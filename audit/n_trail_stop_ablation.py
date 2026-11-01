# -*- coding: utf-8 -*-
"""N-29 移动止损消融（协议见 audit/n_trail_stop_protocol.md，跑前已注册）

面板: 六起点 → 2026-08-20（逐起点自比基线） + lag1_11bp 压力窗(2014-01-02~2026-09-24)
网格: trail_stop ∈ {5, 8, 10, 12}%，基线 = 生产默认(trail_stop=0)

用法: PYTHONPATH=. .venv/bin/python audit/n_trail_stop_ablation.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
import web.bt_runner as br  # noqa: E402
from conftest import replay_nav  # noqa: E402
from audit.n_lag1_stress import run_capture, shift_actions  # noqa: E402

STARTS = ["2014-01-02", "2016-01-04", "2018-01-02", "2020-01-02", "2022-01-04", "2024-01-02"]
END = "2026-08-20"
STRESS = ("2014-01-02", "2026-09-24")
BASE_STRESS = 28.13          # N-20 同口径基线年化(%)
GRID = [5.0, 8.0, 10.0, 12.0]


def run(kw, start, end=END):
    r = br.run_backtest(dict(universe="纳指,黄金,创业", start_date=start, end_date=end, **kw))
    return dict(ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"], trades=r["trade_count"])


def stress(kw):
    res, cap = run_capture(dict(universe="纳指,黄金,创业", start_date=STRESS[0],
                                end_date=STRESS[1], **kw))
    cfg, acts, pm, names = cap["cfg"], cap["actions"], cap["price_mat"], cap["names"]
    if np.max(np.abs(replay_nav(acts, pm, names, cfg)
                     - np.array(res["nav_log"], dtype=float))) > 1e-9:
        raise RuntimeError("replay 不自洽")
    c = dataclasses.replace(cfg, slippage=0.0010)
    nav = np.asarray(replay_nav(shift_actions(acts, pm.shape[1]), pm, names, c), dtype=float)
    nav = nav / nav[0]
    r = nav[1:] / nav[:-1] - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float(nav[-1] ** (244.0 / len(r)) - 1)
    return dict(ann=ann * 100, dd=dd * 100)


def main():
    print("== 基线（六起点）==", flush=True)
    base = {}
    for s in STARTS:
        base[s] = run({}, s)
        print("  %s ann %6.2f dd %7.2f cal %5.2f trades %3d"
              % (s, base[s]["ann"], base[s]["dd"], base[s]["cal"], base[s]["trades"]), flush=True)
    st_base = stress({})
    print("基线 lag1_11bp = %.2f%% (注册值 %.2f)" % (st_base["ann"], BASE_STRESS), flush=True)

    rows = {}
    for g in GRID:
        kw = dict(trail_stop=g)
        res, diffs = {}, {}
        for s in STARTS:
            res[s] = run(kw, s)
            diffs[s] = res[s]["ann"] - base[s]["ann"]
        st = stress(kw)
        wins = sum(1 for s in STARTS if diffs[s] >= 0)
        med = float(np.median(list(diffs.values())))
        worst = min(diffs.values())
        rows[g] = dict(starts=res, diffs=diffs, stress=st, wins=wins, median=med, worst=worst)
        print("trail=%4.1f | " % g
              + " ".join("%s %+.1f" % (s[2:7], diffs[s]) for s in STARTS)
              + " | 胜 %d/6 med %+.2f worst %+.2f | lag1_11bp %.2f%%"
              % (wins, med, worst, st["ann"]), flush=True)

    # 五条门槛
    print("\n== 资格判定（五条全过才是候选）==", flush=True)
    verdict = {}
    for g, r in rows.items():
        gates = {
            "① ≥5/6 起点 ≥ 基线": r["wins"] >= 5,
            "② 无灾难起点(最差 ≥ -3.0pp)": r["worst"] >= -3.0,
            "③ 中位差 ≥ +0.3pp": r["median"] >= 0.3,
            "④ lag1_11bp ≥ %.2f%%" % BASE_STRESS: r["stress"]["ann"] >= BASE_STRESS,
        }
        verdict[g] = gates
        print("trail=%4.1f: %s" % (g, " | ".join("%s %s" % (k, "✓" if v else "✗") for k, v in gates.items())), flush=True)
    # ⑤ 无孤立尖峰
    anns = [np.median(list(rows[g]["diffs"].values())) for g in GRID]
    spike = [abs(anns[i] - anns[i + 1]) > 1.0 for i in range(len(anns) - 1)]
    print("⑤ 相邻档中位差: %s -> %s"
          % (["%.2f" % a for a in anns], "无尖峰" if not any(spike) else "有尖峰 %s" % spike), flush=True)

    out = {"protocol": "n_trail_stop_protocol.md", "starts": STARTS, "end": END,
           "stress_base": st_base, "base": base,
           "rows": {str(g): {"starts": rows[g]["starts"],
                             "diffs": {k: float(v) for k, v in rows[g]["diffs"].items()},
                             "stress": rows[g]["stress"], "wins": int(rows[g]["wins"]),
                             "median": float(rows[g]["median"]), "worst": float(rows[g]["worst"])}
                    for g in GRID},
           "verdicts": {str(g): {k: bool(v) for k, v in verdict[g].items()} for g in GRID}}
    p = os.path.join(HERE, "audit", "out", "n_trail_stop_ablation.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> %s" % p)


if __name__ == "__main__":
    main()
