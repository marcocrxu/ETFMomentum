# -*- coding: utf-8 -*-
"""N-19: min_hold 细网格 1~10 × 六起点 —— 平台还是尖峰? (严格执行 audit/n_minhold_protocol.md)

判定 A「是平台」: (1) mh∈{1..5} 至少 4 档 ≥ 该起点基线; (2) 十档年化中位数 ≥ 基线;
                 (3) 无孤立尖峰(某档比左右相邻同时低 >1.0pp)
判定 B「可作候选」: (1) mh=2 至少 5/6 起点 ≥ 基线; (2) lag1_11bp ≥ 28.13%;
                   (3) mh∈{1,2,3} 六起点中位年化差 ≥ −1.0pp

用法: .venv/bin/python audit/n_minhold_ablation.py
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

MHS = list(range(1, 11))
STARTS = ["2014-01-02", "2016-01-04", "2018-01-02", "2020-01-02", "2022-01-04", "2024-01-02"]
END = "2026-08-20"
STRESS = ("2014-01-02", "2026-09-24")
BASE_STRESS = 28.13


def run(mh, start, end=END):
    r = br.run_backtest(dict(universe="纳指,黄金,创业", start_date=start, end_date=end,
                             min_hold=mh, confirm_days=2))
    return dict(ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"], trades=r["trade_count"])


def stress(mh):
    res, cap = run_capture(dict(universe="纳指,黄金,创业", start_date=STRESS[0],
                                end_date=STRESS[1], min_hold=mh, confirm_days=2))
    cfg, acts, pm, names = cap["cfg"], cap["actions"], cap["price_mat"], cap["names"]
    if np.max(np.abs(replay_nav(acts, pm, names, cfg)
                     - np.array(res["nav_log"], dtype=float))) > 1e-9:
        raise RuntimeError("replay 不自洽")
    c = dataclasses.replace(cfg, slippage=0.0010)
    nav = np.asarray(replay_nav(shift_actions(acts, pm.shape[1]), pm, names, c), dtype=float)
    nav = nav / nav[0]; r = nav[1:] / nav[:-1] - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float(nav[-1] ** (244.0 / len(r)) - 1)
    return dict(ann=ann * 100, dd=dd * 100, cal=(ann / abs(dd) if dd < 0 else float("nan")))


def main():
    tab = {}
    for s in STARTS:
        for mh in MHS:
            tab.setdefault(mh, {})[s] = run(mh, s)
        print("起点 %s 完成" % s, flush=True)

    print("\n===== 年化 %% 表 (行=min_hold, 列=起点; 括号内为相对该起点基线的差 pp) =====")
    hdr = "  mh  " + "".join("%22s" % s[:7] for s in STARTS) + "%12s" % "中位差"
    print(hdr)
    med = {}
    for mh in MHS:
        cells, diffs = [], []
        for s in STARTS:
            base = tab[10][s]["ann"]
            d = tab[mh][s]["ann"] - base
            diffs.append(d)
            cells.append("%7.2f(%+6.2f)" % (tab[mh][s]["ann"], d))
        med[mh] = float(np.median(diffs))
        print("  %-4d" % mh + "".join("%22s" % c for c in cells) + "%12.2f" % med[mh])

    # ── 判定 A ──
    print("\n===== 判定 A「是平台」=====")
    A1 = {}
    for s in STARTS:
        base = tab[10][s]["ann"]
        n_ge = sum(1 for mh in (1, 2, 3, 4, 5) if tab[mh][s]["ann"] >= base)
        A1[s] = n_ge
    a1 = all(v >= 4 for v in A1.values())
    print("  A1 mh∈{1..5} 中 ≥ 基线的档数: %s  => %s"
          % ({s[:7]: v for s, v in A1.items()}, "✅" if a1 else "❌"))
    a2 = all(med[10] == 0.0 or float(np.median([tab[mh][s]["ann"] - tab[10][s]["ann"]
                                                for mh in MHS])) >= 0 for s in STARTS)
    med10 = {s: float(np.median([tab[mh][s]["ann"] - tab[10][s]["ann"] for mh in MHS]))
             for s in STARTS}
    a2 = all(v >= 0 for v in med10.values())
    print("  A2 十档中位数 ≥ 基线 (逐起点): %s  => %s"
          % ({s[:7]: round(v, 2) for s, v in med10.items()}, "✅" if a2 else "❌"))
    spikes = []
    for s in STARTS:
        for i in range(1, 9):
            mh = MHS[i]
            lo, hi = tab[mh - 1][s]["ann"], tab[mh + 1][s]["ann"]
            if tab[mh][s]["ann"] < min(lo, hi) - 1.0:
                spikes.append((mh, s, round(tab[mh][s]["ann"] - min(lo, hi), 2)))
    a3 = len(spikes) == 0
    print("  A3 无孤立尖峰 (某档比左右相邻同时低 >1.0pp): %s  => %s"
          % (spikes if spikes else "无", "✅" if a3 else "❌"))
    A = a1 and a2 and a3
    print("  **判定 A = %s**" % ("成立: 是平台" if A else "不成立: 凹陷/尖峰"))

    # ── 判定 B ──
    print("\n===== 判定 B「可作候选改默认」=====")
    wins2 = sum(1 for s in STARTS if tab[2][s]["ann"] >= tab[10][s]["ann"])
    b1 = wins2 >= 5
    print("  B1 mh=2 在 %d/6 起点 ≥ 基线  => %s" % (wins2, "✅" if b1 else "❌"))
    st2 = stress(2); st10 = stress(10)
    b2 = st2["ann"] >= BASE_STRESS
    print("  B2 mh=2 lag1_11bp %.2f%% vs 基线 %.2f%%  => %s"
          % (st2["ann"], st10["ann"], "✅" if b2 else "❌"))
    nb = [tab[mh][s]["ann"] - tab[10][s]["ann"] for mh in (1, 2, 3) for s in STARTS]
    nb_med = float(np.median(nb))
    b3 = nb_med >= -1.0
    print("  B3 mh∈{1,2,3} 六起点中位差 %+.2fpp  => %s" % (nb_med, "✅" if b3 else "❌"))
    B = b1 and b2 and b3
    print("  **判定 B = %s**" % ("成立: 可作候选" if B else "不成立"))

    print("\n===== lag1_11bp 压力窗: mh=10(基线) %.2f%% / mh=2 %.2f%% =====" % (st10["ann"], st2["ann"]))
    out = dict(protocol="audit/n_minhold_protocol.md", mhs=MHS, starts=STARTS, end=END, table=tab,
               median_diff={str(k): v for k, v in med.items()},
               A1=A1, A1_pass=bool(a1), A2=med10, A2_pass=bool(a2), A3_spikes=spikes, A3_pass=bool(a3),
               judge_A=bool(A), B1_wins=int(wins2), B1_pass=bool(b1), stress2=st2, stress10=st10,
               B2_pass=bool(b2), B3_median=float(nb_med), B3_pass=bool(b3), judge_B=bool(B))
    p = os.path.join(HERE, "audit", "out", "n_minhold_ablation.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("saved -> %s" % p)


if __name__ == "__main__":
    main()
