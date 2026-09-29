# -*- coding: utf-8 -*-
"""N-14: WLS 排名器消融 —— 严格执行 audit/n_wls_protocol.md 的事前注册

网格: wls_window ∈ {20,25,30} × wls_smooth ∈ {1,2,3,4} = 12 格 (不追加)。
窗口: 三个十年窗 (close_1bp) + lag1_11bp 执行压力窗 (2014-01-02~2026-09-24)。
门槛: 见协议 §5; 处置见 §6。**必须披露全部 12 格, 不得只报最优。**

用法: .venv/bin/python audit/n_wls_ablation.py
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

BASELINE = {
    "默认窗": (45.51, -18.08, 2.52, 253),
    "2015长窗": (45.74, -21.09, 2.17, 284),
    "2020短窗": (48.36, -19.62, 2.47, 170),
}
WINDOWS = [("默认窗", "2016-07-07", "2026-08-20"),
           ("2015长窗", "2015-01-05", "2026-08-20"),
           ("2020短窗", "2020-01-01", "2026-08-20")]
STRESS = ("2014-01-02", "2026-09-24")
BASE_STRESS = (28.13, -29.94, 0.94)      # N-13 已测的 lag1_11bp 基线
GRID = [(w, s) for w in (20, 25, 30) for s in (1, 2, 3, 4)]


def stress_stats(params):
    """跑一次 2014-2026, 用 replay-shift 出 lag1_11bp 口径。"""
    res, cap = run_capture(params)
    cfg, acts, pm, names = cap["cfg"], cap["actions"], cap["price_mat"], cap["names"]
    rep = replay_nav(acts, pm, names, cfg)
    if np.max(np.abs(rep - np.array(res["nav_log"], dtype=float))) > 1e-9:
        raise RuntimeError("replay 与引擎不自洽")
    c = dataclasses.replace(cfg, slippage=0.0010)
    nav = np.asarray(replay_nav(shift_actions(acts, pm.shape[1]), pm, names, c), dtype=float)
    nav = nav / nav[0]
    r = nav[1:] / nav[:-1] - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float(nav[-1] ** (244.0 / len(r)) - 1)
    return ann * 100, dd * 100, (ann / abs(dd) if dd < 0 else float("nan"))


def main():
    rows = []
    for w, s in GRID:
        row = {"wls_window": w, "wls_smooth": s, "windows": {}}
        for tag, a, b in WINDOWS:
            p = dict(universe="纳指,黄金,创业", start_date=a, end_date=b,
                     score_wls=True, wls_window=w, wls_smooth=s)
            r = br.run_backtest(p)
            row["windows"][tag] = dict(ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"],
                                       trades=r["trade_count"])
        a, b = STRESS
        ann, dd, cal = stress_stats(dict(universe="纳指,黄金,创业", start_date=a, end_date=b,
                                         score_wls=True, wls_window=w, wls_smooth=s))
        row["stress_lag1_11bp"] = dict(ann=ann, dd=dd, cal=cal)
        rows.append(row)
        print("wls%-3d smooth%d | 默认 %6.2f%%/%6.2f%%/%5.2f | 长窗 %6.2f%% | 短窗 %6.2f%% "
              "| lag1_11bp %6.2f%%/%7.2f%%/%5.2f"
              % (w, s, row["windows"]["默认窗"]["ann"], row["windows"]["默认窗"]["dd"],
                 row["windows"]["默认窗"]["cal"], row["windows"]["2015长窗"]["ann"],
                 row["windows"]["2020短窗"]["ann"], ann, dd, cal), flush=True)

    # ── 按协议 §5 逐条判定 ──
    print("\n===== 事前注册门槛逐条判定 =====")
    def gate(r):
        g = {}
        g["① 三窗皆不低基线"] = all(
            r["windows"][t]["ann"] >= BASELINE[t][0] for t, _, _ in WINDOWS)
        g["② 主窗 Calmar >= 2.52"] = r["windows"]["默认窗"]["cal"] >= 2.52
        g["③ 主窗回撤 <= 19.08%"] = r["windows"]["默认窗"]["dd"] >= -19.08
        g["④ lag1_11bp 年化>=28.13 且回撤>=-30.94"] = (
            r["stress_lag1_11bp"]["ann"] >= BASE_STRESS[0]
            and r["stress_lag1_11bp"]["dd"] >= BASE_STRESS[1] - 1.0)
        return g

    best = max(rows, key=lambda r: r["windows"]["默认窗"]["ann"])
    # 邻域按**网格索引**相邻(窗口轴步长是 5, 不是 1); 修正 2026-09-29 的索引 bug
    WS, SS = (20, 25, 30), (1, 2, 3, 4)
    bi, bj = WS.index(best["wls_window"]), SS.index(best["wls_smooth"])
    nb = []
    for r in rows:
        if r is best:
            continue
        dw = abs(WS.index(r["wls_window"]) - bi)
        ds = abs(SS.index(r["wls_smooth"]) - bj)
        if dw <= 1 and ds <= 1:
            nb.append(r["windows"]["默认窗"]["ann"] - best["windows"]["默认窗"]["ann"])
    nb_med = float(np.median(nb)) if nb else float("nan")

    verdict = {}
    for r in rows:
        g = gate(r)
        g["⑤ 无孤峰(邻域中位年化差 >= -1.5pp)"] = (r is best) or (nb_med >= -1.5)
        ok = all(g.values())
        verdict["wls%d_s%d" % (r["wls_window"], r["wls_smooth"])] = dict(
            gates={k: bool(v) for k, v in g.items()}, qualified=bool(ok))
    for k, v in verdict.items():
        print("  %-12s %s  => %s" % (k, " ".join(("✓" if x else "✗") + n[:2] for n, x in v["gates"].items()),
                                     "合格" if v["qualified"] else "否决"))
    print("  最优格 %s 的邻域中位年化差 = %+.2fpp (n=%d)"
          % ("wls%d_s%d" % (best["wls_window"], best["wls_smooth"]), nb_med, len(nb)))
    print("\n合格格数 = %d / 12" % sum(1 for v in verdict.values() if v["qualified"]))

    out = dict(protocol="audit/n_wls_protocol.md", grid=GRID, rows=rows,
               baseline=BASELINE, baseline_stress=BASE_STRESS, verdict=verdict,
               best_cell=[best["wls_window"], best["wls_smooth"]], neighbor_median_pp=nb_med,
               qualified_count=sum(1 for v in verdict.values() if v["qualified"]))
    path = os.path.join(HERE, "audit", "out", "n_wls_ablation.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("saved -> %s" % path)


if __name__ == "__main__":
    main()
