# -*- coding: utf-8 -*-
"""N-20: 打分平滑 (P1-1) + 波动化止损 (P1-3) —— 严格执行 audit/n_smooth_stopvol_protocol.md

S 组: score_smooth ∈ {2,3,4,5,7,10}
V 组: stop_vol_k ∈ {2,2.5,3,3.5,4} × stop_vol_src ∈ {current20, prior20}
面板: 六起点 → 2026-08-20 (逐起点自比基线) + lag1_11bp(2014-01-02~2026-09-24)
判定: 协议 §资格判定五条; 处置: 每个机制分别判

用法: .venv/bin/python audit/n_smooth_stopvol_ablation.py
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
BASE_STRESS = 28.13

S_GRID = [dict(score_smooth=n) for n in (2, 3, 4, 5, 7, 10)]
V_GRID = [dict(stop_vol_on=True, stop_vol_k=k, stop_vol_src=src)
          for k in (2.0, 2.5, 3.0, 3.5, 4.0) for src in ("current20", "prior20")]


def tag(kw):
    if "score_smooth" in kw:
        return "S_smooth%d" % kw["score_smooth"]
    return "V_k%.1f_%s" % (kw["stop_vol_k"], kw["stop_vol_src"])


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
    nav = nav / nav[0]; r = nav[1:] / nav[:-1] - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float(nav[-1] ** (244.0 / len(r)) - 1)
    return dict(ann=ann * 100, dd=dd * 100)


def judge(name, base, res, st, st_base):
    """按协议五条判定。base/res = {start: ann}; st/st_base = 压力窗年化"""
    wins, diffs = 0, []
    for s in STARTS:
        d = res[s]["ann"] - base[s]["ann"]
        diffs.append(d)
        if d >= 0:
            wins += 1
    med = float(np.median(diffs))
    worst = float(min(diffs))
    g = {"① ≥5/6 起点 ≥ 基线": wins >= 5,
         "② 无灾难起点(最差 ≥ −3.0pp)": worst >= -3.0,
         "③ 中位差 ≥ +0.3pp": med >= 0.3,
         "④ lag1_11bp ≥ 28.13%": st["ann"] >= BASE_STRESS,
         "⑤ 无孤立尖峰(同网格相邻档差 ≤1.0pp)": None}
    return dict(gates=g, wins=wins, median=med, worst=worst)


def main():
    base = {s: run({}, s) for s in STARTS}
    st_base = stress({})
    print("基线 (六起点年化): " + " ".join("%s=%.2f%%" % (s[:7], base[s]["ann"]) for s in STARTS))
    print("基线 lag1_11bp = %.2f%%" % st_base["ann"])

    rows = {}
    for kw in S_GRID + V_GRID:
        t = tag(kw)
        res = {s: run(kw, s) for s in STARTS}
        st = stress(kw)
        rows[t] = dict(params=kw, starts=res, stress=st,
                       diffs={s: res[s]["ann"] - base[s]["ann"] for s in STARTS})
        print("%-20s " % t + " ".join("%+6.2f" % rows[t]["diffs"][s] for s in STARTS)
              + " | med %+6.2f worst %+6.2f | lag1_11bp %6.2f%%" %
              (rows[t]["diffs"][s] for s in STARTS) if False else
              "%-20s " % t + " ".join("%+6.2f" % rows[t]["diffs"][s] for s in STARTS)
              + " | med %+6.2f worst %+6.2f | lag1_11bp %6.2f%%"
              % (np.median(list(rows[t]["diffs"].values())),
                 min(rows[t]["diffs"].values()), st["ann"]), flush=True)

    # 尖峰检查 (同网格内相邻档)
    def spikes(grid):
        ts = [tag(k) for k in grid]
        med = {t: float(np.median(list(rows[t]["diffs"].values()))) for t in ts}
        out = []
        for i in range(1, len(ts) - 1):
            if med[ts[i]] < min(med[ts[i - 1]], med[ts[i + 1]]) - 1.0:
                out.append((ts[i], round(med[ts[i]] - min(med[ts[i - 1]], med[ts[i + 1]]), 2)))
        return out

    verdicts = {}
    for name, grid in (("S_打分平滑", S_GRID), ("V_波动化止损", V_GRID)):
        best = None
        for kw in grid:
            t = tag(kw)
            j = judge(name, base, rows[t]["starts"], rows[t]["stress"], st_base)
            j["gates"]["⑤ 无孤立尖峰(同网格相邻档差 ≤1.0pp)"] = True  # 网格级, 见下
            j["params"] = kw
            verdicts[t] = j
            if best is None or j["median"] > verdicts[best]["median"]:
                best = t
        sp = spikes(grid)
        for t in [tag(k) for k in grid]:
            verdicts[t]["gates"]["⑤ 无孤立尖峰(同网格相邻档差 ≤1.0pp)"] = (len(sp) == 0)
        verdicts[name + "_best"] = best
        print("\n===== %s =====" % name)
        for kw in grid:
            t = tag(kw)
            v = verdicts[t]
            ok = all(v["gates"].values())
            verdicts[t]["qualified"] = bool(ok)
            print("  %-20s %s | 胜 %d/6 中位 %+.2f 最差 %+.2f => %s"
                  % (t, " ".join(("✓" if x else "✗") + k[0] for k, x in v["gates"].items()),
                     v["wins"], v["median"], v["worst"], "合格" if ok else "否决"))
        print("  孤立尖峰: %s" % (sp if sp else "无"))
        print("  最优档 = %s" % best)

    out = dict(protocol="audit/n_smooth_stopvol_protocol.md", starts=STARTS, end=END,
               base={s: base[s] for s in STARTS}, stress_base=st_base,
               rows={t: dict(params=rows[t]["params"], starts=rows[t]["starts"],
                             stress=rows[t]["stress"], diffs=rows[t]["diffs"]) for t in rows},
               verdicts={k: ({kk: (vv if not isinstance(vv, dict) else
                                   {a: bool(b) for a, b in vv.items()})
                              for kk, vv in v.items()} if isinstance(v, dict) else v)
                         for k, v in verdicts.items()})
    p = os.path.join(HERE, "audit", "out", "n_smooth_stopvol_ablation.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> %s" % p)


if __name__ == "__main__":
    main()
