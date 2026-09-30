# -*- coding: utf-8 -*-
"""N-18: 把"果断程度"调到她的水平 —— 严格执行 audit/n_decisive_protocol.md

G1: min_hold{2,3,5,7} × confirm_days{1,2} = 8 格
G2: 底座 min_hold=2/confirm_days=1, min_gap{1.0,2.0} × fast_min_hold{0,2} = 4 格
窗口: 三个十年窗(close_1bp) + lag1_11bp(2014-01-02~2026-09-24)
门槛: 协议 §4 四条; 处置: §5 (含 N-14 教训条款)

用法: .venv/bin/python audit/n_decisive_ablation.py
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

BASE = {"默认窗": (45.51, -18.08, 2.52), "2015长窗": (45.74, -21.09, 2.17),
        "2020短窗": (48.36, -19.62, 2.47)}
WINDOWS = [("默认窗", "2016-07-07", "2026-08-20"),
           ("2015长窗", "2015-01-05", "2026-08-20"),
           ("2020短窗", "2020-01-01", "2026-08-20")]
STRESS = ("2014-01-02", "2026-09-24")
BASE_STRESS = (28.13, -29.94, 0.94)

G1 = [dict(min_hold=mh, confirm_days=cd) for mh in (2, 3, 5, 7) for cd in (1, 2)]
G2 = [dict(min_hold=2, confirm_days=1, min_gap=g, fast_min_hold=f)
      for g in (1.0, 2.0) for f in (0, 2)]
GRID = G1 + G2


def tag_of(kw):
    s = "mh%d_cd%d" % (kw["min_hold"], kw["confirm_days"])
    if "min_gap" in kw:
        s += "_gap%.0f_fmh%d" % (kw["min_gap"], kw["fast_min_hold"])
    return s


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
    return ann * 100, dd * 100, (ann / abs(dd) if dd < 0 else float("nan"))


def main():
    rows = []
    for kw in GRID:
        row = dict(tag=tag_of(kw), params=kw, windows={})
        for t, a, b in WINDOWS:
            r = br.run_backtest(dict(universe="纳指,黄金,创业", start_date=a, end_date=b, **kw))
            row["windows"][t] = dict(ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"],
                                     trades=r["trade_count"])
        a, dd, cal = stress(kw)
        row["stress"] = dict(ann=a, dd=dd, cal=cal)
        rows.append(row)
        print("%-22s 默认 %6.2f%%/%6.2f%%/%5.2f | 长窗 %6.2f%% | 短窗 %6.2f%% | lag1_11bp %6.2f%%/%7.2f%%/%5.2f"
              % (row["tag"], row["windows"]["默认窗"]["ann"], row["windows"]["默认窗"]["dd"],
                 row["windows"]["默认窗"]["cal"], row["windows"]["2015长窗"]["ann"],
                 row["windows"]["2020短窗"]["ann"], a, dd, cal), flush=True)

    print("\n===== 协议 §4 四条门槛逐条判定 =====")
    verdict = {}
    for row in rows:
        g = {
            "① 三窗皆不低基线": all(row["windows"][t]["ann"] >= BASE[t][0] for t, _, _ in WINDOWS),
            "② 主窗 Calmar>=2.52": row["windows"]["默认窗"]["cal"] >= 2.52,
            "③ 主窗回撤<=19.08%": row["windows"]["默认窗"]["dd"] >= -19.08,
            "④ lag1_11bp 年化>=28.13 且回撤>=-30.94": (row["stress"]["ann"] >= BASE_STRESS[0]
                                                    and row["stress"]["dd"] >= BASE_STRESS[1] - 1.0),
        }
        ok = all(g.values())
        verdict[row["tag"]] = dict(gates={k: bool(v) for k, v in g.items()}, qualified=bool(ok),
                                   params={k: (v if isinstance(v, (int, float, str)) else bool(v))
                                           for k, v in row["params"].items()})
        print("  %-22s %s => %s" % (row["tag"],
                                    " ".join(("✓" if v else "✗") + k[0] for k, v in g.items()),
                                    "合格" if ok else "否决"))
    npass = sum(1 for v in verdict.values() if v["qualified"])
    print("\n合格格数 = %d / %d" % (npass, len(rows)))

    # 教训条款: 主口径输/压力口径赢 的模式是否出现
    trade_off = []
    for row in rows:
        lose_main = row["windows"]["默认窗"]["ann"] < BASE["默认窗"][0]
        win_stress = row["stress"]["ann"] > BASE_STRESS[0]
        win_main = not lose_main
        lose_stress = not win_stress
        if (lose_main and win_stress) or (win_main and lose_stress):
            trade_off.append(dict(tag=row["tag"], lose_main=lose_main, win_stress=win_stress,
                                  main_ann=row["windows"]["默认窗"]["ann"],
                                  stress_ann=row["stress"]["ann"]))
    if trade_off:
        print("\n⚠️ 出现'主口径/压力口径相反'的格 %d 个 (协议 §5 教训条款: 不得据此改判):" % len(trade_off))
        for t in trade_off:
            print("   %-22s 主窗 %+.2fpp  压力 %+.2fpp" % (t["tag"],
                  t["main_ann"] - BASE["默认窗"][0], t["stress_ann"] - BASE_STRESS[0]))
    else:
        print("\n未出现'主口径/压力口径相反'的模式 (所有格两个口径同向)。")

    out = dict(protocol="audit/n_decisive_protocol.md", grid=[tag_of(k) for k in GRID],
               rows=rows, baseline=BASE, baseline_stress=BASE_STRESS, verdict=verdict,
               qualified_count=npass,
               trade_off=[{k: (v if isinstance(v, (int, float, str)) else bool(v))
                           for k, v in t.items()} for t in trade_off])
    p = os.path.join(HERE, "audit", "out", "n_decisive_ablation.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print("saved -> %s" % p)


if __name__ == "__main__":
    main()
