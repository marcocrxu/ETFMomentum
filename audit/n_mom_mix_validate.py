# -*- coding: utf-8 -*-
"""N-24 阶段 2: 全网格前 20 名 → 六起点面板 + lag1_11bp 验证

严格执行 audit/n_mom_mix_protocol.md §3/§4/§5。
输入: audit/out/n_mom_mix_grid_A.csv (阶段 1 全 1771 格)

用法: .venv/bin/python audit/n_mom_mix_validate.py
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
import web.bt_runner as br  # noqa: E402
from conftest import replay_nav  # noqa: E402
from audit.n_lag1_stress import run_capture, shift_actions  # noqa: E402


class _NoOp:
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return self
    def __setitem__(self, k, v): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


br.plt = _NoOp()
_orig = br.FEED.get_dataframe
_c = {}


def _cg(codes, start, end, *a, **k):
    key = (tuple(codes), start, end)
    if key not in _c:
        _c[key] = _orig(codes, start, end, *a, **k)
    return _c[key]


br.FEED.get_dataframe = _cg

STARTS = ["2014-01-02", "2016-01-04", "2018-01-02", "2020-01-02", "2022-01-04", "2024-01-02"]
END = "2026-08-20"
STRESS = ("2014-01-02", "2026-09-24")
BASE_STRESS = 28.13


def spec_of(r):
    if abs(r.w20 - 1.0) < 1e-9:
        return ""
    return "5:%.2f,10:%.2f,15:%.2f,20:%.2f" % (r.w5, r.w10, r.w15, r.w20)


def run(spec, start, end=END):
    kw = dict(score_mom_mix=spec) if spec else {}
    r = br.run_backtest(dict(universe="纳指,黄金,创业", start_date=start, end_date=end, **kw))
    return dict(ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"], trades=r["trade_count"])


def stress(spec):
    kw = dict(score_mom_mix=spec) if spec else {}
    res, cap = run_capture(dict(universe="纳指,黄金,创业", start_date=STRESS[0],
                                end_date=STRESS[1], **kw))
    cfg, acts, pm, names = cap["cfg"], cap["actions"], cap["price_mat"], cap["names"]
    assert np.max(np.abs(replay_nav(acts, pm, names, cfg)
                         - np.array(res["nav_log"], dtype=float))) < 1e-9
    c = dataclasses.replace(cfg, slippage=0.0010)
    nav = np.asarray(replay_nav(shift_actions(acts, pm.shape[1]), pm, names, c), dtype=float)
    nav = nav / nav[0]; r = nav[1:] / nav[:-1] - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float(nav[-1] ** (244.0 / len(r)) - 1)
    return dict(ann=ann * 100, dd=dd * 100)


def main():
    g = pd.read_csv(os.path.join(HERE, "audit", "out", "n_mom_mix_grid_A.csv"))
    top = g.sort_values("ann", ascending=False).head(20).reset_index(drop=True)
    print("=== 阶段 1 全网格 %d 格; 超过基线(45.51)的 %d 格 ===" % (len(g), (g.ann > 45.51).sum()))
    print("阶段 1 第一名: w5=%.2f w10=%.2f w15=%.2f w20=%.2f ann=%.2f dd=%.2f cal=%.2f"
          % tuple(top.iloc[0][["w5", "w10", "w15", "w20", "ann", "dd", "cal"]]))

    base = {s: run("", s) for s in STARTS}
    st_base = stress("")
    print("\n基线六起点: " + " ".join("%s=%.2f%%" % (s[:7], base[s]["ann"]) for s in STARTS)
          + " | lag1_11bp=%.2f%%" % st_base["ann"])

    rows = []
    for _, r in top.iterrows():
        sp = spec_of(r)
        res = {s: run(sp, s) for s in STARTS}
        st = stress(sp)
        diffs = {s: res[s]["ann"] - base[s]["ann"] for s in STARTS}
        v = dict(
            spec=sp or "20:1.00(基线)", w5=r.w5, w10=r.w10, w15=r.w15, w20=r.w20,
            stage1_ann=r.ann, stages={s: res[s]["ann"] for s in STARTS}, diffs=diffs,
            wins=sum(1 for d in diffs.values() if d >= 0),
            median=float(np.median(list(diffs.values()))), worst=float(min(diffs.values())),
            stress_ann=st["ann"], stress_dd=st["dd"])
        v["g1"] = v["wins"] >= 5
        v["g2"] = v["worst"] >= -3.0
        v["g3"] = v["median"] >= 0.3
        v["g4"] = v["stress_ann"] >= BASE_STRESS
        v["qualified"] = bool(v["g1"] and v["g2"] and v["g3"] and v["g4"])
        rows.append(v)
        print("%-34s 阶段1 %6.2f%% | 胜 %d/6 中位 %+6.2f 最差 %+7.2f | lag1_11bp %6.2f%% | %s"
              % (v["spec"], r.ann, v["wins"], v["median"], v["worst"], v["stress_ann"],
                 "合格" if v["qualified"] else "否决"), flush=True)

    nq = sum(1 for v in rows if v["qualified"])
    print("\n=== 阶段 2 结果: %d / %d 合格 ===" % (nq, len(rows)))
    # 5% 权重步长下的邻居检查 (把 5% 从一维搬到另一维)
    print("\n=== 邻域检查 (前 20 名之间互为 5% 单步邻居的对) ===")
    idx = {(round(v["w5"], 2), round(v["w10"], 2), round(v["w15"], 2), round(v["w20"], 2)): v
           for v in rows}
    for a in idx:
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                b = list(a); b[i] -= 0.05; b[j] += 0.05
                b = tuple(round(x, 2) for x in b)
                if b in idx and b > a:
                    print("  %s ↔ %s : 阶段2中位差 %+.2fpp"
                          % (a, b, idx[b]["median"] - idx[a]["median"]))

    json.dump(dict(stage1_n=len(g), stage1_over_base=int((g.ann > 45.51).sum()),
                   base={s: base[s] for s in STARTS}, base_stress=st_base,
                   rows=rows, qualified=nq),
              open(os.path.join(HERE, "audit", "out", "n_mom_mix_validate.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> audit/out/n_mom_mix_validate.json")


if __name__ == "__main__":
    main()
