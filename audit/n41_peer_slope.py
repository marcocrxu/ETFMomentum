# -*- coding: utf-8 -*-
"""N-41 移植试验: 把 N-40 的 mom20 斜率门搬到同行 v9 策略上

目的(用户 2026-11-02 提议): 检验"斜率门"是通用 alpha, 还是只对我们三标的小池有效。
口径: 收盘成交(N-23 已证 14:57-15:00 集合竞价 = 收盘价, 零口径差), 不加额外滑点。
基线: v9 官方代码默认参数 (2016-01-01 起: 年化 42.6% / 回撤 -27.0% / 卡玛 1.58 / 206 笔)
"""
import sys, itertools
sys.path.insert(0, '/home/ecs-user/etf_strategy/backtest/v9_official')
import strategy
from market_data import UNIVERSE, fetch_history
from backtest import backtest, report

HIST = {c: fetch_history(c) for c in UNIVERSE}
CAL = [r[0] for r in HIST["510300"]]
STARTS = ["2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]
END = "2026-10-29"

CYB = ("159915",)                 # 创业板(与我们 N-40 完全对应)
ALL = tuple(UNIVERSE.keys())      # 全部(线性, 无标的限制)

def run(**kw):
    for k, v in kw.items():
        setattr(strategy, k, v)
    out = {}
    for ws in STARTS:
        r = backtest(HIST, CAL, start=ws, end=END)
        out[ws] = {"ann": r["ann"] * 100, "dd": r["max_dd"] * 100,
                   "calmar": r["calmar"],
                   "trades": r["switches"]}
    for k in kw:
        setattr(strategy, k, _ORIG[k])
    return out

_ORIG = {k: getattr(strategy, k) for k in
         ("SLOPE_GATE_ON", "SLOPE_GATE_CODES", "SLOPE_GATE_WIN",
          "SLOPE_GATE_MOM_MIN", "SLOPE_GATE_PERSIST")}

print("=== 基线 (默认参数) ===", flush=True)
BASE = run()
for ws in STARTS:
    b = BASE[ws]
    print("  %s 年化 %6.2f%% 回撤 %7.2f%% 卡玛 %5.2f 笔 %d" % (ws, b["ann"], b["dd"], b["calmar"], b["trades"]), flush=True)

rows = []
GRID = []
for codes, ctag in ((CYB, "仅创业"), (ALL, "全部")):
    for win in (10, 15):
        for persist in (1, 3, 5):
            for mm in (0.0, 0.02):
                GRID.append((ctag, codes, win, persist, mm))

print("\n=== 网格 %d 格 x %d 起点 ===" % (len(GRID), len(STARTS)), flush=True)
for ctag, codes, win, persist, mm in GRID:
    r = run(SLOPE_GATE_ON=True, SLOPE_GATE_CODES=codes, SLOPE_GATE_WIN=win,
            SLOPE_GATE_PERSIST=persist, SLOPE_GATE_MOM_MIN=mm)
    da = [r[ws]["ann"] - BASE[ws]["ann"] for ws in STARTS]
    dd = [r[ws]["dd"] - BASE[ws]["dd"] for ws in STARTS]
    dc = [r[ws]["calmar"] - BASE[ws]["calmar"] for ws in STARTS]
    wins = sum(1 for x in da if x >= 0)
    tag = "%s w%d p%d mm%.0f" % (ctag, win, persist, mm * 100)
    row = dict(tag=tag, ann_med=sorted(da)[2], ann_worst=min(da), dd_med=sorted(dd)[2],
               cal_med=sorted(dc)[2], wins=wins, per_start=r)
    rows.append(row)
    print("%-20s 胜 %d/5  年化中位 %+6.2f  最差 %+6.2f  ΔDD中位 %+6.2f  Δ卡玛中位 %+6.2f"
          % (tag, wins, row["ann_med"], row["ann_worst"], row["dd_med"], row["cal_med"]), flush=True)

import json
json.dump({"baseline": BASE, "rows": rows}, open("/home/ecs-user/etf_strategy/audit/out/n41_peer_slope.json", "w"),
          ensure_ascii=False, indent=1, default=float)
print("\nsaved -> audit/out/n41_peer_slope.json")
print("\n=== 按年化中位排序 ===")
for r in sorted(rows, key=lambda x: -x["ann_med"])[:10]:
    print("%-20s 胜%d/5 年化%+6.2f 最差%+6.2f ΔDD%+6.2f Δ卡玛%+6.2f"
          % (r["tag"], r["wins"], r["ann_med"], r["ann_worst"], r["dd_med"], r["cal_med"]))
