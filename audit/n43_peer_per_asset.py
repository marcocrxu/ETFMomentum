# -*- coding: utf-8 -*-
"""N-43: 斜率门(w12 p5) 在同行 v9 上逐标的扫描

用户 2026-11-02 指令: "这个参数在同行的标的上挨个试一下, 不要只开创业板, 也因为有黄金"
—— 我方(纳指/黄金/创业)上线配置是「创业+黄金」, 故同行侧除「仅单标的」外, 也测「单标的+黄金」。

固定参数: SLOPE_GATE_WIN=12, SLOPE_GATE_PERSIST=5, SLOPE_GATE_MOM_MIN=0.0, mode=exit(默认分支)
口径: 收盘成交 (N-23: 14:57-15:00 集合竞价 = 收盘价, 零口径差), 无额外滑点
多起点: 2016/2018/2020/2022/2024-01 -> 2026-10-29
"""
import sys, json, time
sys.path.insert(0, '/home/ecs-user/etf_strategy/backtest/v9_official')
import strategy
from market_data import UNIVERSE, fetch_history
from backtest import backtest

HIST = {c: fetch_history(c) for c in UNIVERSE}
CAL = [r[0] for r in HIST["510300"]]
STARTS = ["2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]
END = "2026-10-29"
GOLD = "518880"

# 可交易标的(排除货币ETF)
ASSETS = [(c, v[0]) for c, v in UNIVERSE.items() if v[2] != "cash"]

KEYS = ("SLOPE_GATE_ON", "SLOPE_GATE_CODES", "SLOPE_GATE_WIN",
        "SLOPE_GATE_MOM_MIN", "SLOPE_GATE_PERSIST")
ORIG = {k: getattr(strategy, k) for k in KEYS}

def run(codes=None, on=False):
    strategy.SLOPE_GATE_ON = on
    strategy.SLOPE_GATE_CODES = codes
    strategy.SLOPE_GATE_WIN = 12
    strategy.SLOPE_GATE_PERSIST = 5
    strategy.SLOPE_GATE_MOM_MIN = 0.0
    out = {}
    for ws in STARTS:
        r = backtest(HIST, CAL, start=ws, end=END)
        out[ws] = {"ann": r["ann"] * 100, "dd": r["max_dd"] * 100,
                   "calmar": r["calmar"], "trades": r["switches"]}
    return out

t0 = time.time()
print("=== 基线 ===", flush=True)
BASE = run(on=False)
for ws in STARTS:
    b = BASE[ws]
    print("  %s 年化 %6.2f%% 回撤 %7.2f%% 卡玛 %5.2f 笔 %d" % (
        ws, b["ann"], b["dd"], b["calmar"], b["trades"]), flush=True)
print("(%.0fs)" % (time.time() - t0), flush=True)

rows = []
for code, name in ASSETS:
    for tag, codes in (("仅%s" % name, (code,)),
                       ("%s+黄金" % name, (code, GOLD))):
        r = run(codes=codes, on=True)
        da = [r[ws]["ann"] - BASE[ws]["ann"] for ws in STARTS]
        dd = [r[ws]["dd"] - BASE[ws]["dd"] for ws in STARTS]
        dc = [r[ws]["calmar"] - BASE[ws]["calmar"] for ws in STARTS]
        wins = sum(1 for x in da if x >= 0)
        row = dict(code=code, name=name, scope=tag,
                   ann_med=round(sorted(da)[2], 2), ann_worst=round(min(da), 2),
                   dd_med=round(sorted(dd)[2], 2), cal_med=round(sorted(dc)[2], 2),
                   wins=wins, per_start=r)
        rows.append(row)
        print("%-22s 胜 %d/5  年化中位 %+6.2f  最差 %+6.2f  dDD %+6.2f  dCalmar %+6.2f  (%.0fs)"
              % (tag, wins, row["ann_med"], row["ann_worst"], row["dd_med"],
                 row["cal_med"], time.time() - t0), flush=True)

json.dump({"baseline": BASE, "rows": rows},
          open("/home/ecs-user/etf_strategy/audit/out/n43_peer_per_asset.json", "w"),
          ensure_ascii=False, indent=1, default=float)
print("\nsaved -> audit/out/n43_peer_per_asset.json")

print("\n=== 汇总: 按年化中位排序 ===")
for r in sorted(rows, key=lambda x: -x["ann_med"]):
    print("%-22s 胜%d/5 年化%+6.2f 最差%+6.2f dDD%+6.2f dCalmar%+6.2f"
          % (r["scope"], r["wins"], r["ann_med"], r["ann_worst"], r["dd_med"], r["cal_med"]))
