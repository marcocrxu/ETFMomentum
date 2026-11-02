"""N-40 第二轮 (修订): 加入抗抖动维度
首轮实测发现无状态机时出现"1 天卖出/次日买回"抖动(2017-08~09 连续多次),
故第二轮把 persist / cooldown 作为必测维度。
协议补充见 audit/n40_mom_slope_protocol.md 第 9 节。
"""
import sys, time
sys.path.insert(0, "/home/ecs-user/etf_strategy")
from web.bt_runner import run_backtest
from audit.harness import save, diff_independence

UNIV = "纳指,黄金,创业"
END = "2026-10-29"
STARTS = ["2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]

def bt(ws, **kw):
    p = {"universe": UNIV, "start_date": ws, "end_date": END}
    p.update(kw)
    return run_backtest(p)

t0 = time.time()
base = {}
for ws in STARTS:
    base[ws] = bt(ws)
    print("BASE %s ann %.2f dd %.2f trades %d" % (ws, base[ws]["ann_ret"], base[ws]["max_dd"], base[ws]["trade_count"]), flush=True)

GRID = []
for win in (10, 15):
    for persist in (1, 3, 5):
        for cd in (0, 5, 10):
            GRID.append(dict(tag="w%d p%d cd%d" % (win, persist, cd),
                             slope_gate_names="创业", slope_gate_win=win,
                             slope_gate_mode="exit", slope_gate_persist=persist,
                             slope_gate_cooldown=cd))

out = {"baseline": {ws: {k: base[ws][k] for k in ("ann_ret", "max_dd", "calmar", "trade_count")} for ws in STARTS},
       "variants": []}
print("\n网格 %d 格 x %d 起点" % (len(GRID), len(STARTS)), flush=True)

for g in GRID:
    tag = g.pop("tag")
    diffs = []; dd_d = []; res = {}
    for ws in STARTS:
        r = bt(ws, **g)
        res[ws] = {"ann": r["ann_ret"], "dd": r["max_dd"], "cal": r["calmar"], "trades": r["trade_count"]}
        diffs.append(r["ann_ret"] - base[ws]["ann_ret"])
        dd_d.append(r["max_dd"] - base[ws]["max_dd"])
    wins = sum(1 for d in diffs if d >= 0)
    med = sorted(diffs)[len(diffs)//2]
    row = {"tag": tag, "params": g, "wins": "%d/%d" % (wins, len(diffs)),
           "median": round(med, 2), "worst": round(min(diffs), 2),
           "dd_median": round(sorted(dd_d)[len(dd_d)//2], 2),
           "trade_ratio": round(res[STARTS[-1]]["trades"] / max(base[STARTS[-1]]["trade_count"], 1), 2),
           "per_start": res}
    out["variants"].append(row)
    print("%-14s 胜%s 中位%+.2f 最差%+.2f ΔDD中位%+.2f 交易比%.2fx (%.0fs)" % (
        tag, row["wins"], med, min(diffs), row["dd_median"], row["trade_ratio"], time.time() - t0), flush=True)

save(out, "n40_mom_slope_r2")
print("\n=== 按中位排序 ===")
for r in sorted(out["variants"], key=lambda x: -x["median"]):
    print("%-14s 胜%-4s 中位%+.2f 最差%+.2f ΔDD%+.2f 交易比%.2fx" % (
        r["tag"], r["wins"], r["median"], r["worst"], r["dd_median"], r["trade_ratio"]))
