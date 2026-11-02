"""N-40: 创业板 mom20 斜率门消融 (2026-11-02)
协议: audit/n40_mom_slope_protocol.md (跑前注册)
网格: win{5,10,15} x mom_min{0,2} x mode{exit} + win{5,10,15} x mom_min{0,2} x mode{tight} sigma{1.5,2.0}
五起点: 2016/2018/2020/2022/2024-01 -> 2026-10-29 (2014 起点需缓存外数据, 见下)
只作用于创业; 默认关 = 基线逐分不变
"""
import sys, json, time
sys.path.insert(0, "/home/ecs-user/etf_strategy")
from web.bt_runner import run_backtest
from audit.harness import save, diff_independence

UNIV = "纳指,黄金,创业"
END = "2026-10-29"
# 说明: 2014-01 起点需要 2012 年数据, 会触发 DataFeed 头部回补写盘(缓存只读, 沙箱拒绝)。
# 缓存覆盖 2013-01 起, 故用 2016/2018/2020/2022/2024 五个起点(与 N-36/N-37 后段一致口径)。
STARTS = ["2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]

def bt(ws, **kw):
    p = {"universe": UNIV, "start_date": ws, "end_date": END}
    p.update(kw)
    return run_backtest(p)

def ann_of(r):
    return r["ann_ret"] if "error" not in r else None

# 基线
t0 = time.time()
base = {}
for ws in STARTS:
    r = bt(ws)
    base[ws] = r
    print("BASE %s ann %.2f dd %.2f cal %.2f trades %d" % (ws, r["ann_ret"], r["max_dd"], r["calmar"], r["trade_count"]), flush=True)

# 网格
GRID = []
for mode in ("exit", "tight"):
    for win in (5, 10, 15):
        for mm in (0, 2):
            if mode == "exit":
                GRID.append(dict(tag="exit w%d m%d" % (win, mm),
                                 slope_gate_names="创业", slope_gate_win=win,
                                 slope_gate_mode="exit", slope_gate_mom_min=mm))
            else:
                for sg in (1.5, 2.0):
                    GRID.append(dict(tag="tight w%d m%d s%.1f" % (win, mm, sg),
                                     slope_gate_names="创业", slope_gate_win=win,
                                     slope_gate_mode="tight", slope_gate_sigma=sg,
                                     slope_gate_mom_min=mm))

print("\n网格 %d 格 x %d 起点" % (len(GRID), len(STARTS)), flush=True)
out = {"baseline": {ws: {k: base[ws][k] for k in ("ann_ret", "max_dd", "calmar", "trade_count")} for ws in STARTS},
       "variants": []}

for g in GRID:
    tag = g.pop("tag")
    res = {}
    diffs = []
    for ws in STARTS:
        r = bt(ws, **g)
        res[ws] = {"ann": r.get("ann_ret"), "dd": r.get("max_dd"), "cal": r.get("calmar"),
                   "trades": r.get("trade_count")}
        if "error" not in r:
            diffs.append(r["ann_ret"] - base[ws]["ann_ret"])
    wins = sum(1 for d in diffs if d >= 0)
    med = sorted(diffs)[len(diffs)//2]
    row = {"tag": tag, "params": g, "per_start": res, "wins": "%d/%d" % (wins, len(diffs)),
           "median": round(med, 2), "worst": round(min(diffs), 2),
           "trade_ratio": round(res[STARTS[-1]]["trades"] / max(base[STARTS[-1]]["trade_count"], 1), 2)}
    out["variants"].append(row)
    print("%-22s 胜%s 中位%+.2f 最差%+.2f 交易比%.2fx  (%.0fs)" % (
        tag, row["wins"], med, min(diffs), row["trade_ratio"], time.time() - t0), flush=True)

save(out, "n40_mom_slope")
print("\n=== 排序 (按中位) ===")
for r in sorted(out["variants"], key=lambda x: -x["median"])[:8]:
    print("%-22s 胜%s 中位%+.2f 最差%+.2f 交易比%.2fx" % (r["tag"], r["wins"], r["median"], r["worst"], r["trade_ratio"]))
