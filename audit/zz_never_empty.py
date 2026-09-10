"""never_empty (v9式避险接盘) 消融 (2026-08-28)
叠在「条件扩池+risk_adj」组合上, 并单测窄池/无条件扩池上的表现。"""
import sys, json, time
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.bt_runner import run_backtest

WINDOWS = [
    ("2016-07-07", "2026-08-20", "默认窗"),
    ("2015-01-05", "2026-08-20", "2015长窗"),
    ("2020-01-01", "2026-08-20", "2020短窗"),
    ("2025-01-01", "2026-08-20", "25/26窗"),
]
NARROW = "纳指,黄金,创业"
EXTRA_UNIV = "纳指,黄金,创业,有色,红利低波,科创50,中证2000"
EXTRA = "有色,红利低波,科创50,中证2000"

out = []


def run(tag, ws, we, wtag, **kw):
    p = {"start_date": ws, "end_date": we, **kw}
    t0 = time.time()
    r = run_backtest(p)
    if "error" in r:
        print("%-46s ERROR %s" % (tag, r["error"]))
        return
    print("%-46s ann %6.2f%% dd %6.2f%% cal %5.2f sharpe %4.2f total %7.1f%% trades %3d (%.0fs)"
          % (tag, r["ann_ret"], r["max_dd"], r["calmar"], r.get("sharpe", 0),
             r["total_ret"], r["trade_count"], time.time() - t0))
    out.append({"tag": tag, "window": wtag, "ann": r["ann_ret"], "dd": r["max_dd"],
                "cal": r["calmar"], "sharpe": r.get("sharpe"), "total": r["total_ret"],
                "trades": r["trade_count"], "yearly": r.get("yearly")})


for ws, we, wtag in WINDOWS:
    print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we))
    run("基线 窄池3", ws, we, wtag, universe=NARROW)
    run("窄池3 + never_empty", ws, we, wtag, universe=NARROW, never_empty=True)
    run("条件扩池+risk_adj (组合基线)", ws, we, wtag, universe=EXTRA_UNIV,
        bull_pool_extra=EXTRA, bull_pool_rule="sh_mom20", risk_adj_on=True)
    run("条件扩池+risk_adj+never_empty ★", ws, we, wtag, universe=EXTRA_UNIV,
        bull_pool_extra=EXTRA, bull_pool_rule="sh_mom20", risk_adj_on=True, never_empty=True)
    if wtag != "25/26窗":
        run("无条件扩池+risk_adj+never_empty", ws, we, wtag, universe=EXTRA_UNIV,
            risk_adj_on=True, never_empty=True)

json.dump(out, open('/home/ecs-user/etf_strategy/audit/out/never_empty_ablation.json', 'w'),
          ensure_ascii=False, indent=1)
print("\nsaved -> audit/out/never_empty_ablation.json")
