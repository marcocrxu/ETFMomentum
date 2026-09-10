"""v9 牛市行为学习消融 (2026-08-25)
开关1 panic_drop: 持仓单日跌<=X%当日离场 (v9式左尾保险)
开关2 bull_pool_extra: 牛市(上证mom20>0)才把A股附加池纳入排名, 熊市制度性离场 (学v9 regime条件池)
基线对照: 纳指,黄金,创业 (45.51%/-18.08%/2.52/253笔) + 无条件扩池(F4式)"""
import sys, json, time
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.bt_runner import run_backtest

WINDOWS = [
    ("2016-07-07", "2026-08-20", "默认窗(主样本)"),
    ("2015-01-05", "2026-08-20", "2015长窗"),
    ("2020-01-01", "2026-08-20", "2020短窗"),
]
BASE = {"universe": "纳指,黄金,创业"}
EXTRA_UNIV = "纳指,黄金,创业,有色,红利低波,科创50,中证2000"

def run(tag, **kw):
    p = dict(BASE); p["start_date"], p["end_date"] = kw["start_date"], kw["end_date"]
    p["universe"] = kw.get("universe", BASE["universe"])
    for k in ("panic_drop", "bull_pool_extra", "bull_pool_rule"):
        if k in kw: p[k] = kw[k]
    t0 = time.time()
    r = run_backtest(p)
    if "error" in r:
        print("%-46s ERROR %s" % (tag, r["error"]))
        return
    print("%-46s ann %6.2f%% dd %6.2f%% cal %5.2f sharpe %4.2f total %7.1f%% trades %3d  (%.0fs)"
          % (tag, r["ann_ret"], r["max_dd"], r["calmar"], r.get("sharpe", 0),
             r["total_ret"], r["trade_count"], time.time() - t0))
    out.append({"tag": tag, "ann": r["ann_ret"], "dd": r["max_dd"], "cal": r["calmar"],
                "sharpe": r.get("sharpe"), "total": r["total_ret"], "trades": r["trade_count"],
                "yearly": r.get("yearly")})

out = []
for ws, we, wtag in WINDOWS:
    print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we))
    run("基线 3标的池", start_date=ws, end_date=we)
    run("扩池(无条件) 开关OFF", start_date=ws, end_date=we, universe=EXTRA_UNIV)
    run("扩池 + bull_pool_extra ON", start_date=ws, end_date=we, universe=EXTRA_UNIV,
        bull_pool_extra="有色,红利低波,科创50,中证2000")
    for pd_ in (3.0, 4.0, 5.0):
        run("基线 + panic_drop=%.0f%%" % pd_, start_date=ws, end_date=we, panic_drop=pd_)
    run("扩池+bull_pool + panic_drop=4%", start_date=ws, end_date=we, universe=EXTRA_UNIV,
        bull_pool_extra="有色,红利低波,科创50,中证2000", panic_drop=4.0)

json.dump(out, open('/home/ecs-user/etf_strategy/audit/out/v9_learn_ablation.json', 'w'),
          ensure_ascii=False, indent=1)
print("\nsaved -> audit/out/v9_learn_ablation.json")
