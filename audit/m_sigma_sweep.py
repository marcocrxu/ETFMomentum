"""accel_sigma 敏感性扫描 (2026-09-08): 3σ 是不是合理配置?
universe=纳指,黄金,创业; 多窗口; 含 accel_on=False 对照"""
import sys, json, time
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.bt_runner import run_backtest

WINDOWS = [
    ("2016-07-07", "2026-09-04", "主样本"),
    ("2015-01-05", "2026-09-04", "2015长窗"),
    ("2020-01-01", "2026-09-04", "2020短窗"),
]
out = []
def run(tag, ws, we, **kw):
    p = {"universe": "纳指,黄金,创业", "start_date": ws, "end_date": we}
    p.update(kw)
    t0 = time.time()
    r = run_backtest(p)
    if "error" in r:
        print("%-30s ERROR %s" % (tag, r["error"]), flush=True); return
    print("%-30s ann %6.2f%% dd %6.2f%% cal %5.2f sharpe %4.2f total %8.1f%% trades %3d (%.0fs)"
          % (tag, r["ann_ret"], r["max_dd"], r["calmar"], r.get("sharpe", 0),
             r["total_ret"], r["trade_count"], time.time() - t0), flush=True)
    out.append({"tag": tag, "window": tag.split("|")[0].strip(), "sigma": tag.split("|")[1].strip(),
                "ann": r["ann_ret"], "dd": r["max_dd"], "cal": r["calmar"],
                "sharpe": r.get("sharpe"), "total": r["total_ret"], "trades": r["trade_count"],
                "yearly": r.get("yearly")})

for ws, we, wtag in WINDOWS:
    print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we), flush=True)
    run("%s | 止盈OFF" % wtag, ws, we, accel_on=False)
    for s in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        run("%s | σ=%.1f" % (wtag, s), ws, we, accel_sigma=s)

json.dump(out, open('/home/ecs-user/etf_strategy/audit/out/m_sigma_sweep.json', 'w'),
          ensure_ascii=False, indent=1)
print("\nsaved -> audit/out/m_sigma_sweep.json")
