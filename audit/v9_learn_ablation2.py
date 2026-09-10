"""bull_pool_rule 变体: sh_mom20 vs sh_ma250 (v9式300>MA250, 用上证指数代理)"""
import sys, json
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.bt_runner import run_backtest

EXTRA_UNIV = "纳指,黄金,创业,有色,红利低波,科创50,中证2000"
for ws, we, wtag in [("2016-07-07","2026-08-20","默认窗"),
                     ("2015-01-05","2026-08-20","2015长窗"),
                     ("2020-01-01","2026-08-20","2020短窗")]:
    print("===== %s =====" % wtag)
    for rule in ("sh_mom20", "sh_ma250"):
        r = run_backtest({"universe": EXTRA_UNIV, "start_date": ws, "end_date": we,
                          "bull_pool_extra": "有色,红利低波,科创50,中证2000",
                          "bull_pool_rule": rule})
        if "error" in r:
            print("ERROR", r["error"]); continue
        print("rule=%-9s ann %6.2f%% dd %6.2f%% cal %5.2f sharpe %4.2f total %7.1f%% trades %3d"
              % (rule, r["ann_ret"], r["max_dd"], r["calmar"], r.get("sharpe",0),
                 r["total_ret"], r["trade_count"]))
