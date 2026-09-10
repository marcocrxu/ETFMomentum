"""yanyan 移植 vs 主引擎窄池3, 同窗口 (2016-07-07 ~ 2026-08-20)"""
import sys
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.yanyan_runner import run_yanyan
from web.bt_runner import run_backtest

WS, WE = "2016-07-07", "2026-08-20"

r = run_yanyan({"start_date": WS, "end_date": WE})
if "error" in r:
    print("yanyan ERROR:", r["error"]); sys.exit(1)
print("yanyan 基线(默认参数): 总 %+.1f%% 年化 %.2f%% DD %.2f%% 夏普 %.2f 卡玛 %.2f 换手 %d"
      % (r["total_ret"], r["ann_ret"], r["max_dd"], r.get("sharpe", 0),
         r.get("calmar", 0), r.get("trade_count", 0)))

rb = run_backtest({"universe": "纳指,黄金,创业", "start_date": WS, "end_date": WE})
print("主引擎窄池3(QQQ0.7默认): 总 %+.1f%% 年化 %.2f%% DD %.2f%% 夏普 %.2f 卡玛 %.2f 换手 %d"
      % (rb["total_ret"], rb["ann_ret"], rb["max_dd"], rb.get("sharpe", 0),
         rb["calmar"], rb["trade_count"]))
