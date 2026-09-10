"""v9 官方 vs yanyan 移植 同窗口对比 + v9 组件归因 (2026-08-28, 用户提问"对比策略用了很多标的效果很好, 为啥我的不行")"""
import sys
sys.path.insert(0, '/home/ecs-user/etf_strategy')
sys.path.insert(0, '/home/ecs-user/etf_strategy/backtest/v9_official')

WS, WE = "2016-07-07", "2026-08-20"


def fmt(tag, r):
    nav = r["nav"]
    return ("%-34s 总 %8.0f%%  年化 %5.2f%%  DD %5.1f%%  夏普 %4.2f  卡玛 %4.2f  换手 %3d"
            % (tag, (nav - 1) * 100, r["ann"] * 100, r["max_dd"] * 100,
               r["sharpe"], r["calmar"], r["switches"]))


def run_v9(**kw):
    from market_data import UNIVERSE, fetch_history
    import backtest as v9bt
    codes = list(UNIVERSE)
    histories = {c: fetch_history(c) for c in codes}
    calendar = [r[0] for r in histories["510300"]]
    return v9bt.backtest(histories, calendar, start=WS, end=WE, **kw)


def run_yanyan(tag, **kw):
    from web.yanyan_runner import run_yanyan
    p = {"start_date": WS, "end_date": WE}
    p.update(kw)
    r = run_yanyan(p)
    if "error" in r:
        print("%-34s ERROR %s" % (tag, r["error"]))
        return None
    nav = r["final_nav"] if "final_nav" in r else None
    if nav is None:
        print("%-34s no final_nav keys=%s" % (tag, sorted(r.keys())))
        return None
    return {"nav": nav, "ann": r.get("ann_ret", 0) / 100, "max_dd": r.get("max_dd", 0) / 100,
            "sharpe": r.get("sharpe", 0), "calmar": r.get("calmar", 0),
            "switches": r.get("trade_count", 0)}


print("===== 窗口 %s ~ %s =====" % (WS, WE))
print()
print("--- v9 官方 (作者代码, 默认全组件) ---")
for tag, kw in [
    ("v9 基线(默认全组件)", {}),
    ("  └去 WLS25 排名(回 mom20/vol)", {"score_wls": False}),
    ("  └去永不空仓(空仓停货基)", {"never_empty": False}),
    ("  └去单日急跌离场 panic4%", {"panic_drop": 0.0}),
    ("  └去深跌恐慌抄底", {"crash_mom5": 0.0}),
    ("  └去熊市进场门槛 7%", {"bear_enter_mom": 0.0}),
    ("  └去轮动缓冲 2%", {"buffer": 0.0}),
    ("  └去过热快离场", {"overheat": 9.9}),
]:
    r = run_v9(**kw)
    print(fmt(tag, r))

print()
print("--- yanyan 移植 (你的引擎, 11 标的) ---")
r = run_yanyan("yanyan 基线(默认参数)", )
if r:
    print(fmt("yanyan 基线(默认参数)", r))

print()
print("--- 对照: 你的主引擎窄池 3 基线 ---")
from web.bt_runner import run_backtest
rb = run_backtest({"universe": "纳指,黄金,创业", "start_date": WS, "end_date": WE})
print("%-34s 总 %8.0f%%  年化 %5.2f%%  DD %5.1f%%  夏普 %4.2f  卡玛 %4.2f  换手 %3d"
      % ("主引擎窄池3(QQQ0.7默认)", rb["total_ret"], rb["ann_ret"], rb["max_dd"],
         rb.get("sharpe", 0), rb["calmar"], rb["trade_count"]))
