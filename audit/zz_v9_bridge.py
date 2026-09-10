"""v9 引擎 → yanyan 配置的桥接归因 (同窗口 2016-07-07~2026-08-20)
逐步关闭 v9 组件直到与 yanyan 等价, 看差距落在哪."""
import sys
sys.path.insert(0, '/home/ecs-user/etf_strategy/backtest/v9_official')
from market_data import UNIVERSE, fetch_history
import backtest as v9bt

WS, WE = "2016-07-07", "2026-08-20"
codes = list(UNIVERSE)
histories = {c: fetch_history(c) for c in codes}
calendar = [r[0] for r in histories["510300"]]


def fmt(tag, r):
    return ("%-44s 总 %8.0f%%  年化 %5.2f%%  DD %5.1f%%  夏普 %4.2f  卡玛 %4.2f  换手 %3d"
            % (tag, (r["nav"] - 1) * 100, r["ann"] * 100, r["max_dd"] * 100,
               r["sharpe"], r["calmar"], r["switches"]))


steps = [
    ("v9 默认全组件", {}),
    ("→ 去深跌恐慌抄底", {"crash_mom5": 0.0}),
    ("→ 去单日急跌离场 panic", {"crash_mom5": 0.0, "panic_drop": 0.0}),
    ("→ 去熊市进场门槛7%", {"crash_mom5": 0.0, "panic_drop": 0.0, "bear_enter_mom": 0.0}),
    ("→ 去 WLS25(回 mom20/vol)", {"crash_mom5": 0.0, "panic_drop": 0.0, "bear_enter_mom": 0.0,
                                  "score_wls": False}),
    ("→ 去永不空仓(空仓停货基) = yanyan等价", {"crash_mom5": 0.0, "panic_drop": 0.0, "bear_enter_mom": 0.0,
                                               "score_wls": False, "never_empty": False}),
    ("yanyan 等价 + 永不空仓加回", {"crash_mom5": 0.0, "panic_drop": 0.0, "bear_enter_mom": 0.0,
                                   "score_wls": False, "never_empty": True}),
    ("yanyan 等价 + WLS25加回", {"crash_mom5": 0.0, "panic_drop": 0.0, "bear_enter_mom": 0.0,
                                "score_wls": True, "never_empty": False}),
]
for tag, kw in steps:
    r = v9bt.backtest(histories, calendar, start=WS, end=WE, **kw)
    print(fmt(tag, r))
print()
print("对照: yanyan 你的引擎同窗口 = 总 +579.7%% / 年化 21.70%% / DD -26.64%% / 夏普 0.90 / 卡玛 0.81 / 换手 224")
