"""yanyan 残余差异定位: 空仓入场缓冲 / MA周期 (同窗口 2016-07-07~2026-08-20)"""
import sys
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.yanyan_runner import run_yanyan

WS, WE = "2016-07-07", "2026-08-20"
variants = [
    ("yanyan 默认", {}),
    ("  └关空仓入场2%缓冲", {"cash_skip_buffer": False}),
    ("  └regime MA250(v9式)", {"regime_ma": 250}),
    ("  └关缓冲+MA250", {"cash_skip_buffer": False, "regime_ma": 250}),
]
for tag, kw in variants:
    p = {"start_date": WS, "end_date": WE}
    p.update(kw)
    r = run_yanyan(p)
    if "error" in r:
        print("%-26s ERROR %s" % (tag, r["error"])); continue
    print("%-26s 总 %+7.1f%%  年化 %5.2f%%  DD %5.2f%%  夏普 %4.2f  卡玛 %4.2f  换手 %3d"
          % (tag, r["total_ret"], r["ann_ret"], r["max_dd"], r.get("sharpe", 0),
             r.get("calmar", 0), r.get("trade_count", 0)))
