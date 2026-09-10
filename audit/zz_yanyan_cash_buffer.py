"""cash_skip_buffer 消融 (2026-08-28): yanyan 空仓入场 2% 缓冲 开 vs 关, 三窗口 + 逐年"""
import sys, json
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.yanyan_runner import run_yanyan

WINDOWS = [
    ("2016-07-07", "2026-08-20", "默认窗"),
    ("2015-01-05", "2026-08-20", "2015长窗"),
    ("2020-01-01", "2026-08-20", "2020短窗"),
]

out = []
for ws, we, wtag in WINDOWS:
    print("===== %s (%s ~ %s) =====" % (wtag, ws, we))
    for on, tag in [(True, "缓冲开(默认)"), (False, "缓冲关")]:
        r = run_yanyan({"start_date": ws, "end_date": we, "cash_skip_buffer": on})
        if "error" in r:
            print("%-14s ERROR %s" % (tag, r["error"]))
            continue
        print("%-14s 总 %+8.1f%%  年化 %5.2f%%  DD %5.2f%%  夏普 %4.2f  卡玛 %4.2f  换手 %3d"
              % (tag, r["total_ret"], r["ann_ret"], r["max_dd"], r.get("sharpe", 0),
                 r.get("calmar", 0), r.get("trade_count", 0)))
        out.append({"window": wtag, "skip_buffer": on, "total": r["total_ret"],
                    "ann": r["ann_ret"], "dd": r["max_dd"], "sharpe": r.get("sharpe"),
                    "calmar": r.get("calmar"), "trades": r.get("trade_count"),
                    "yearly": r.get("yearly")})

print("\n===== 逐年 (默认窗) =====")
for on, tag in [(True, "缓冲开"), (False, "缓冲关")]:
    r = run_yanyan({"start_date": "2016-07-07", "end_date": "2026-08-20", "cash_skip_buffer": on})
    yrs = r.get("yearly") or []
    s = "  ".join("%s:%+.1f%%" % (y.get("year", y), y.get("ret", 0)) for y in yrs)
    print("%-6s  %s" % (tag, s))

json.dump(out, open('/home/ecs-user/etf_strategy/audit/out/yanyan_cash_buffer.json', 'w'),
          ensure_ascii=False, indent=1, default=str)
print("\nsaved -> audit/out/yanyan_cash_buffer.json")
