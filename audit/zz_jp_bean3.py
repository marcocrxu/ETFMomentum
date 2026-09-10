"""L-6 补2: 顶配(条件+rate_shock) + skew150 多窗口验证 + 归因"""
import json
from web.bt_runner import run_backtest

WINDOWS = [
    ("2016-07-07", "2026-08-12", "默认窗"),
    ("2015-01-05", "2026-08-12", "2015长窗"),
    ("2020-01-01", "2026-08-12", "2020短窗"),
    ("2025-01-01", "2026-08-12", "25/26窗"),
]
NARROW = "纳指,黄金,创业"
POOL5 = "纳指,黄金,创业,日经225,豆粕"

CONFIGS = [
    ("基线3", NARROW, {}),
    ("基线3+rate_shock0.4", NARROW, {"rate_shock_gate": 0.4}),
    ("基线3+rate_shock+skew150", NARROW, {"rate_shock_gate": 0.4, "skew_gate": 150.0}),
    ("5池 条件+rate_shock", POOL5, {"bull_pool_extra": "日经225,豆粕", "rate_shock_gate": 0.4}),
    ("5池 条件+rate_shock+skew150", POOL5,
     {"bull_pool_extra": "日经225,豆粕", "rate_shock_gate": 0.4, "skew_gate": 150.0}),
]

out = {}
for ws, we, wtag in WINDOWS:
    print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we))
    for label, uni, extra in CONFIGS:
        p = {"universe": uni, "start_date": ws, "end_date": we}
        p.update(extra)
        r = run_backtest(p)
        out[f"{wtag}|{label}"] = {"ret": r["total_ret"], "dd": r["max_dd"],
                                  "calmar": r["calmar"], "sharpe": r["sharpe"],
                                  "trades": r["trade_count"], "neg_years": r["neg_years"],
                                  "last": r["last_position"], "checks": r["checks"]}
        print("%-32s ret=%9.2f  dd=%7.2f  calmar=%5.2f  sharpe=%4.2f  trades=%3d  negY=%d" % (
            label, r["total_ret"], r["max_dd"], r["calmar"], r["sharpe"],
            r["trade_count"], r["neg_years"]))

with open("audit/out/jp_bean_skew.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved audit/out/jp_bean_skew.json")
