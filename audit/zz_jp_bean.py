"""日经225+豆粕 扩池消融 (L-6) — 多窗口验证
universe 5池 = 纳指,黄金,创业,日经225,豆粕 (无德国30, 用户决议)
"""
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
    ("5池 plain", POOL5, {}),
    ("5池 条件(sh_mom20)", POOL5, {"bull_pool_extra": "日经225,豆粕"}),
    ("5池 条件+risk_adj", POOL5, {"bull_pool_extra": "日经225,豆粕", "risk_adj_on": True}),
    ("5池 +vix25", POOL5, {"vix_gate": 25.0}),
    ("5池 条件+vix25", POOL5, {"bull_pool_extra": "日经225,豆粕", "vix_gate": 25.0}),
    ("5池 +skew150", POOL5, {"skew_gate": 150.0}),
    ("5池 +vix25+skew150", POOL5, {"vix_gate": 25.0, "skew_gate": 150.0}),
    ("5池 条件+vix25+skew150", POOL5, {"bull_pool_extra": "日经225,豆粕", "vix_gate": 25.0, "skew_gate": 150.0}),
]

out = {}
for ws, we, wtag in WINDOWS:
    print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we))
    rows = []
    for label, uni, extra in CONFIGS:
        p = {"universe": uni, "start_date": ws, "end_date": we}
        p.update(extra)
        r = run_backtest(p)
        key = f"{wtag}|{label}"
        out[key] = {"ret": r["total_ret"], "dd": r["max_dd"], "calmar": r["calmar"],
                    "sharpe": r["sharpe"], "trades": r["trade_count"],
                    "neg_years": r["neg_years"], "last": r["last_position"],
                    "universe": r["actual_universe"], "checks": r["checks"]}
        rows.append((label, r["total_ret"], r["max_dd"], r["calmar"], r["sharpe"], r["trade_count"]))
    for label, ret, dd, cal, sh, tr in rows:
        print("%-28s ret=%9.2f  dd=%7.2f  calmar=%5.2f  sharpe=%4.2f  trades=%3d" % (label, ret, dd, cal, sh, tr))

with open("audit/out/jp_bean.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved audit/out/jp_bean.json")
