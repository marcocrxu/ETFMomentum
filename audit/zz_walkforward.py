"""L-8: 双窗拟合 + 2014-2016 外推检验
A组参数(入场/风控) 在 2015-2020 拟合; B组参数(退出/趋势) 在 2020-2025 拟合;
整合后在 2014-2016 (拟合窗之外) 与当前默认参数对比。
目标函数: Calmar (项目惯例, 风险调整)。
"""
import json
from web.bt_runner import run_backtest

U = "纳指,黄金,创业"
WA = ("2015-01-01", "2020-12-31")
WB = ("2020-01-01", "2025-12-31")

# 当前默认值 (基线)
DEFAULTS = {"mom_period": 20, "stop_loss": -5.0, "confirm_days": 2,
            "crash_sigma": -2.5, "exception_gap": 3.0, "accel_sigma": 3.0,
            "accel_cooldown": 3, "min_hold": 10, "min_gap": 1.0,
            "nz_mom_qqq_w": 0.7}

GROUP_A = [("mom_period", [10, 15, 20, 25, 30]),
           ("stop_loss", [-3.0, -5.0, -7.0]),
           ("confirm_days", [1, 2, 3]),
           ("crash_sigma", [-2.0, -2.5, -3.0]),
           ("exception_gap", [2.0, 2.5, 3.0, 4.0])]
GROUP_B = [("accel_sigma", [2.5, 3.0, 3.5]),
           ("accel_cooldown", [0, 2, 3, 5]),
           ("min_hold", [5, 10, 15, 20]),
           ("min_gap", [0.5, 1.0, 1.5, 2.0]),
           ("nz_mom_qqq_w", [0.6, 0.7, 0.8, 1.0])]

def score(p, ws, we):
    r = run_backtest({"universe": U, "start_date": ws, "end_date": we, **p})
    return r

def fit(group, ws, we, fixed):
    """逐个参数贪心拟合: 其他参数保持默认/已定值, 按 Calmar 选最优"""
    chosen = {}
    print("=== 拟合 %s ~ %s ===" % (ws, we))
    for name, vals in group:
        best_v, best_c = None, -1
        for v in vals:
            p = dict(DEFAULTS); p.update(chosen); p.update(fixed); p[name] = v
            r = score(p, ws, we)
            cal = r["calmar"]
            mark = ""
            if cal > best_c:
                best_v, best_c = v, cal
                mark = " <-- best"
            print("  %-16s=%-6s ret=%9.2f dd=%7.2f calmar=%5.2f%s" %
                  (name, v, r["total_ret"], r["max_dd"], cal, mark))
        chosen[name] = best_v
        print("  >> %s = %s (calmar %.2f)\n" % (name, best_v, best_c))
    return chosen

# 顺序: 先 A 后 B (互不重叠的参数名, 顺序无关)
fa = fit(GROUP_A, *WA, fixed={})
fb = fit(GROUP_B, *WB, fixed=fa)
merged = dict(fa); merged.update(fb)
print("整合参数:", json.dumps(merged, ensure_ascii=False))

# ── 检验 ──
def ev(p, ws, we, tag):
    r = score(p, ws, we)
    print("%-28s %s~%s: ret=%9.2f dd=%7.2f calmar=%5.2f sharpe=%4.2f trades=%3d negY=%d" %
          (tag, ws, we, r["total_ret"], r["max_dd"], r["calmar"], r["sharpe"],
           r["trade_count"], r["neg_years"]))
    return r

out = {}
print("\n=== 样本内复测 ===")
ev(DEFAULTS, *WA, "默认参数 (A窗)")
ev(merged, *WA, "整合参数 (A窗)")
ev(DEFAULTS, *WB, "默认参数 (B窗)")
ev(merged, *WB, "整合参数 (B窗)")
print("\n=== 外推检验 (拟合窗之外的年份) ===")
for ws, we, tag in (("2014-01-01", "2016-12-31", "2014-2016 外推"),
                    ("2014-01-01", "2014-12-31", "2014 纯外推"),
                    ("2016-07-07", "2026-08-12", "全窗(默认口径)")):
    r0 = ev(DEFAULTS, ws, we, "默认参数: " + tag)
    r1 = ev(merged, ws, we, "整合参数: " + tag)
    out[tag] = {"default": {"ret": r0["total_ret"], "dd": r0["max_dd"], "calmar": r0["calmar"],
                            "sharpe": r0["sharpe"], "trades": r0["trade_count"], "negY": r0["neg_years"]},
                "merged": {"ret": r1["total_ret"], "dd": r1["max_dd"], "calmar": r1["calmar"],
                           "sharpe": r1["sharpe"], "trades": r1["trade_count"], "negY": r1["neg_years"]}}

out["fitA"] = fa; out["fitB"] = fb; out["merged"] = merged
with open("audit/out/walk_forward.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved audit/out/walk_forward.json")
