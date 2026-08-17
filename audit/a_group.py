"""A组: 参数与稳健性验证
A1 accel_sigma 补档 2.5/3.5 (+2/3/4 全梯度)
A2 空仓冷却补档 1/2/4 (+0/3/5/10 全梯度)
A3 冷却默认档重估 (修正引擎口径下 0/3/5)
A4 止盈冷却转正验证 (accel_cooldown=3 复验 + 分段一致性)
A5 regime_window 解耦 (上证空仓动量窗口 20/40/60)
A6 波动率归一化变体 (mom/vol20 z-score 化)
A7 σ 窗口口径审计 (扩张窗钝化: 阈值时序 + 触发频率 + ewma/percentile 变体消融)
B1-fix 消融 (confirm_bind_identity=True 十年对比)
"""
import sys
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd
from harness import bt, default_params, load_mats, run_cfg, metrics, base_cfg, save
from strategy.signal import calc_accel_thresholds

res = {}

# ── A1: accel_sigma 全梯度 ──
rows = {}
for s in [2.0, 2.5, 3.0, 3.5, 4.0]:
    r = bt(accel_sigma=s)
    rows[s] = {k: r[k] for k in ["total_ret", "ann_ret", "max_dd", "calmar", "sharpe", "trade_count"]}
res["A1"] = {"table": rows,
             "verdict": "3.0 为峰" if rows[3.0]["total_ret"] >= max(rows[2.5]["total_ret"], rows[3.5]["total_ret"]) else "3.0 非峰(平台或偏移)"}

# ── A2+A3: 空仓冷却全梯度(修正引擎) ──
rows = {}
for c in [0, 1, 2, 3, 4, 5, 10]:
    r = bt(sh_cash_cooldown=c)
    rows[c] = {k: r[k] for k in ["total_ret", "ann_ret", "max_dd", "calmar", "sharpe", "trade_count"]}
best = max(rows, key=lambda c: rows[c]["calmar"])
res["A2"] = {"table": rows}
res["A3"] = {"cooldown_0_3_5": {c: rows[c] for c in [0, 3, 5]},
             "best_by_calmar": best,
             "verdict": f"修正口径下 Calmar 最优档={best}; 当前默认3档: "
                        f"{'仍为最优' if best == 3 else '非最优, 建议改 ' + str(best)}"}

# ── A4: 止盈冷却转正验证 (全段 + 分段) ──
full0 = bt(accel_cooldown=0)
full3 = bt(accel_cooldown=3)
seg = {}
for tag, (s, e) in {"seg1_2016_2021": ("2016-07-07", "2021-06-30"),
                    "seg2_2021_2026": ("2021-07-01", "2026-08-12")}.items():
    a = bt(accel_cooldown=0, start_date=s, end_date=e)
    b = bt(accel_cooldown=3, start_date=s, end_date=e)
    seg[tag] = {"cd0": {k: a[k] for k in ["total_ret", "calmar", "max_dd", "ann_ret"]},
                "cd3": {k: b[k] for k in ["total_ret", "calmar", "max_dd", "ann_ret"]},
                "cd3_better": b["calmar"] > a["calmar"]}
res["A4"] = {
    "full": {"cd0": {k: full0[k] for k in ["total_ret", "calmar", "max_dd", "ann_ret", "trade_count"]},
             "cd3": {k: full3[k] for k in ["total_ret", "calmar", "max_dd", "ann_ret", "trade_count"]}},
    "segments": seg,
    "consistent": all(v["cd3_better"] for v in seg.values()) and full3["calmar"] > full0["calmar"],
}

# ── A5: regime_window 解耦 ──
rows = {}
mats_cache = {}
for w in [20, 40, 60]:
    mats = load_mats(["纳指", "黄金", "创业"], sh_window=w)
    cfg = base_cfg()
    actions, nav, user_nav = run_cfg(cfg, mats)
    m = metrics(user_nav, len(mats["all_dates"]))
    m["n_actions"] = len(actions)
    rows[w] = m
res["A5"] = {"table": rows,
             "verdict": ("解耦有改善" if max(rows[40]["calmar"], rows[60]["calmar"]) > rows[20]["calmar"] * 1.05
                         else "不改善→维持单参数")}

# ── A6: 波动率归一化 (mom20/vol20 排名与阈值全部 z-score 化) ──
mats = load_mats(["纳指", "黄金", "创业"], norm_vol=True)
cfg = base_cfg()
actions, nav, user_nav = run_cfg(cfg, mats)
m = metrics(user_nav, len(mats["all_dates"]))
base = bt()
res["A6"] = {"norm_vol": {**m, "n_actions": len(actions)},
             "baseline": {k: base[k] for k in ["total_ret", "ann_ret", "max_dd", "calmar"]},
             "verdict": ("Calmar 显著提升→采纳" if m["calmar"] > base["calmar"] * 1.1 else "未显著提升→不采纳")}

# ── A7: σ 窗口口径审计 ──
mats = load_mats(["纳指", "黄金", "创业"])
mom10 = mats["mom10_mat"]; mom10_idx_cyb = np.full(mats["n_full"], np.nan)
cyb = mats["cyb_p"]; sh = mats["sh_price"]
mom10_idx_cyb[10:] = (cyb[10:] / cyb[:-10] - 1) * 100
mom10_idx_sh = np.full(mats["n_full"], np.nan)
mom10_idx_sh[10:] = (sh[10:] / sh[:-10] - 1) * 100

# 止盈: 扩张窗 mean+3σ 时序统计 (逐年末值 + 触发频率/年)
cfg = base_cfg()
accel_th = calc_accel_thresholds(mom10, cfg)
dates = mats["all_full"]
years = sorted(set(d.year for d in dates))
audit = {}
for i, nm in enumerate(mats["etf_names"]):
    th = accel_th[i]; m = mom10[i]
    trig = (~np.isnan(th)) & (~np.isnan(m)) & (m > th)
    per_year = {y: int(np.sum([trig[t] for t in range(len(dates)) if dates[t].year == y])) for y in years}
    th_eoy = {y: (round(float(th[[t for t in range(len(dates)) if dates[t].year == y][-1]]), 2)
                  if any(dates[t].year == y for t in range(len(dates))) and not np.isnan(th[[t for t in range(len(dates)) if dates[t].year == y][-1]]) else None)
              for y in years}
    audit[nm] = {"triggers_per_year": per_year, "threshold_eoy": th_eoy,
                 "total_triggers": int(trig.sum())}
# 恐慌: 扩张窗 mean-2.5σ 钝化检查 — 阈值随时间漂移幅度
panic = {}
for tag, arr in [("cyb", mom10_idx_cyb), ("sh", mom10_idx_sh)]:
    s = pd.Series(arr)
    th = (s.expanding(min_periods=60).mean() - 2.5 * s.expanding(min_periods=60).std()).shift(1)
    below = (arr < th.values) & ~np.isnan(arr) & ~np.isnan(th.values)
    ma200 = pd.Series(cyb if tag == "cyb" else sh).rolling(200).mean().values
    px = cyb if tag == "cyb" else sh
    crash_days = below & (px < ma200)
    per_year = {y: int(np.sum([crash_days[t] for t in range(len(dates)) if dates[t].year == y])) for y in years}
    th_vals = th.values
    panic[tag] = {"crash_days_per_year": per_year,
                  "threshold_min": round(float(np.nanmin(th_vals)), 2),
                  "threshold_max": round(float(np.nanmax(th_vals)), 2),
                  "threshold_last": round(float(th_vals[~np.isnan(th_vals)][-1]), 2),
                  "total_crash_days": int(crash_days.sum())}
# 变体消融: method = sigma(扩张) / ewma_z(半衰252) / percentile
variants = {}
for meth in ["sigma", "ewma_z", "percentile"]:
    r = bt(method=meth)
    variants[meth] = {k: r[k] for k in ["total_ret", "calmar", "max_dd", "ann_ret", "trade_count"]}
res["A7"] = {"accel_audit": audit, "panic_audit": panic, "method_variants": variants}

# ── B1-fix 消融 ──
mats = load_mats(["纳指", "黄金", "创业"])
for bind in [False, True]:
    cfg = base_cfg(confirm_bind_identity=bind)
    actions, nav, user_nav = run_cfg(cfg, mats)
    m = metrics(user_nav, len(mats["all_dates"]))
    res.setdefault("B1fix", {})[f"bind={bind}"] = {**m, "n_actions": len(actions)}
res["B1fix"]["verdict"] = ("修复转正" if res["B1fix"]["bind=True"]["calmar"] >= res["B1fix"]["bind=False"]["calmar"] else "修复为负优化, 维持旧语义并文档化")

save(res, "a_group")
import json
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
