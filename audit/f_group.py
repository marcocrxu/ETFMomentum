"""F组: 压力测试与情景推演
F1 2015式连续暴跌沙盘 (-8%×5日, 逐日走全部规则; 跌停无法成交的口径问题)
F2 抄底锁仓压力测试 (连续暴跌测锁仓期最大单笔损失)
F3 2018 逐月参照表 (净值+持仓)
F4 池子规模梯度 (3/4/8/14, 时变宇宙)
"""
import sys
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd
from strategy.signal import SignalConfig, compute_signals
from harness import bt, save

res = {}

# ── F1: 2015 式暴跌沙盘 ──
# 300 天平盘(MA200/阈值就绪) → 创业 -8%×5 连续跌停 → 之后阴跌 -2%/天 20 天
n = 400
cyb_idx = np.concatenate([np.full(300, 4000.0),
                          4000 * 0.92 ** np.arange(1, 6),
                          4000 * 0.92 ** 5 * (1 - 0.02) ** np.arange(1, 96)])
sh_idx = np.concatenate([np.full(300, 5000.0),
                         5000 * 0.94 ** np.arange(1, 6),
                         5000 * 0.94 ** 5 * (1 - 0.01) ** np.arange(1, 96)])
pe_cyb = np.concatenate([np.full(300, 100.0), 100 * 0.92 ** np.arange(1, 6),
                         100 * 0.92 ** 5 * (1 - 0.02) ** np.arange(1, 96)])
pe_nas = np.full(n, 100.0) * (1.0005 ** np.arange(n))   # 纳指缓慢上行(避险)
pe_gold = np.full(n, 100.0) * (1.0003 ** np.arange(n))  # 黄金微涨
price_mat = np.array([pe_nas, pe_gold, pe_cyb])
mom20 = np.full(price_mat.shape, np.nan); mom20[:, 20:] = (price_mat[:, 20:] / price_mat[:, :-20] - 1) * 100
mom10 = np.full(price_mat.shape, np.nan); mom10[:, 10:] = (price_mat[:, 10:] / price_mat[:, :-10] - 1) * 100
sh_mom = np.full(n, np.nan)
sh_mom[20:] = (sh_idx[20:] / sh_idx[:-20] - 1) * 100

cfg = SignalConfig()  # 默认全开关
acts, nav_f, nav_log = compute_signals(cfg, ["纳指", "黄金", "创业"], price_mat,
                                       mom20, mom10, sh_mom, cyb_idx, sh_idx, n)
day_log = []
pos_now = -1
for t in range(295, 340):
    daily = {"t": t, "cyb_px": round(float(pe_cyb[t]), 2)}
    for a in acts:
        if a[0] == t:
            daily["action"] = (a[1], a[2], a[3])
    day_log.append(daily)
acts_zone = [a for a in acts if a[0] >= 295]
navs = np.array(nav_log)
mdd_zone = float(np.min(navs[295:] / np.maximum.accumulate(navs[:296])[-1] - 1) * 100) if len(navs) > 295 else None
res["F1"] = {
    "actions_after_t295": [(a[0], a[1], a[2], a[3]) for a in acts_zone],
    "final_nav": round(float(nav_f), 4),
    "verdict": "",
    "limit_down_note": "引擎按收盘价成交, 不建模跌停无法成交; 连续跌停日反复触发止损/抄底时, "
                       "实盘可能无法按信号价成交——见建议规则",
}

# ── F2: 抄底锁仓压力: -6%×15 连续暴跌 ──
n2 = 400
cyb2 = np.concatenate([np.full(300, 4000.0), 4000 * 0.94 ** np.arange(1, 101)])
sh2 = np.full(n2, 5000.0)
pe2 = np.concatenate([np.full(300, 100.0), 100 * 0.94 ** np.arange(1, 101)])
price_mat2 = np.array([np.full(n2, 100.0), np.full(n2, 100.0), pe2])
mom20b = np.full(price_mat2.shape, np.nan); mom20b[:, 20:] = (price_mat2[:, 20:] / price_mat2[:, :-20] - 1) * 100
mom10b = np.full(price_mat2.shape, np.nan); mom10b[:, 10:] = (price_mat2[:, 10:] / price_mat2[:, :-10] - 1) * 100
sh_mom2 = np.full(n2, np.nan); sh_mom2[20:] = 0.0
cfg2 = SignalConfig()
acts2, nav_f2, nav_log2 = compute_signals(cfg2, ["纳指", "黄金", "创业"], price_mat2,
                                          mom20b, mom10b, sh_mom2, cyb2, sh2, n2)
crash_entries = [a for a in acts2 if a[3] == "恐慌抄底"]
worst = None
for a in crash_entries:
    t0 = a[0]
    lock_end = min(t0 + 7, n2 - 1)
    window = pe2[t0:lock_end + 1]
    float_loss = float((window.min() / pe2[t0] - 1) * 100)
    exit_px_i = min(t0 + 30, n2 - 1)
    held_ret = float((pe2[exit_px_i] / pe2[t0] - 1) * 100)
    if worst is None or float_loss < worst["float_loss"]:
        worst = {"entry_t": t0, "float_loss": round(float_loss, 2),
                 "ret_after_30d": round(held_ret, 2)}
res["F2"] = {
    "crash_entries": [(a[0], a[1], a[2]) for a in crash_entries],
    "all_actions_295+": [(a[0], a[1], a[2], a[3]) for a in acts2 if a[0] >= 295][:20],
    "worst_lock_float_loss": worst,
    "verdict": "锁仓7天×每日-6% → 单笔试探底; 尾部定价=低频大亏的期望损失",
}

# ── F3: 2018 逐月参照 ──
base = bt()
monthly18 = {k: v for k, v in base["monthly"].items() if k.startswith("2018")}
dates = base["dates"]; ps = base["position_series"]; pl = base["position_labels"]
hold18 = {}
for i, d in enumerate(dates):
    if d.startswith("2018"):
        m = d[:7]; lb = pl[ps[i] + 1] if ps[i] >= -1 else "?"
        hold18.setdefault(m, {})[lb] = hold18.setdefault(m, {}).get(lb, 0) + 1
res["F3"] = {"monthly_nav_ret": monthly18, "holdings_by_month": hold18,
             "yearly_2018": base["yearly"].get("2018")}

# ── F4: 池子规模梯度 ──
pools = {
    3: "纳指,黄金,创业",
    4: "纳指,黄金,创业,沪深300",
    8: "纳指,黄金,创业,沪深300,上证50,红利低波,5年国债,科创50",
    14: "纳指,黄金,创业,科创50,沪深300,上证50,中证2000,港股创新药,有色,通信,半导体,红利低波,5年国债,30年国债",
}
f4 = {}
for k, u in pools.items():
    r = bt(universe=u)
    if "error" in r:
        f4[k] = {"error": r["error"]}; continue
    sh_exits = sum(1 for d in r["decisions"] if d["reason"] == "上证空仓")
    f4[k] = {"total_ret": r["total_ret"], "ann": r["ann_ret"], "mdd": r["max_dd"],
             "calmar": r["calmar"], "sharpe": r["sharpe"], "trades": r["trade_count"],
             "sh_cash_exits": sh_exits, "neg_years": r["neg_years"],
             "yearly": r["yearly"]}
res["F4"] = {"pools": f4,
             "verdict": "扩池未验证前不许扩; 时变宇宙(各时点只含已上市ETF)由NaN掩码自然实现"}

save(res, "f_group")
import json
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:4000])
