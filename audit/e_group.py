"""E组: 事件级专项审计 (长窗口 2015-01-05 起, 窄池)
E1 交易统计表 (四通道: 正常切换/快速通道/止损/抄底)
E2 exception_gap 通道审计 (持有≤2天被切子集)
E3 止盈后 5/10/20 日前向收益 (卖飞 vs 躲崩)
E4 抄底逐笔浮亏 (锁仓期最大浮亏/是否触止损/解锁盈亏)
E5 抄底 vs 空仓冲突审计
E6 空仓区间反事实 (持池内最强) + leave-one-interval-out
E7 上证空仓误伤事件表 (触发后纳指/黄金走势)
E8 恐慌抄底基准检验 (bootstrap vs 全部恐慌日)
E9 抄底挪动剖面 (±5天平移 + 随机对照)
"""
import sys
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd
from harness import FEED, bt, load_mats, ALL_CODES, save

START = "2015-01-05"
res = {}
base = bt(start_date=START)
dec = base["decisions"]
dates = base["dates"]
nav = np.array(base["nav"])
pos_series = base["position_series"]
pos_labels = base["position_labels"]
etf_names = [n for n in base["actual_universe"]]

# 价格序列(复权)对齐 dates
dfs = FEED.get_dataframe([ALL_CODES[n] for n in etf_names] + ["399006.SZ", "000001.SH"],
                         START.replace("-", ""), "20260812")
px = {}
for n in etf_names:
    s = dfs[ALL_CODES[n]].set_index("trade_date")["close"]
    px[n] = np.array([s.get(pd.Timestamp(d), np.nan) for d in dates])
cyb_idx = np.array([dfs["399006.SZ"].set_index("trade_date")["close"].get(pd.Timestamp(d), np.nan) for d in dates])
sh_idx = np.array([dfs["000001.SH"].set_index("trade_date")["close"].get(pd.Timestamp(d), np.nan) for d in dates])
date_i = {d: i for i, d in enumerate(dates)}

def fwd(arr, i, k):
    if i + k >= len(arr) or np.isnan(arr[i]): return None
    return (arr[i + k] / arr[i] - 1) * 100

# 重建逐笔交易: (入场决策, 出场决策) 配对
trades = []
entry = None
for tr in dec:
    a = tr["action"]
    if "买入" in a:
        asset = a.replace("买入", "").split("(")[0]
        entry = {"asset": asset, "entry_date": tr["date"], "entry_nav": tr["nav"],
                 "entry_reason": tr["reason"]}
    elif entry and ("卖出" in a or "→" in a):
        trades.append({**entry, "exit_date": tr["date"], "exit_nav": tr["nav"],
                       "ret": tr["pnl"], "days": tr["days"], "exit_reason": tr["reason"]})
        if "→" in a:
            asset = a.split("→")[1].split("(")[0]
            entry = {"asset": asset, "entry_date": tr["date"], "entry_nav": tr["nav"],
                     "entry_reason": tr["reason"]}
        else:
            entry = None
    elif "持有" in a and entry:
        trades.append({**entry, "exit_date": tr["date"], "exit_nav": tr["nav"],
                       "ret": tr["pnl"], "days": tr["days"], "exit_reason": "期末"})

def chan(r):
    er = (r["exit_reason"] or "").replace("%%", "%")
    if er == "恐慌抄底" or (r["entry_reason"] or "").replace("%%", "%") == "恐慌抄底": return "抄底"
    if er in ("止损", "绝对止损"): return "止损"
    if er == ">3%切换": return "快速通道"
    if er == "正常切换": return "正常切换"
    if er == "加速止盈": return "止盈"
    if er == "上证空仓": return "上证空仓"
    return "其他"

for t in trades: t["channel"] = chan(t)

def agg(ts):
    rets = [t["ret"] for t in ts if t["ret"] is not None]
    if not rets: return {}
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    return {"n": len(rets), "win_rate": round(100 * len(wins) / len(rets), 1),
            "avg_ret": round(np.mean(rets), 2), "median_ret": round(np.median(rets), 2),
            "avg_win": round(np.mean(wins), 2) if wins else None,
            "avg_loss": round(np.mean(losses), 2) if losses else None,
            "pl_ratio": round(abs(np.mean(wins) / np.mean(losses)), 2) if wins and losses else None,
            "max_single": round(max(rets), 2), "min_single": round(min(rets), 2),
            "avg_days": round(np.mean([t["days"] for t in ts if t["days"] is not None]), 1)}

# ── E1 ──
by_chan = {}
for c in ["正常切换", "快速通道", "止损", "止盈", "抄底", "上证空仓", "其他"]:
    ts = [t for t in trades if t["channel"] == c]
    if ts: by_chan[c] = agg(ts)
res["E1"] = {"by_channel": by_chan, "overall": agg(trades), "n_trades": len(trades)}

# ── E2: exception_gap 通道 ──
fast = [t for t in trades if (t["exit_reason"] or "").replace("%%", "%") == ">3%切换"]
norm = [t for t in trades if t["exit_reason"] == "正常切换"]
fast_days = [t["days"] for t in fast if t["days"] is not None]
short_cut = [t for t in fast if t["days"] is not None and t["days"] <= 2]
res["E2"] = {
    "fast": agg(fast), "normal": agg(norm),
    "fast_hold_days_hist": {str(d): fast_days.count(d) for d in sorted(set(fast_days))},
    "hold_le2_subset": {**agg(short_cut), "share_of_fast": round(100 * len(short_cut) / max(len(fast), 1), 1)},
    "verdict": "持有≤2天被切子集稳定亏→加最短持有" if short_cut and np.mean([t["ret"] for t in short_cut]) < 0 else "无系统性亏损",
}

# ── E3: 止盈后前向收益 ──
tps = [t for t in trades if t["exit_reason"] == "加速止盈"]
e3 = []
for t in tps:
    i = date_i.get(t["exit_date"])
    if i is None: continue
    arr = px[t["asset"]]
    e3.append({"date": t["exit_date"], "asset": t["asset"], "ret": round(t["ret"], 2),
               "fwd5": fwd(arr, i, 5), "fwd10": fwd(arr, i, 10), "fwd20": fwd(arr, i, 20)})
f5 = [x["fwd5"] for x in e3 if x["fwd5"] is not None]
f10 = [x["fwd10"] for x in e3 if x["fwd10"] is not None]
f20 = [x["fwd20"] for x in e3 if x["fwd20"] is not None]
res["E3"] = {"events": e3, "n": len(e3),
             "fwd5_median": round(np.median(f5), 2) if f5 else None,
             "fwd10_median": round(np.median(f10), 2) if f10 else None,
             "fwd20_median": round(np.median(f20), 2) if f20 else None,
             "fwd20_mean": round(np.mean(f20), 2) if f20 else None,
             "sold_early_share": round(100 * np.mean([x > 0 for x in f20]), 1) if f20 else None,
             "verdict": ""}

# ── E4: 抄底逐笔 ──
crash_trades = [t for t in trades if t["entry_reason"] == "恐慌抄底"]
e4 = []
for t in crash_trades:
    i0 = date_i.get(t["entry_date"]); i1 = date_i.get(t["exit_date"])
    if i0 is None: continue
    arr = px["创业"]
    lock_end = min(i0 + 7, len(arr) - 1)
    window = arr[i0:lock_end + 1]
    window = window[~np.isnan(window)]
    max_float = round(float((window.min() / arr[i0] - 1) * 100), 2) if len(window) else None
    e4.append({**t, "max_float_in_lock": max_float,
               "stopped": t["exit_reason"] in ("止损", "绝对止损")})
res["E4"] = {"crash_trades": e4, "n": len(e4),
             "verdict": "浮亏超-5%+跳空即异常" if any((x["max_float_in_lock"] or 0) < -5 for x in e4) else "锁仓期浮亏均在-5%内"}

# ── E5: 抄底 vs 空仓冲突 ──
mats = load_mats(etf_names, start=START)
sh_mom_u = mats["sh_mom"][mats["pre_days"]:]
ms_u = mats["mom20_mat"][:, mats["pre_days"]:]
e5 = []
for t in crash_trades:
    i = date_i.get(t["entry_date"])
    if i is None: continue
    sc_same_day = bool(sh_mom_u[i] > np.nanmax(ms_u[:, i])) if not np.isnan(sh_mom_u[i]) else None
    e5.append({"date": t["entry_date"], "ret": t["ret"], "sh_cash_same_day": sc_same_day,
               "exit_reason": t["exit_reason"]})
conflict = [x for x in e5 if x["sh_cash_same_day"]]
res["E5"] = {"events": e5, "conflict_n": len(conflict),
             "conflict_rets": [x["ret"] for x in conflict],
             "verdict": "冲突样本全盈利" if conflict and all(x["ret"] > 0 for x in conflict) else
                        ("无冲突样本" if not conflict else "冲突样本有亏损→重审")}

# ── E6: 空仓区间反事实 ──
intervals = []
i = 0
while i < len(pos_series):
    if pos_series[i] == -1:
        j = i
        while j + 1 < len(pos_series) and pos_series[j + 1] == -1: j += 1
        if j > i: intervals.append((i, j))
        i = j + 1
    else: i += 1
cf = []
for (a, b) in intervals:
    # 区间起点动量最强标的
    ms0 = ms_u[:, a]
    if np.isnan(ms0).all(): continue
    best = int(np.argmax(np.where(~np.isnan(ms0), ms0, -np.inf)))
    arr = px[etf_names[best]]
    if np.isnan(arr[a]) or np.isnan(arr[b]): cf_ret = None
    else: cf_ret = (arr[b] / arr[a] - 1) * 100
    cf.append({"start": dates[a], "end": dates[b], "days": b - a,
               "best_asset": etf_names[best], "counterfactual_ret": round(cf_ret, 2) if cf_ret is not None else None})
cf_valid = [x for x in cf if x["counterfactual_ret"] is not None]
# leave-one-interval-out: 反事实总超额 (连乘)
def total_cf(ex):
    prod = 1.0
    for k, x in enumerate(cf_valid):
        if k == ex: continue
        prod *= (1 + x["counterfactual_ret"] / 100)
    return (prod - 1) * 100
loio = {f"剔除{x['start']}~{x['end']}({x['best_asset']})": round(total_cf(k), 1)
        for k, x in enumerate(cf_valid)}
res["E6"] = {"intervals": cf, "n_cash_intervals": len(cf_valid),
             "counterfactual_total_%": round(total_cf(-1), 1),
             "loio": loio,
             "verdict": "剔除单一区间后反事实仍为负/微利→空仓非区间纪念品" if
                        all(v < 50 for v in loio.values()) else "优势集中于个别区间→区间纪念品嫌疑"}

# ── E7: 上证空仓误伤表 ──
sh_exits = [t for t in trades if t["exit_reason"] == "上证空仓"]
# 另需"触发日"= 卖出日; 统计卖出后 纳指/黄金 5/10/20日前向
e7 = []
for tr in dec:
    if tr["reason"] == "上证空仓" and ("卖出" in tr["action"] or "→" in tr["action"] or True):
        i = date_i.get(tr["date"])
        if i is None: continue
        row = {"date": tr["date"], "action": tr["action"]}
        for n in ["纳指", "黄金"]:
            if n in px:
                row[f"{n}_fwd10"] = fwd(px[n], i, 10)
                row[f"{n}_fwd20"] = fwd(px[n], i, 20)
        row["sh_fwd20"] = fwd(sh_idx, i, 20)
        e7.append(row)
hurt = [x for x in e7 if (x.get("纳指_fwd20") or 0) > 3 or (x.get("黄金_fwd20") or 0) > 3]
res["E7"] = {"events": e7, "n": len(e7),
             "sh_fwd20_median": round(float(np.median([x["sh_fwd20"] for x in e7 if x["sh_fwd20"] is not None])), 2),
             "hurt_events(纳指或黄金20日涨>3%)": len(hurt),
             "verdict": "量化保险保费: sh_fwd20 中位数 vs 误伤事件占比"}

# ── E8: 恐慌日基准检验 ──
m10_cyb = np.full(len(dates), np.nan); m10_cyb[10:] = (cyb_idx[10:] / cyb_idx[:-10] - 1) * 100
m10_sh = np.full(len(dates), np.nan); m10_sh[10:] = (sh_idx[10:] / sh_idx[:-10] - 1) * 100
ma200_cyb = pd.Series(cyb_idx).rolling(200).mean().values
ma200_sh = pd.Series(sh_idx).rolling(200).mean().values
panic_set = set()
for m10, idx, ma in [(m10_cyb, cyb_idx, ma200_cyb), (m10_sh, sh_idx, ma200_sh)]:
    s = pd.Series(m10)
    th = (s.expanding(min_periods=60).mean() - 2.5 * s.expanding(min_periods=60).std()).shift(1).values
    hit = np.where((m10 < th) & (idx < ma) & ~np.isnan(ma))[0]
    panic_set.update(hit.tolist())
panic_days = sorted(panic_set)
panic_fwd20 = [fwd(cyb_idx, i, 20) for i in panic_days]
panic_fwd20 = [x for x in panic_fwd20 if x is not None]
actual_entries = [date_i[t["entry_date"]] for t in crash_trades if t["entry_date"] in date_i]
actual_fwd20 = [fwd(cyb_idx, i, 20) for i in actual_entries]
actual_fwd20 = [x for x in actual_fwd20 if x is not None]
rng = np.random.default_rng(42)
boots = []
if len(panic_fwd20) >= len(actual_fwd20) and actual_fwd20:
    for _ in range(5000):
        boots.append(np.mean(rng.choice(panic_fwd20, size=len(actual_fwd20), replace=False)))
    pct = round(100 * np.mean(np.array(boots) <= np.mean(actual_fwd20)), 1)
else:
    pct = None
res["E8"] = {"panic_days_total": len(panic_days),
             "panic_fwd20_mean": round(np.mean(panic_fwd20), 2) if panic_fwd20 else None,
             "actual_entries_n": len(actual_fwd20),
             "actual_fwd20_mean": round(np.mean(actual_fwd20), 2) if actual_fwd20 else None,
             "bootstrap_percentile": pct,
             "verdict": "百分位>50 → 有选择增量; ≈50 → 纯恐慌反弹beta"}

# ── E9: 挪动剖面 ──
shifts = {}
for k in [-5, -3, -1, 0, 1, 3, 5]:
    vals = [fwd(cyb_idx, min(max(i + k, 0), len(cyb_idx) - 21), 20) for i in actual_entries]
    vals = [x for x in vals if x is not None]
    shifts[f"{k:+d}"] = round(np.mean(vals), 2) if vals else None
rand_days = rng.choice(range(200, len(cyb_idx) - 21), size=200, replace=False)
rand_fwd = [fwd(cyb_idx, int(i), 20) for i in rand_days]
rand_fwd = [x for x in rand_fwd if x is not None]
res["E9"] = {"shift_profile_fwd20_mean": shifts, "random_control": round(np.mean(rand_fwd), 2),
             "verdict": "0偏移为峰→时点有增量; 平坦≈随机→状态驱动(恐慌期普遍反弹)"}

save(res, "e_group")
import json
print(json.dumps({k: (v if k not in ("E6",) else {kk: v[kk] for kk in ["n_cash_intervals", "counterfactual_total_%", "verdict"]}) for k, v in res.items()}, ensure_ascii=False, indent=1, default=str)[:4000])
