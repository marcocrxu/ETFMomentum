"""E2 消融: 快速通道(>3%切换)最短持有 fast_min_hold ∈ {0,1,2,3,4}
背景: E2 发现"持有≤2天被快速通道切走"6笔, 胜率16.7%, 盈亏比0.09, 稳定小亏。
开关语义: hd < fast_min_hold 时 >3%切换不触发(止损/止盈/空仓等保护不受影响);
         持仓满N天后若 gap 仍 >=3% 且确认计数满足, 再切。
"""
import sys
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, json
from harness import load_mats, run_cfg, base_cfg, metrics, trades_from_actions, save

mats = load_mats(["纳指", "黄金", "创业"])
n_days = len(mats["all_dates"])

def fast_stats(trades):
    fast = [t for t in trades if (t["exit_reason"] or "").replace("%%", "%") == ">3%切换"]
    rets = [t["ret"] for t in fast]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    short = [t for t in fast if t["days"] is not None and t["days"] <= 2]
    return {
        "n_fast": len(fast),
        "win_rate": round(100 * len(wins) / len(rets), 1) if rets else None,
        "pl": round(abs(np.mean(wins) / np.mean(losses)), 2) if wins and losses else None,
        "avg": round(np.mean(rets), 2) if rets else None,
        "short_le2_n": len(short),
        "short_le2_sum": round(sum(t["ret"] for t in short), 2),
    }

res = {}
per_setting_trades = {}
for fm in [0, 1, 2, 3, 4]:
    cfg = base_cfg(fast_min_hold=fm)
    actions, final_nav, user_nav = run_cfg(cfg, mats)
    trades = trades_from_actions(actions, mats, user_nav)
    per_setting_trades[fm] = trades
    res[str(fm)] = {**metrics(user_nav, n_days), "n_actions": len(actions),
                    **fast_stats(trades)}

# ── 基线里"≤2天被切"的 6 笔, 在 fm=3 下分别变成了什么 ──
base = per_setting_trades[0]
fm3 = per_setting_trades[3]
base_short = [t for t in base
              if (t["exit_reason"] or "").replace("%%", "%") == ">3%切换"
              and t["days"] is not None and t["days"] <= 2]

def find(trades, asset, entry_date):
    """fm3 账本里找同一资产、入场日接近(±6天)的那笔"""
    cands = [t for t in fm3 if t["asset"] == asset
             and abs((np.datetime64(t["entry_date"]) - np.datetime64(entry_date))
                     / np.timedelta64(1, "D")) <= 6]
    return cands[0] if cands else None

trace = []
for t in base_short:
    m = find(fm3, t["asset"], t["entry_date"])
    trace.append({
        "asset": t["asset"], "entry": t["entry_date"],
        "base_exit": t["exit_date"], "base_days": t["days"], "base_ret": round(t["ret"], 2),
        "fm3_exit": m["exit_date"] if m else None,
        "fm3_days": m["days"] if m else None,
        "fm3_ret": round(m["ret"], 2) if m else None,
        "fm3_exit_reason": (m["exit_reason"] or "").replace("%%", "%") if m else None,
    })

res["_trace_base_short_under_fm3"] = trace
res["_verdict"] = "fm=0 应与基线 3595.13/2.09 完全一致(默认行为不变)"
save(res, "e2_minhold")
print(json.dumps(res, ensure_ascii=False, indent=1))
