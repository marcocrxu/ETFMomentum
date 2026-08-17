"""G 组 · 快速通道"卖飞"审计 + 分档切换(3%→50% / ≥5%→100%)消融 — 2026-08-17

用户观察: 通过 exception gap 切换容易卖飞; 提议 gap∈[3%,5%) 切50%、≥5% 切100%。
本脚本: ①逐笔统计快速通道卖飞率(前向5/10/20日被卖者vs买入者) ②按 gap 档位分层
        ③基线 vs 分档切换(fast_lane_staged) 的消融与逐笔对比。
输出: audit/out/g_staged.json
"""
import sys, os, json
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np

from harness import (load_mats, run_cfg, base_cfg, save, metrics,
                     trades_from_actions, BASE_UNIVERSE, BASE_START, BASE_END)

FAST_REASONS = {">3%切换", ">3%切换(50%)", "分档升级全切"}
COMBO_REASONS = {"卫星回撤", "卫星追新", "卫星止损", "卫星止盈",
                 "核心止损", "核心止盈", "组合止损", "组合止盈"}


def fwd_ret(pm, i, fd, n):
    p0 = pm[i, fd]
    p1 = pm[i, min(fd + n, pm.shape[1] - 1)]
    if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
        return None
    return round(float((p1 / p0 - 1) * 100), 2)


def fast_events(actions, mats):
    """快速通道卖出事件: 每笔 (被卖者, 买入者, gap, 前向收益)"""
    names = mats["etf_names"]; pre = mats["pre_days"]; pm = mats["price_mat"]
    out = []
    for day_idx, old_p, new_p, reason in actions:
        if reason not in FAST_REASONS or old_p < 0 or new_p < 0:
            continue
        fd = min(pre + day_idx, mats["n_full"] - 1)
        gap = float(mats["mom20_mat"][new_p, fd] - mats["mom20_mat"][old_p, fd])
        ev = {"date": str(mats["all_full"][fd])[:10], "from": names[old_p],
              "to": names[new_p], "gap": round(gap, 2), "reason": reason}
        for n in (5, 10, 20):
            f0, f1 = fwd_ret(pm, old_p, fd, n), fwd_ret(pm, new_p, fd, n)
            ev[f"fwd{n}_from"], ev[f"fwd{n}_to"] = f0, f1
            ev[f"sellfly{n}"] = (f0 > f1) if (f0 is not None and f1 is not None) else None
        out.append(ev)
    return out


def band_stats(evs, band):
    rows = [e for e in evs if e["reason"] == ">3%切换" and
            ((band == "3-5" and 3 <= e["gap"] < 5) or
             (band == ">=5" and e["gap"] >= 5) or band == "all")]
    if not rows:
        return None
    def _pct(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(float(np.mean(vals)), 2) if vals else None
    def _rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(100 * sum(vals) / len(vals), 1) if vals else None
    return {"n": len(rows), "avg_gap": _pct("gap"),
            "fwd5_from": _pct("fwd5_from"), "fwd5_to": _pct("fwd5_to"),
            "sellfly5%": _rate("sellfly5"),
            "fwd10_from": _pct("fwd10_from"), "fwd10_to": _pct("fwd10_to"),
            "sellfly10%": _rate("sellfly10"),
            "fwd20_from": _pct("fwd20_from"), "fwd20_to": _pct("fwd20_to"),
            "sellfly20%": _rate("sellfly20")}


def trade_stats(trades, reasons, label):
    rows = [t for t in trades if t.get("exit_reason") in reasons]
    if not rows:
        return {label: {"n": 0}}
    wins = [t for t in rows if t["ret"] > 0]
    rets = [t["ret"] for t in rows]
    s = {"n": len(rows), "win%": round(100 * len(wins) / len(rows), 1),
         "avg_ret": round(float(np.mean(rets)), 2),
         "pl": round(abs(float(np.mean([r for r in rets if r > 0]))) /
                     abs(float(np.mean([r for r in rets if r <= 0]))), 2)
         if any(r <= 0 for r in rets) else None,
         "avg_days": round(float(np.mean([t["days"] for t in rows])), 1)}
    return {label: s}


def main():
    mats = load_mats(BASE_UNIVERSE.split(","))
    out = {}

    # ── 基线 ──
    cfg0 = base_cfg()
    act0, nav0, user_nav0 = run_cfg(cfg0, mats)
    out["baseline"] = metrics(user_nav0, len(user_nav0))
    ev0 = fast_events(act0, mats)
    out["fast_events_baseline"] = ev0
    for band in ("3-5", ">=5", "all"):
        out[f"band_{band}"] = band_stats(ev0, band)
    t0 = trades_from_actions(act0, mats, user_nav0)
    out["trades_baseline"] = trade_stats(t0, FAST_REASONS, "快速通道(基线)")
    # 快速通道按 gap 档分层 (用事件 gap 匹配笔)
    fast_t0 = [t for t in t0 if t.get("exit_reason") == ">3%切换"]
    ev_by_date = {}
    for e in ev0:
        ev_by_date.setdefault((e["date"], e["from"], e["to"]), e)
    lo = [t for t in fast_t0
          if ev_by_date.get((t["exit_date"], t["asset"], t.get("switch_to")), {}).get("gap", 9) < 5]
    hi = [t for t in fast_t0
          if ev_by_date.get((t["exit_date"], t["asset"], t.get("switch_to")), {}).get("gap", 9) >= 5]
    def _s(rows, label):
        if not rows:
            return {label: {"n": 0}}
        wins = [t for t in rows if t["ret"] > 0]
        rets = [t["ret"] for t in rows]
        return {label: {"n": len(rows), "win%": round(100 * len(wins) / len(rows), 1),
                        "avg_ret": round(float(np.mean(rets)), 2),
                        "avg_days": round(float(np.mean([t["days"] for t in rows])), 1)}}
    out["fast_gap_3_5_trades"] = _s(lo, "gap∈[3,5)")
    out["fast_gap_ge5_trades"] = _s(hi, "gap≥5")

    # ── 分档切换 ──
    cfg1 = base_cfg(fast_lane_staged=True, fast_full_gap=5.0)
    act1, nav1, user_nav1 = run_cfg(cfg1, mats)
    out["staged"] = metrics(user_nav1, len(user_nav1))
    out["staged_delta"] = {k: round(out["staged"][k] - out["baseline"][k], 2)
                           for k in ("total_ret", "mdd", "calmar")}
    ev1 = fast_events(act1, mats)
    out["fast_events_staged"] = ev1
    t1 = trades_from_actions(act1, mats, user_nav1)
    out["trades_staged"] = trade_stats(
        t1, FAST_REASONS, "快速通道(分档)")
    out["trades_staged_combo"] = trade_stats(t1, COMBO_REASONS, "组合仓归并(分档新增)")
    # 动作数对比
    from collections import Counter
    out["action_count_baseline"] = dict(Counter(a[3] for a in act0))
    out["action_count_staged"] = dict(Counter(a[3] for a in act1))

    save(out, "g_staged")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("baseline", "staged", "staged_delta",
                               "band_3-5", "band_>=5", "band_all",
                               "trades_baseline", "trades_staged",
                               "trades_staged_combo", "fast_gap_3_5_trades",
                               "fast_gap_ge5_trades", "action_count_baseline",
                               "action_count_staged")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
