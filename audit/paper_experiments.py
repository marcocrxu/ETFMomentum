"""2026-08-20 分析页数据生成: 用当前默认参数 + 最新数据重跑全部核心实验
输出 audit/out/paper_2026.json, 供 web/templates/analysis.html (paper 版) 使用。
只依赖 web.bt_runner.run_backtest (权威引擎)。"""
import sys, os, json, time
sys.path.insert(0, "/root/etf_strategy")
import numpy as np
from web.bt_runner import run_backtest

END = "2026-08-20"
START = "2016-07-07"
LONG_START = "2015-01-05"
UNIV = "纳指,黄金,创业"

def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

def core(r):
    return {k: r.get(k) for k in ("total_ret", "ann_ret", "max_dd", "calmar",
                                  "sharpe", "volatility", "trade_count",
                                  "neg_years", "last_position", "checks", "yearly")}

def run(params):
    return run_backtest(params)

def nav_stats(navs, dates=None):
    nav = np.array(navs, dtype=float)
    nav = nav[~np.isnan(nav) & (nav > 0)]
    if len(nav) < 2:
        return None
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(len(nav) / 252.0, 0.01)
    ann = (nav[-1] ** (1 / years) - 1) * 100
    dr = np.diff(nav) / nav[:-1]
    sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 1e-9 else 0.0
    return {"total_ret": round((nav[-1] - 1) * 100, 1), "ann": round(float(ann), 2),
            "mdd": round(mdd, 2), "calmar": round(float(ann / abs(mdd)), 2) if mdd else None,
            "sharpe": round(sharpe, 2)}

def dd_episodes(dates, nav, dd, threshold=-10.0):
    """回撤事件: 峰值日/谷底日/修复日/深度/修复交易日数"""
    eps, peak_i, trough_i = [], None, None
    prev_zero = True
    for i, v in enumerate(dd):
        if v is None:
            continue
        if v == 0.0 and not prev_zero:
            if trough_i is not None and dd[trough_i] <= threshold:
                eps.append((peak_i, trough_i, i))
            peak_i, trough_i = None, None
        if peak_i is None or (trough_i is not None and v < dd[trough_i]):
            if peak_i is None:
                peak_i = i
            trough_i = i
        prev_zero = (v == 0.0)
    if not prev_zero and trough_i is not None and dd[trough_i] <= threshold:
        # 期末仍在水下: 未修复
        eps.append((peak_i, trough_i, len(dd)))
    out = []
    for p, t, r in eps:
        recovered = r < len(dates)
        out.append({"peak": dates[p], "trough": dates[t],
                    "recover": dates[r] if recovered else None,
                    "depth": round(dd[t], 2), "days": t - p,
                    "recover_days": r - t if recovered else None})
    return out

def trades_from_decisions(decisions):
    """从 decisions 重建逐笔交易: 每条含 asset, entry, exit, pnl, reason, days"""
    trades, entry = [], None
    for d in decisions:
        act = d["action"]
        if act.startswith("买入"):
            entry = {"asset": act[2:].split("(")[0], "entry": d["date"],
                     "reason": d["reason"]}
        elif act.startswith("卖出"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                trades.append(entry)
            entry = None
        elif "→" in act:
            # 切换行同时关闭旧腿并开启新腿 (与 build_analysis.py 同口径, 2026-08-23 修)
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                trades.append(entry)
            entry = {"asset": act.split("→")[1].split("(")[0], "entry": d["date"],
                     "reason": d["reason"]}
        if act.startswith("持有"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                trades.append(entry)
            entry = None
    return trades

def channel_stats(trades):
    groups = {}
    for t in trades:
        g = groups.setdefault(t["exit_reason"], [])
        g.append(t["pnl"])
    out = {}
    for k, v in groups.items():
        arr = np.array(v)
        out[k] = {"n": len(arr), "win_rate": round(float(np.mean(arr > 0) * 100), 1),
                  "avg": round(float(np.mean(arr)), 2), "median": round(float(np.median(arr)), 2),
                  "total": round(float(np.sum(arr)), 1)}
    return out

def main():
    results = {"meta": {"generated": "2026-08-20", "end": END, "start": START,
                        "universe": UNIV, "engine": "web.bt_runner.run_backtest"}}
    t0 = time.time()
    base = {"universe": UNIV, "start_date": START, "end_date": END}

    # ── 1. 基线 (当前默认参数) ──
    r = run(dict(base))
    results["baseline"] = core(r)
    results["baseline"]["bench"] = nav_stats(r["benchmark"])
    results["baseline"]["benchmark"] = r["benchmark"]   # 等权基准净值序列(供分析页图)
    results["baseline"]["etf_navs"] = {k: nav_stats(v) for k, v in r["etf_navs"].items()}
    results["baseline"]["dd_episodes"] = dd_episodes(r["dates"], r["nav"], r["drawdown"])
    # 2026-06/07 局部回撤详情 (旧页最大回撤段, 新默认下已不是最深)
    _dd = r["drawdown"]; _dts = r["dates"]
    _seg = [(_dts[i], _dd[i]) for i in range(len(_dts))
            if "2026-06-01" <= _dts[i] <= "2026-08-20" and _dd[i] is not None]
    results["baseline"]["dd_2026_seg"] = {"min": min(x[1] for x in _seg),
                                          "trough_date": min(_seg, key=lambda x: x[1])[0],
                                          "end": _seg[-1][1]}
    results["baseline"]["trades"] = trades_from_decisions(r["decisions"])
    results["baseline"]["channels"] = channel_stats(results["baseline"]["trades"])
    results["baseline"]["crash_entries"] = [d for d in r["decisions"] if d["reason"] == "恐慌抄底"]
    results["baseline"]["decisions"] = r["decisions"]
    results["baseline"]["monthly"] = r["monthly"]
    results["baseline"]["nav"] = r["nav"]; results["baseline"]["dates"] = r["dates"]
    results["baseline"]["drawdown"] = r["drawdown"]
    print("baseline", core(r)["total_ret"], "%.0fs" % (time.time() - t0))

    # ── 2. 消融 (单开关关掉) ──
    abl = {}
    ab_defs = [
        ("sh_cash_off", {"sh_cash": False}),
        ("abs_stop_off", {"abs_stop_on": False}),
        ("accel_off", {"accel_on": False}),
        ("crash_off", {"crash_on": False}),
        ("sh_cash_cooldown0", {"sh_cash_cooldown": 0}),
        ("accel_cooldown0", {"accel_cooldown": 0}),
        ("accel_cd_all_assets", {"accel_cooldown_gold_only": False}),
        ("accel_cooldown5", {"accel_cooldown": 5}),
        ("nz_qqq_off", {"nz_mom_qqq_on": False}),
        ("exception_gap999", {"exception_gap": 999}),
        ("risk_adj_on", {"risk_adj_on": True}),
        ("abs_mom_on", {"abs_mom_on": True}),
        ("defensive_on", {"defensive_on": True}),
        ("top_k2", {"top_k": 2}),
        ("mom_skip3", {"mom_skip": 3}),
        ("stop_cooldown3", {"stop_cooldown": 3}),
        ("multi_mom_on", {"multi_mom_on": True}),
        ("fip_on", {"fip_on": True}),
        ("sigma_stop_floor3", {"stop_sigma": 3.0, "stop_sigma_floor": True}),
    ]
    for label, over in ab_defs:
        p = dict(base); p.update(over)
        r = run(p)
        abl[label] = core(r)
        print("abl", label, core(r)["total_ret"], "%.0fs" % (time.time() - t0))
    results["ablation"] = abl

    # ── 3. 参数网格 ──
    grids = {}
    grids["mom_period"] = {}
    for m in range(10, 31):
        p = dict(base); p["mom_period"] = m
        grids["mom_period"][m] = core(run(p))
    grids["stop_loss"] = {s: core(run(dict(base, stop_loss=s))) for s in (-3, -8)}
    grids["confirm_days"] = {c: core(run(dict(base, confirm_days=c))) for c in (1, 3)}
    grids["exception_gap"] = {g: core(run(dict(base, exception_gap=g))) for g in (2.0, 5.0)}
    grids["crash_lock"] = {l: core(run(dict(base, crash_lock=l))) for l in (5, 14, 21, 25, 30)}
    grids["accel_sigma"] = {s: core(run(dict(base, accel_sigma=s))) for s in (2.5, 3.5)}
    grids["min_hold"] = {h: core(run(dict(base, min_hold=h))) for h in (5, 15)}
    grids["min_gap"] = {g: core(run(dict(base, min_gap=g))) for g in (0.5, 2.0)}
    grids["sh_cash_cooldown"] = {c: core(run(dict(base, sh_cash_cooldown=c))) for c in (0, 1, 2, 4, 5, 10)}
    grids["crash_alloc"] = {50: core(run(dict(base, crash_alloc=50)))}
    grids["slippage"] = {s: core(run(dict(base, slippage=s))) for s in (0.0005, 0.001, 0.002)}
    results["grids"] = grids
    print("grids done %.0fs" % (time.time() - t0))

    # ── 4. 长窗口 (2015-01-05 起, 含 2015 股灾) ──
    lw = {"universe": UNIV, "start_date": LONG_START, "end_date": END}
    rl = run(dict(lw))
    results["long_window"] = {"baseline": core(rl),
                              "cooldown0": core(run(dict(lw, sh_cash_cooldown=0))),
                              "cooldown5": core(run(dict(lw, sh_cash_cooldown=5))),
                              "sigma_stop_floor3": core(run(dict(lw, stop_sigma=3.0, stop_sigma_floor=True))),
                              "dd_episodes": dd_episodes(rl["dates"], rl["nav"], rl["drawdown"])}
    print("long window done %.0fs" % (time.time() - t0))

    # ── 5. 两半窗口 (walk-forward: 2016-2021 定参, 2021-2026 验证) ──
    h1 = {"universe": UNIV, "start_date": "2016-07-07", "end_date": "2021-08-20"}
    h2 = {"universe": UNIV, "start_date": "2021-08-23", "end_date": END}
    halves = {}
    halves["h1"] = {"baseline": core(run(dict(h1)))}
    halves["h2"] = {"baseline": core(run(dict(h2)))}
    halves["h1"]["mom"] = {m: core(run(dict(h1, mom_period=m))) for m in range(10, 31)}
    halves["h2"]["mom"] = {m: core(run(dict(h2, mom_period=m))) for m in range(10, 31)}
    # 三年近期窗口 (2023-08 ~ 2026-08, 最新市场节奏)
    h3 = {"universe": UNIV, "start_date": "2023-08-21", "end_date": END}
    halves["h3"] = {"baseline": core(run(dict(h3)))}
    halves["h3"]["mom"] = {m: core(run(dict(h3, mom_period=m))) for m in range(10, 31)}
    results["halves"] = halves
    print("halves done %.0fs" % (time.time() - t0))

    os.makedirs("/root/etf_strategy/audit/out", exist_ok=True)
    with open("/root/etf_strategy/audit/out/paper_2026.json", "w") as f:
        json.dump(clean(results), f, ensure_ascii=False, indent=1)
    print("saved audit/out/paper_2026.json  total %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
