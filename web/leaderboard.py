"""打榜 — 按参数组合批量回测排行 (主引擎 + yanyan), 支持按策略过滤与指标排序

网格 (研究固化的敏感参数轴, 组合数有上限):
- 主引擎 ETF: exception_gap × 空仓冷却 × 纳指QQQ混合   = 4×2×2 = 16 组合
- yanyan:     轮动缓冲 × 过热门槛 × 过热Q × 牛熊MA     = 3×3×2×3 = 54 组合
数据只加载一次 (yanyan 快速路径直接调引擎); 结果持久化到 web/leaderboard_last.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

from strategy.yanyan import YanyanConfig, UNIVERSE, compute_yanyan
from web.bt_runner import run_backtest, FEED

LAST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard_last.json")

# ── 主引擎打榜网格 ──
ETF_GRID = []
for gap in (2.5, 3.0, 3.5, 4.0):
    for cd in (0, 3):
        for qqq_on, qqq_w in ((False, 0.0), (True, 0.7)):
            p = {"universe": "纳指,黄金,创业",
                 "exception_gap": gap, "sh_cash_cooldown": cd}
            label = "gap=%.1f 冷却=%d QQQ=%s" % (gap, cd, "0.7" if qqq_on else "关")
            if qqq_on:
                p["nz_mom_qqq_on"] = True
                p["nz_mom_qqq_w"] = qqq_w
            ETF_GRID.append((label, p))

# ── yanyan 打榜网格 ──
YANYAN_GRID = []
for buf in (1.0, 2.0, 3.0):
    for oh_mom in (30.0, 40.0, 50.0):
        for oh_q in (0.0, 3.0):
            for rma in (150, 200, 250):
                YANYAN_GRID.append(
                    ("buf=%.0f 过热=%.0f Q=%.0f MA=%d" % (buf, oh_mom, oh_q, rma),
                     {"switch_buffer": buf, "overheat_mom": oh_mom,
                      "overheat_q": oh_q, "regime_ma": rma}))

YANYAN_UNIVERSES = {
    "全池10只": "",
    "精简TOP6": "沪深300,中证500,创业板,红利低波,纳指,港股创新药",
    "跨境黄金3只": "纳指,港股创新药,黄金",
}

METRICS = ("calmar", "total_ret", "ann_ret", "sharpe", "mdd")


def _row(strategy, label, m, extra=None):
    r = {"strategy": strategy, "params": label,
         "total_ret": m["total_ret"],
         "mdd": m.get("mdd", m.get("max_dd")),
         "calmar": m.get("calmar"), "ann_ret": m.get("ann_ret"),
         "sharpe": m.get("sharpe"), "trade_count": m.get("trade_count"),
         "neg_years": m.get("neg_years")}
    if extra:
        r.update(extra)
    return r


def _metrics_from_log(nav_log, actions=None):
    nav = np.array(nav_log)
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(len(nav) / 252, 0.01)
    ann = (nav[-1] ** (1 / years) - 1) * 100 if nav[-1] > 0 else -100.0
    dr = np.diff(nav) / nav[:-1]
    dr = dr[~np.isnan(dr)]
    sharpe = round(float(np.mean(dr) / np.std(dr) * np.sqrt(252)), 2) \
        if len(dr) > 10 and np.std(dr) > 1e-9 else 0.0
    m = {"total_ret": round(float((nav[-1] - 1) * 100), 1),
         "ann_ret": round(float(ann), 2), "mdd": round(mdd, 2),
         "calmar": round(ann / abs(mdd), 2) if mdd else 0.0,
         "sharpe": sharpe}
    if actions is not None:
        m["trade_count"] = len(actions)
    return m


def _run_yanyan_grid(start, end, universe=""):
    """数据只加载一次, 直接调引擎跑全部 yanyan 组合"""
    names = [u[0] for u in UNIVERSE]
    groups = [u[2] for u in UNIVERSE]
    codes = [u[1] for u in UNIVERSE]

    dfs = FEED.get_dataframe(codes, "20150301", end.replace("-", ""))
    _all = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
    _all = [d for d in _all if pd.Timestamp("20150301") <= d <= pd.Timestamp(end)]
    if not _all:
        return []

    def bp(code):
        s = dfs[code].set_index("trade_date")["close"]
        return np.array([s.get(d, np.nan) for d in _all])
    pm = np.array([bp(c) for c in codes])
    hs = pm[names.index("沪深300")]

    # universe 子集
    rank_all = [i for i, g in enumerate(groups) if g != "C"]
    uni_raw = [x.strip() for x in (universe or "").split(",") if x.strip()]
    sel = [names.index(n) for n in uni_raw if n in names and names.index(n) in rank_all]
    if len(sel) < 2:
        sel = rank_all
    cash_i = names.index("货币")
    sub_idx = sel + [cash_i]
    names_sub = [names[i] for i in sel] + ["货币"]
    groups_sub = [groups[i] for i in sel] + ["C"]

    day0 = next(i for i, d in enumerate(_all) if str(d)[:10] >= start)
    day1 = next(i for i, d in enumerate(_all) if str(d)[:10] >= end)
    lo = max(0, day0 - 220)
    pm_sub = pm[sub_idx, lo:day1 + 1]
    hs_sub = hs[lo:day1 + 1]

    rows = []
    for label, p in YANYAN_GRID:
        cfg = YanyanConfig(**p)
        actions, nav, log = compute_yanyan(cfg, names_sub, groups_sub, pm_sub, hs_sub,
                                           pm_sub.shape[1])
        m = _metrics_from_log(log, actions)
        rows.append(_row("yanyan", label, m))
    return rows


def run_leaderboard(params):
    strategy = params.get("strategy", "all")       # all | etf | yanyan
    metric = params.get("metric", "calmar")
    if metric not in METRICS:
        metric = "calmar"
    start = params.get("start_date", "2016-07-07")
    end = params.get("end_date", "2026-08-12")
    universe = params.get("universe", "")

    t0 = time.time()
    rows = []
    errors = []
    if strategy in ("all", "etf"):
        for label, p in ETF_GRID:
            p2 = dict(p)
            p2["start_date"] = start
            p2["end_date"] = end
            r = run_backtest(p2)
            if "error" in r:
                errors.append(f"ETF[{label}]: {r['error']}")
                continue
            rows.append(_row("etf", label, r))
    if strategy in ("all", "yanyan"):
        rows += _run_yanyan_grid(start, end, universe)

    rows.sort(key=lambda r: -(r[metric] if r[metric] is not None else -1e9))
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    result = {"rows": rows, "metric": metric, "errors": errors,
              "elapsed": round(time.time() - t0, 1),
              "window": f"{start} ~ {end}",
              "server_time": time.strftime("%H:%M:%S")}
    # 持久化最近一次榜单
    tmp = LAST_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LAST_FILE)
    return result


def load_last():
    if not os.path.exists(LAST_FILE):
        return None
    try:
        with open(LAST_FILE) as f:
            return json.load(f)
    except Exception:
        return None
