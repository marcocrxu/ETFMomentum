"""yanyan 策略回测编排 — 供 web/app.py 单独页面使用
返回 schema 与 bt_runner.run_backtest 一致 (dates/nav/benchmark/drawdown/yearly/monthly/
decisions/position_series/position_labels/etf_navs/...), 前端直接复用 charts.js
"""
import datetime as _dt

import numpy as np
import pandas as pd

from strategy.yanyan import (YanyanConfig, UNIVERSE, compute_yanyan,
                             rolling_vol20, mom)
from web.bt_runner import FEED, clean_nan

DEFAULT_START = "2016-07-07"
DEFAULT_END = "2026-08-12"


def run_yanyan(params):
    cfg = YanyanConfig(
        regime_ma=int(params.get("regime_ma", 200)),
        mom_period=int(params.get("mom_period", 20)),
        vol_period=int(params.get("vol_period", 20)),
        switch_buffer=float(params.get("switch_buffer", 2.0)),
        overheat_mom=float(params.get("overheat_mom", 40.0)),
        overheat_q=float(params.get("overheat_q", 3.0)),
        commission=float(params.get("commission", 0.0001)),
        slippage=float(params.get("slippage", 0.0)),
        cash_park=bool(params.get("cash_park", True)),
        cash_skip_buffer=bool(params.get("cash_skip_buffer", True)),
    )
    bt_start = params.get("start_date", DEFAULT_START)
    bt_end = params.get("end_date", DEFAULT_END)

    names = [u[0] for u in UNIVERSE]
    codes = [u[1] for u in UNIVERSE]
    groups = [u[2] for u in UNIVERSE]

    # 预热: 牛熊 MA200 需 320 自然日
    _warmup = max(cfg.regime_ma + 60, 320)
    _bt_start_dt = _dt.datetime.strptime(bt_start.replace("-", ""), "%Y%m%d")
    _fetch_start = (_bt_start_dt - _dt.timedelta(days=_warmup)).strftime("%Y%m%d")
    dfs = FEED.get_dataframe(codes, _fetch_start, bt_end.replace("-", ""))
    if not dfs:
        return {"error": "数据加载失败"}

    _fetch_ts = pd.Timestamp(_fetch_start)
    _end_ts = pd.Timestamp(bt_end.replace("-", ""))
    for code in list(dfs.keys()):
        df = dfs[code]
        df = df[(df["trade_date"] >= _fetch_ts) & (df["trade_date"] <= _end_ts)]
        if len(df) == 0:
            del dfs[code]
        else:
            dfs[code] = df.reset_index(drop=True)

    _all_full = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
    _all_full = [d for d in _all_full if _fetch_ts <= d <= _end_ts]
    all_dates = [d for d in _all_full if pd.Timestamp(bt_start) <= d]
    n_full = len(_all_full)
    n = len(all_dates)
    _pre_days = n_full - n
    if n == 0 or n_full == 0:
        return {"error": "数据加载失败"}

    def _bp(code):
        if code not in dfs:
            return np.full(n_full, np.nan)
        s = dfs[code].set_index("trade_date")["close"]
        return np.array([s.get(d, np.nan) for d in _all_full])

    price_mat = np.array([_bp(c) for c in codes])
    hs300_p = price_mat[names.index("沪深300")]

    # ── 标的池可选: universe=逗号分隔名称 (默认全部10只排名标的; 货币为固定空仓停靠, 不入选择) ──
    rank_names_all = [n for n, g in zip(names, groups) if g != "C"]
    _uni_raw = [x.strip() for x in str(params.get("universe", "")).split(",") if x.strip()]
    sel_names = [n for n in _uni_raw if n in rank_names_all]
    if len(sel_names) == 0:
        sel_names = rank_names_all              # 未传/全无效 → 默认全池
    sel_names = sorted(set(sel_names), key=rank_names_all.index)
    if len(sel_names) < 2:
        return {"error": "至少选择 2 个排名标的"}
    cash_i_full = names.index("货币")
    sel_idx = [names.index(n) for n in sel_names]
    _price_sel = price_mat[sel_idx + [cash_i_full]]
    _names_sel = sel_names + ["货币"]
    _groups_sel = [groups[names.index(n)] for n in sel_names] + ["C"]

    # MA200 在全量历史预计算 (防窗口起点冷启动)
    ma200_hs = pd.Series(hs300_p).rolling(cfg.regime_ma).mean().values

    # 决策区切片
    _p_user = _price_sel[:, _pre_days:]
    ma200_user = ma200_hs[_pre_days:]

    actions, final_nav, nav_log = compute_yanyan(
        cfg, _names_sel, _groups_sel, _p_user, hs300_p[_pre_days:],
        n, ma200_hs=ma200_user)

    _vs_user = n - len(nav_log)
    user_nav = [1.0] * _vs_user + list(nav_log)
    if len(user_nav) == 0:
        user_nav = [1.0]

    # ── 决策记录 ──
    def _safe_price(val):
        if val is None or np.isnan(val):
            return None
        return round(float(val), 3)

    decisions = []
    _entry_nav = 1.0
    _entry_full_day = _pre_days
    for day_idx, old_p, new_p, reason in actions:
        full_day = min(_pre_days + day_idx, n_full - 1)
        user_day = min(day_idx, len(user_nav) - 1)
        nav_now = user_nav[user_day] if 0 <= user_day < len(user_nav) else 1.0
        if old_p >= 0 and new_p >= 0:
            pnl = (nav_now / _entry_nav - 1) * 100 if _entry_nav > 0 else 0
            decisions.append({"date": str(_all_full[full_day])[:10],
                              "action": f"{_names_sel[old_p]}→{_names_sel[new_p]}",
                              "price": _safe_price(_price_sel[old_p, full_day]),
                              "pnl": round(pnl, 2), "nav": round(nav_now, 4),
                              "days": full_day - _entry_full_day, "reason": reason})
            _entry_nav = nav_now
            _entry_full_day = full_day
        elif old_p >= 0:
            pnl = (nav_now / _entry_nav - 1) * 100 if _entry_nav > 0 else 0
            decisions.append({"date": str(_all_full[full_day])[:10],
                              "action": f"卖出{_names_sel[old_p]}",
                              "price": _safe_price(_price_sel[old_p, full_day]),
                              "pnl": round(pnl, 2), "nav": round(nav_now, 4),
                              "days": full_day - _entry_full_day, "reason": reason})
        elif new_p >= 0:
            _entry_nav = nav_now
            _entry_full_day = full_day
            decisions.append({"date": str(_all_full[full_day])[:10],
                              "action": f"买入{_names_sel[new_p]}",
                              "price": _safe_price(_price_sel[new_p, full_day]),
                              "pnl": None, "nav": round(nav_now, 4),
                              "days": None, "reason": reason})
    _final_pos = -1
    for _, _, np_val, _ in actions:
        _final_pos = np_val if np_val is not None else _final_pos
    if _final_pos >= 0:
        fd = n_full - 1
        pnl = (user_nav[-1] / _entry_nav - 1) * 100 if _entry_nav > 0 else 0
        decisions.append({"date": str(_all_full[fd])[:10],
                          "action": f"持有{_names_sel[_final_pos]}",
                          "price": _safe_price(_price_sel[_final_pos, fd]),
                          "pnl": round(pnl, 2), "nav": round(user_nav[-1], 4),
                          "days": fd - _entry_full_day, "reason": "期末"})

    # ── 持仓热力图序列 ──
    position_labels = ["空仓"] + sel_names
    position_series = [-1] * n
    current_pos = -1
    act_idx = 0
    actions_sorted = sorted(actions, key=lambda x: x[0])
    for t in range(n):
        while act_idx < len(actions_sorted) and actions_sorted[act_idx][0] <= t:
            a = actions_sorted[act_idx]
            current_pos = a[2] if a[2] is not None else current_pos
            act_idx += 1
        position_series[t] = current_pos

    # ── 等权买入持有基准 (10 只排名池, 不含货币) ──
    rank_idx = list(range(len(sel_names)))
    benchmark = [1.0] * n
    for t in range(1, n):
        prev = _p_user[rank_idx, t - 1]
        curr = _p_user[rank_idx, t]
        valid = ~np.isnan(prev) & ~np.isnan(curr) & (prev > 0)
        if valid.any():
            day_ret = np.mean(curr[valid] / prev[valid] - 1)
            benchmark[t] = benchmark[t - 1] * (1 + day_ret)
        else:
            benchmark[t] = benchmark[t - 1]

    # ── 各标的归一化净值 ──
    etf_navs = {}
    for idx, nm in enumerate(sel_names):
        s = _p_user[idx]
        fv = np.where(~np.isnan(s) & (s > 0))[0]
        if len(fv) > 0:
            base = s[fv[0]]
            etf_navs[nm] = [round(float(v / base), 4) if not np.isnan(v) and v > 0 else None for v in s]
        else:
            etf_navs[nm] = [None] * n

    # ── 回撤/月度/年度/风险指标 ──
    nav_arr = np.array(user_nav)
    cm = np.maximum.accumulate(nav_arr)
    drawdown = list(((nav_arr - cm) / cm * 100).round(2))

    monthly = {}
    month_nav = {}
    for i, d in enumerate(all_dates):
        key = f"{d.year}-{d.month:02d}"
        if key not in month_nav:
            month_nav[key] = [nav_arr[i], nav_arr[i]]
        else:
            month_nav[key][1] = nav_arr[i]
    for key in sorted(month_nav.keys()):
        s0, s1 = month_nav[key]
        monthly[key] = round((s1 / s0 - 1) * 100, 2) if s0 > 0 else 0.0

    daily_rets = []
    for t in range(1, n):
        if nav_arr[t - 1] > 0 and not np.isnan(nav_arr[t]):
            daily_rets.append(nav_arr[t] / nav_arr[t - 1] - 1)
    if len(daily_rets) > 10:
        dr = np.array(daily_rets)
        sharpe = round(float(np.mean(dr) / np.std(dr) * np.sqrt(252)), 2) if np.std(dr) > 1e-9 else 0.0
        volatility = round(float(np.std(dr) * np.sqrt(252) * 100), 2)
    else:
        sharpe = 0.0
        volatility = 0.0

    total_ret = round((user_nav[-1] - 1) * 100, 2)
    max_dd = round(float(np.min((nav_arr - cm) / cm * 100)), 2) if len(nav_arr) > 1 else 0
    years = max(n / 252, 0.01)
    ann_ret = (user_nav[-1] ** (1 / years) - 1) * 100 if user_nav[-1] > 0 else 0
    calmar = round(ann_ret / abs(max_dd), 2) if max_dd != 0 else 0

    yrs_map = {}
    for i, d in enumerate(all_dates):
        yrs_map[d.year] = float(nav_arr[i])
    prev = 1.0
    yearly = {}
    for yr in sorted(yrs_map):
        yearly[str(yr)] = round((yrs_map[yr] / prev - 1) * 100, 2)
        prev = yrs_map[yr]

    last_pos = "空仓"
    for d in reversed(decisions):
        if "买入" in d["action"] or "持有" in d["action"]:
            last_pos = d["action"][2:]
            break
        if "卖出" in d["action"] or "→" in d["action"]:
            break

    checks = []
    if abs(user_nav[0] - 1.0) > 0.01:
        checks.append(f"NAV起点异常: {user_nav[0]:.4f}")
    if not checks:
        checks = ["OK"]

    return clean_nan({
        "strategy": "yanyan",
        "total_ret": total_ret, "max_dd": max_dd, "ann_ret": round(ann_ret, 2),
        "calmar": calmar, "sharpe": sharpe, "volatility": volatility,
        "yearly": yearly, "monthly": monthly,
        "neg_years": sum(1 for v in yearly.values() if v < 0),
        "decisions": decisions, "nav_log": user_nav,
        "last_position": last_pos, "trade_count": len(decisions),
        "actual_universe": sel_names,
        "data_range": f"{str(all_dates[0])[:10]} ~ {str(all_dates[-1])[:10]}",
        "data_days": n_full, "server_time": _dt.datetime.now().strftime("%H:%M:%S"),
        "checks": checks,
        "dates": [str(d)[:10] for d in all_dates],
        "nav": [round(float(v), 4) for v in user_nav],
        "benchmark": [round(float(v), 4) for v in benchmark],
        "drawdown": drawdown,
        "etf_navs": etf_navs,
        "position_series": position_series,
        "position_labels": position_labels,
    })
