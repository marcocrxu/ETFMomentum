"""回测引擎 — 供 web/app.py 和 stock-bot/app.py 共用"""
import numpy as np, pandas as pd, io, base64, datetime as _dt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import matplotlib.ticker as mticker

from strategy.signal import (SignalConfig, compute_signals,
                             calc_accel_thresholds, calc_mom60,
                             calc_vol20, calc_up_ratio20)
from data.feed import TushareFeed

import os
try:
    from web.config_local import TUSHARE_TOKEN
except ImportError:
    TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")

FEED = TushareFeed(TUSHARE_TOKEN,
                    cache_dir="/root/backtest/cache/etf_momentum")

ALL_CODES = {
    "纳指":"513100.SH","黄金":"518880.SH","创业":"159915.SZ",
    "科创50":"588000.SH","沪深300":"510300.SH","上证50":"510050.SH",
    "中证2000":"563300.SH","港股创新药":"513120.SH","有色":"512400.SH",
    "通信":"515880.SH","半导体":"512480.SH","红利低波":"512890.SH",
    "5年国债":"511010.SH",
    "30年国债":"511090.SH",
}

def clean_nan(obj):
    """把 float('nan') / numpy.nan 递归转成 None，保证 JSON 合法"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def run_backtest(params):
    """params: universe, start_date, end_date, sh_cash, method, mom_period,
       accel_on, accel_sigma, crash_on, crash_sigma, crash_lock,
       min_hold, min_gap, stop_loss, exception_gap, confirm_days, commission
       返回 dict with total_ret, max_dd, ann_ret, calmar, yearly, neg_years,
       decisions, chart(base64), nav_log, last_position, checks, data_days, server_time
    """
    etf_names = params.get("universe", "纳指,黄金,创业").split(",")
    etf_names = [n for n in etf_names if n in ALL_CODES]
    if len(etf_names) < 2:
        return {"error": "至少选择2个标的"}

    name_to_code = {n: ALL_CODES[n] for n in etf_names}
    codes = list(name_to_code.values()) + ["000001.SH", "399006.SZ"]
    _park_code = params.get("cash_park_code", "")
    if _park_code:
        codes += [_park_code]   # 空仓停泊标的 (不入排名池)

    bt_start = params.get("start_date", "2016-07-07")
    bt_end = params.get("end_date", "2026-07-28")
    mom_period = int(params.get("mom_period", 20))

    # Config
    sh_cash_raw = params.get("sh_cash", True)
    if isinstance(sh_cash_raw, str):
        sh_cash_exempt = (sh_cash_raw == "2")
        sh_cash = sh_cash_raw in ("1", "2") or sh_cash_raw.lower() == "true"
    else:
        sh_cash_exempt = bool(params.get("sh_cash_exempt", False))
        sh_cash = bool(sh_cash_raw)
    cfg = SignalConfig(
        sh_cash=sh_cash, sh_cash_exempt=sh_cash_exempt, method=params.get("method", "sigma"),
        accel_sigma=float(params.get("accel_sigma", 3.0)),
        accel_enabled=params.get("accel_on", True),
        crash_sigma=float(params.get("crash_sigma", -2.5)),
        crash_enabled=params.get("crash_on", True),
        crash_lock=int(params.get("crash_lock", 7)),
        min_hold=int(params.get("min_hold", 10)),
        min_gap=float(params.get("min_gap", 1.0)),
        stop_loss=float(params.get("stop_loss", -5.0)),
        stop_sigma=float(params.get("stop_sigma", 0.0)),
        stop_sigma_floor=bool(params.get("stop_sigma_floor", False)),
        entry_mom_cap=float(params.get("entry_mom_cap", 0.0)),
        entry_decel_on=bool(params.get("entry_decel_on", False)),
        entry_bias_cap=float(params.get("entry_bias_cap", 0.0)),
        vol_scale_target=float(params.get("vol_scale_target", 0.0)),
        trend_ma=int(params.get("trend_ma", 0)),
        trend_ma_exempt=bool(params.get("trend_ma_exempt", False)),
        rank_sticky=int(params.get("rank_sticky", 0)),
        cash_park_code=str(params.get("cash_park_code", "")),
        exception_gap=float(params.get("exception_gap", 3.0)),
        exception_gap_max=float(params.get("exception_gap_max", 0.0)),
        exception_gap_sigma=float(params.get("exception_gap_sigma", 0.0)),
        confirm_days=int(params.get("confirm_days", 2)),
        commission=float(params.get("commission", 0.0001)),
        slippage=float(params.get("slippage", 0.0)),
        abs_mom_enabled=params.get("abs_mom_on", False),
        defensive_enabled=params.get("defensive_on", False),
        risk_adj_mom=params.get("risk_adj_on", False),
        accel_cooldown=int(params.get("accel_cooldown", 0)),
        stop_cooldown=int(params.get("stop_cooldown", 0)),
        sh_cash_cooldown=int(params.get("sh_cash_cooldown", 3)),
        multi_mom=params.get("multi_mom_on", False),
        fip_tiebreak=params.get("fip_on", False),
        abs_stop_loss=params.get("abs_stop_on", True),
        crash_alloc=float(params.get("crash_alloc", 100)) / 100.0,
        top_k=int(params.get("top_k", 1)),
        mom_skip=int(params.get("mom_skip", 0)),
        fast_lane_staged=bool(params.get("fast_lane_staged", False)),
        fast_full_gap=float(params.get("fast_full_gap", 5.0)),
        fast_lane_ma=int(params.get("fast_lane_ma", 0)),
        vix_gate=float(params.get("vix_gate", 0.0)),
        vix_spike_exit=float(params.get("vix_spike_exit", 0.0)),
        skew_gate=float(params.get("skew_gate", 0.0)),
    )

    # 数据: 起始日往前拉预热天数
    #  - 动量/多周期/山脊集成: 各自周期 + 缓冲
    #  - MA200 (恐慌抄底/MA200判据) 需要 200 个交易日历史 ≈ 320 自然日
    #    提前拉够, 避免短回测区间内 MA200 为 NaN 导致恐慌抄底被静默禁用
    _mom_skip = int(params.get("mom_skip", 0))
    _warmup = mom_period + _mom_skip + 15
    if params.get("multi_mom_on", False):
        _warmup = max(_warmup, 75 + _mom_skip)
    if params.get("ridge_on", False):
        _warmup = max(_warmup, mom_period + _mom_skip + 17)
    _warmup = max(_warmup, 320)   # 保证 MA200 在回测起点就已就绪
    _bt_start_dt = _dt.datetime.strptime(bt_start.replace("-",""), "%Y%m%d")
    _fetch_start = (_bt_start_dt - _dt.timedelta(days=_warmup)).strftime("%Y%m%d")
    dfs = FEED.get_dataframe(codes, _fetch_start, bt_end.replace("-",""))
    if not dfs: return {"error": "数据加载失败"}

    # 截断到请求范围 (缓存可能返回更早的数据)
    _fetch_ts = pd.Timestamp(_fetch_start)
    _end_ts = pd.Timestamp(bt_end.replace("-",""))
    for code in list(dfs.keys()):
        df = dfs[code]
        df = df[(df["trade_date"] >= _fetch_ts) & (df["trade_date"] <= _end_ts)]
        if len(df) == 0: del dfs[code]
        else: dfs[code] = df.reset_index(drop=True)
    if not dfs: return {"error": "数据加载失败"}

    # 剔除无法加载的标的
    missing = [n for n, c in name_to_code.items() if c not in dfs]
    for m in missing:
        etf_names.remove(m); del name_to_code[m]
    if len(etf_names) < 2:
        return {"error": f"可用数据不足(仅{len(etf_names)}个标的), 缺失: {missing}"}

    # 构建全量时间轴
    _all_full = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
    _all_full = [d for d in _all_full if pd.Timestamp(_fetch_start) <= d <= pd.Timestamp(bt_end)]
    all_dates = [d for d in _all_full if pd.Timestamp(bt_start) <= d]
    n_full = len(_all_full); n = len(all_dates); _pre_days = n_full - n

    def _build_prices(code):
        s = dfs[code].set_index("trade_date")["close"]
        return np.array([s.get(d, np.nan) for d in _all_full])

    prices = {nm: _build_prices(code) for nm, code in name_to_code.items()}
    sh_price = _build_prices("000001.SH")
    cyb_p = _build_prices("399006.SZ") if "399006.SZ" in dfs else np.full(n_full, np.nan)
    bond_full = _build_prices(_park_code) if (_park_code and _park_code in dfs) else None
    price_mat = np.array([prices[nm] for nm in etf_names])

    def mom(arr, w, s=0):
        m = np.full(n_full, np.nan)
        if s <= 0:
            m[w:] = (arr[w:] / arr[:-w] - 1) * 100
        else:
            # 跳过最近 s 天: mom[t] = price[t-s]/price[t-s-w] - 1
            m[w+s:] = (arr[w:n_full-s] / arr[:n_full-w-s] - 1) * 100
        return m
    mom10_mat = np.array([mom(prices[nm], max(5, mom_period//2)) for nm in etf_names])

    # 动量口径: ratio=价格比(默认), ols=回归斜率(与生产 cron 同口径, 用于统一实盘/回测)
    _mom_type = params.get("mom_type", "ratio")
    if _mom_type == "ols":
        from strategy.signal import ols_momentum
        def _rank_mom(arr, w, s=0):
            base = ols_momentum(arr, w)
            if s > 0:
                out = np.full(n_full, np.nan); out[s:] = base[:n_full - s]
                return out
            return base
    else:
        _rank_mom = mom

    # 山脊集成: mom(mp-1..mp+2) 四周期等权平均作为集成动量,
    # 替换 mom20_mat / sh_mom (排名/阈值/上证空仓比较全部自动生效)
    if params.get("ridge_on", False):
        import warnings as _w
        _ws = [w for w in (mom_period - 1, mom_period, mom_period + 1, mom_period + 2) if w >= 5]
        def mom_ens(arr):
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(np.array([_rank_mom(arr, w, _mom_skip) for w in _ws]), axis=0)
        mom20_mat = np.array([mom_ens(prices[nm]) for nm in etf_names])
        sh_mom = mom_ens(sh_price)
    else:
        mom20_mat = np.array([_rank_mom(prices[nm], mom_period, _mom_skip) for nm in etf_names])
        sh_mom = _rank_mom(sh_price, mom_period)

    # 动量滞后窗口 (Novy-Marx 中间期回声): 排名动量改用"滞后lag天前的20日收益", 排名/阈值/上证比较全生效
    _mom_lag = int(params.get("mom_lag", 0))
    if _mom_lag > 0:
        def _lag_mom(arr, w, lag):
            m = np.full(n_full, np.nan)
            m[w+lag:] = (arr[w+lag:] / arr[:n_full-w-lag] - 1) * 100
            return m
        mom20_mat = np.array([_lag_mom(prices[nm], mom_period, _mom_lag) for nm in etf_names])
        sh_mom = _lag_mom(sh_price, mom_period, _mom_lag)

    # ── 纳指动量信号源混合: (1-w)×513100 ETF动量 + w×QQQ-CNY动量 (I组审计 2026-08-17) ──
    # 信号用混合动量, 净值/止损仍按 513100 成交价; on=False(默认) 与基线逐分一致
    _nz_on = bool(params.get("nz_mom_qqq_on", False))
    _nz_w = min(max(float(params.get("nz_mom_qqq_w", 0.7)), 0.0), 1.0)
    if _nz_on and _nz_w > 0 and "纳指" in etf_names:
        from data.usfeed import load_qqq_cny, align_to_cn
        _qq = align_to_cn(_all_full, load_qqq_cny())
        if params.get("ridge_on", False):
            _qm20 = mom_ens(_qq)
        elif _mom_lag > 0:
            _qm20 = _lag_mom(_qq, mom_period, _mom_lag)
        else:
            _qm20 = _rank_mom(_qq, mom_period, _mom_skip)
        _qm10 = mom(_qq, max(5, mom_period // 2))
        _nz_i = etf_names.index("纳指")
        mom20_mat[_nz_i] = (1 - _nz_w) * mom20_mat[_nz_i] + _nz_w * _qm20
        mom10_mat[_nz_i] = (1 - _nz_w) * mom10_mat[_nz_i] + _nz_w * _qm10

    # 从用户起始日重新跑模拟 (动量已预热)
    _p_user = price_mat[:, _pre_days:]
    # 派生矩阵在全量历史上预计算再切片: accel阈值/mom60/vol/fip 与恐慌抄底同口径,
    # 消除窗口起点冷启动空白 (此前在切片后数据上expanding, 窗口前60天加速止盈静默失效)
    pre_mats = {"accel_th": calc_accel_thresholds(mom10_mat, cfg)[:, _pre_days:]}
    if cfg.multi_mom:
        pre_mats["mom60"] = calc_mom60(price_mat)[:, _pre_days:]
    if cfg.risk_adj_mom or cfg.stop_sigma > 0 or cfg.vol_scale_target > 0:
        pre_mats["vol"] = calc_vol20(price_mat)[:, _pre_days:]
    if cfg.trend_ma > 0:
        _ma_full = np.full(price_mat.shape, np.nan)
        for _i in range(price_mat.shape[0]):
            _ma_full[_i] = pd.Series(price_mat[_i]).rolling(cfg.trend_ma).mean().values
        pre_mats["ma"] = _ma_full[:, _pre_days:]
    if cfg.fast_lane_ma > 0:
        # 快速通道追涨闸门: 全量历史预计算 MA(N) 再切片, 防窗口起点冷启动
        _maf_full = np.full(price_mat.shape, np.nan)
        for _i in range(price_mat.shape[0]):
            _maf_full[_i] = pd.Series(price_mat[_i]).rolling(cfg.fast_lane_ma).mean().values
        pre_mats["ma_fast"] = _maf_full[:, _pre_days:]
    if bond_full is not None:
        pre_mats["park"] = bond_full[_pre_days:]
    if cfg.vix_gate > 0 or cfg.vix_spike_exit > 0:
        from data.usfeed import align_fear_to_cn
        pre_mats["vix"] = align_fear_to_cn(_all_full, "vix")[_pre_days:]
    if cfg.skew_gate > 0:
        from data.usfeed import align_fear_to_cn
        pre_mats["skew"] = align_fear_to_cn(_all_full, "skew")[_pre_days:]
    if cfg.fip_tiebreak:
        pre_mats["up_ratio"] = calc_up_ratio20(price_mat)[:, _pre_days:]
    # cyb_p/sh_price 传完整历史序列(含预热), signal 内部用 pre_days 偏移访问,
    # 保证短回测区间下恐慌抄底的阈值/MA200 仍用足量历史计算
    actions, final_nav, nav_log = compute_signals(
        cfg, etf_names, _p_user,
        mom20_mat[:, _pre_days:], mom10_mat[:, _pre_days:],
        sh_mom[_pre_days:], cyb_p, sh_price,
        n_full - _pre_days,
        pre_days=_pre_days, pre_mats=pre_mats)

    _vs_user = (n_full - _pre_days) - len(nav_log)
    user_nav = [1.0] * _vs_user + list(nav_log)
    if len(user_nav) == 0: user_nav = [1.0]

    # 决策记录
    def _safe_price(val):
        if val is None or np.isnan(val): return None
        return round(float(val), 3)

    decisions = []
    _entry_nav = 1.0; _entry_full_day = _pre_days
    _crash_alloc_pct = int(round(float(params.get("crash_alloc", 100))))
    for day_idx, old_p, new_p, reason in actions:
        # day_idx 是决策区绝对索引 (signal 内 t 从 vs 起算), 直接映射, 不再叠加 _vs_user
        # (此前 full_day=_pre_days+_vs_user+day_idx 在 vs>0 时重复计入 vs, 如池内多标的晚上市)
        full_day = min(_pre_days + day_idx, n_full - 1)
        user_day = min(day_idx, len(user_nav) - 1)
        nav_now = user_nav[user_day] if 0 <= user_day < len(user_nav) else 1.0

        is_switch = (old_p >= 0 and new_p is not None and new_p >= 0)
        if is_switch:
            pnl = (nav_now / _entry_nav - 1) * 100 if _entry_nav > 0 else 0
            # 恐慌抄底(已持其他切换): 标注切过去的仓位比例
            if reason == "恐慌抄底":
                _tgt = f"{etf_names[new_p]}({_crash_alloc_pct}%)" if _crash_alloc_pct < 100 else etf_names[new_p]
                _act = f"{etf_names[old_p]}→{_tgt}"
            else:
                _act = f"{etf_names[old_p]}→{etf_names[new_p]}"
            decisions.append({"date": str(_all_full[full_day])[:10],
                "action": _act,
                "price": _safe_price(price_mat[old_p, full_day]),
                "pnl": round(pnl, 2), "nav": round(nav_now, 4),
                "days": full_day - _entry_full_day, "reason": reason})
            _entry_nav = nav_now; _entry_full_day = full_day
        elif old_p >= 0:
            pnl = (nav_now / _entry_nav - 1) * 100 if _entry_nav > 0 else 0
            decisions.append({"date": str(_all_full[full_day])[:10],
                "action": f"卖出{etf_names[old_p]}",
                "price": _safe_price(price_mat[old_p, full_day]),
                "pnl": round(pnl, 2), "nav": round(nav_now, 4),
                "days": full_day - _entry_full_day, "reason": reason})
        elif new_p is not None and new_p >= 0:
            _entry_nav = nav_now; _entry_full_day = full_day
            _act = f"买入{etf_names[new_p]}"
            if reason == "恐慌抄底":
                _act = f"买入{etf_names[new_p]}({_crash_alloc_pct}%)"
            decisions.append({"date": str(_all_full[full_day])[:10],
                "action": _act,
                "price": _safe_price(price_mat[new_p, full_day]),
                "pnl": None, "nav": round(nav_now, 4),
                "days": None, "reason": reason})

    # 期末
    _final_pos = -1
    for _, _, np_val, _ in actions:
        _final_pos = np_val if np_val is not None else _final_pos
    if _final_pos >= 0:
        fd = n_full - 1
        pnl = (user_nav[-1] / _entry_nav - 1) * 100 if _entry_nav > 0 else 0
        decisions.append({"date": str(_all_full[fd])[:10],
            "action": f"持有{etf_names[_final_pos]}",
            "price": _safe_price(price_mat[_final_pos, fd]),
            "pnl": round(pnl, 2), "nav": round(user_nav[-1], 4),
            "days": fd - _entry_full_day, "reason": "期末"})

    # ── 重建每日持仓序列 (用于前端热力图) ──
    # 部分仓位恐慌抄底(已持其他+crash_alloc<100)期间, 用组合标签表达:
    #   如 "黄金(50%)+创业(50%)", 动态并入 position_labels
    position_labels = ["空仓"] + etf_names
    position_series = [-1] * n
    current_pos = -1
    mix_idx = {}
    act_idx = 0
    actions_sorted = sorted(actions, key=lambda x: x[0])
    for t in range(n):
        while act_idx < len(actions_sorted) and actions_sorted[act_idx][0] <= t:
            a = actions_sorted[act_idx]
            old_p, new_p, reason = a[1], a[2], a[3]
            if (reason == "恐慌抄底" and old_p >= 0 and _crash_alloc_pct < 100):
                # 部分抄底: 组合持仓 {base}(1-alloc) + {创业}(alloc)
                key = (old_p, new_p)
                if key not in mix_idx:
                    mix_idx[key] = len(position_labels)
                    position_labels.append(
                        f"{etf_names[old_p]}({100 - _crash_alloc_pct}%)+"
                        f"{etf_names[new_p]}({_crash_alloc_pct}%)")
                current_pos = mix_idx[key]
            else:
                current_pos = new_p if new_p is not None else current_pos
            act_idx += 1
        position_series[t] = current_pos

    # ── 等权买入持有基准 (用用户窗口切片 _p_user, 此前误用全量矩阵前n列导致错位_pre_days天) ──
    benchmark = [1.0] * n
    for t in range(1, n):
        prev = _p_user[:, t - 1]
        curr = _p_user[:, t]
        valid = ~np.isnan(prev) & ~np.isnan(curr) & (prev > 0)
        if valid.any():
            day_ret = np.mean(curr[valid] / prev[valid] - 1)
            benchmark[t] = benchmark[t - 1] * (1 + day_ret)
        else:
            benchmark[t] = benchmark[t - 1]

    # ── 各 ETF 归一化净值 (同用 _p_user, 与 dates 逐日对齐) ──
    etf_navs = {}
    for idx, nm in enumerate(etf_names):
        s = _p_user[idx]
        first_valid = np.where(~np.isnan(s) & (s > 0))[0]
        if len(first_valid) > 0:
            base = s[first_valid[0]]
            etf_navs[nm] = [round(float(v / base), 4) if not np.isnan(v) and v > 0 else None for v in s]
        else:
            etf_navs[nm] = [None] * n

    # ── 回撤序列 ──
    nav_arr = np.array(user_nav)
    cm = np.maximum.accumulate(nav_arr)
    drawdown = list(((nav_arr - cm) / cm * 100).round(2))

    # ── 月度收益 ──
    monthly = {}
    month_nav = {}
    for i, d in enumerate(all_dates):
        key = f"{d.year}-{d.month:02d}"
        if key not in month_nav:
            month_nav[key] = [nav_arr[i] if i < len(nav_arr) else 1.0, nav_arr[i] if i < len(nav_arr) else 1.0]
        else:
            month_nav[key][1] = nav_arr[i] if i < len(nav_arr) else month_nav[key][1]
    prev_month_ret = 0.0
    for key in sorted(month_nav.keys()):
        start_nav, end_nav = month_nav[key]
        ret = (end_nav / start_nav - 1) * 100 if start_nav > 0 else 0.0
        monthly[key] = round(ret, 2)

    # ── 风险指标 (基于日收益率) ──
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

    # 统计
    total_ret = round((user_nav[-1] - 1) * 100, 2)
    nav_arr = np.array(user_nav)
    cm = np.maximum.accumulate(nav_arr)
    max_dd = round(float(np.min((nav_arr - cm) / cm * 100)), 2) if len(nav_arr) > 1 else 0
    years = max(n / 252, 0.01)
    ann_ret = (user_nav[-1] ** (1 / years) - 1) * 100 if user_nav[-1] > 0 else 0
    calmar = round(ann_ret / abs(max_dd), 2) if max_dd != 0 else 0

    yrs_map = {}
    for i, d in enumerate(all_dates):
        yrs_map[d.year] = float(nav_arr[i]) if i < len(nav_arr) else 1.0
    prev = 1.0; yearly = {}
    for yr in sorted(yrs_map):
        ret = (yrs_map[yr] / prev - 1) * 100
        yearly[str(yr)] = round(ret, 2); prev = yrs_map[yr]

    # 最后持仓
    last_pos = "空仓"
    for d in reversed(decisions):
        if "买入" in d["action"] or "持有" in d["action"]:
            last_pos = d["action"][2:]; break
        if "卖出" in d["action"] or "→" in d["action"]: break

    # Chart
    plt.figure(figsize=(14, 6))
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False; plt.style.use("dark_background")
    nav_plot = np.array(user_nav)
    if len(nav_plot) > 0 and len(all_dates) >= len(nav_plot):
        plt.plot(all_dates[-len(nav_plot):], nav_plot, color="#e94560", linewidth=1.2)
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    plt.title("ETF Momentum Rotation NAV", fontsize=14)
    plt.ylabel("NAV"); plt.grid(True, alpha=0.2)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(); buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode()

    # 自检
    checks = []
    if abs(user_nav[0] - 1.0) > 0.01:
        checks.append(f"NAV起点异常: {user_nav[0]:.4f}")
    _prod = 1.0
    for yr in sorted(yearly): _prod *= (1 + yearly[yr] / 100)
    if abs(_prod - (1 + total_ret / 100)) > 0.5:
        checks.append(f"年度乘积不匹配: {_prod:.2f} vs {1+total_ret/100:.2f}")
    _dr = (_dt.datetime.strptime(bt_end.replace("-",""), "%Y%m%d") -
           _dt.datetime.strptime(bt_start.replace("-",""), "%Y%m%d")).days
    if n > _dr * 0.7 + 60:   # 比较用户窗口天数 n (n_full 含预热段, 不参与此判定)
        checks.append(f"数据天数过多: {n}")
    for d in decisions:
        if d.get("days") is not None and d["days"] < 0:
            checks.append(f"负天数: {d['date']} {d['action']}")
    if not checks: checks = ["OK"]

    return clean_nan({
        "total_ret": total_ret, "max_dd": max_dd, "ann_ret": round(ann_ret, 2),
        "calmar": calmar, "sharpe": sharpe, "volatility": volatility,
        "yearly": yearly, "monthly": monthly, "neg_years": sum(1 for v in yearly.values() if v < 0),
        "decisions": decisions, "chart": chart_b64, "nav_log": user_nav,
        "last_position": last_pos, "trade_count": len(decisions),
        "actual_universe": etf_names,
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
