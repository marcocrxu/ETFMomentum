"""审计共用 harness: 复刻 bt_runner 数据准备, 支持自定义 sh_mom 窗口/动量归一化/任意参数"""
import sys, os, json
sys.path.insert(0, "/root/etf_strategy")
import numpy as np, pandas as pd, datetime as _dt

from strategy.signal import (SignalConfig, compute_signals,
                             calc_accel_thresholds, calc_mom60,
                             calc_vol20, calc_up_ratio20)
from data.feed import TushareFeed
from web.config_local import TUSHARE_TOKEN
from web.bt_runner import ALL_CODES, run_backtest

FEED = TushareFeed(TUSHARE_TOKEN, cache_dir="/root/backtest/cache/etf_momentum")

BASE_UNIVERSE = "纳指,黄金,创业"
BASE_START = "2016-07-07"
BASE_END = "2026-08-12"

def default_params(**over):
    p = {"universe": BASE_UNIVERSE, "start_date": BASE_START, "end_date": BASE_END}
    p.update(over)
    return p

def bt(**over):
    """快捷跑 run_backtest"""
    return run_backtest(default_params(**over))

def load_mats(etf_names, start=BASE_START, end=BASE_END, mom_period=20,
              sh_window=None, norm_vol=False):
    """复刻 bt_runner 数据准备。sh_window: 上证动量独立窗口(A5); norm_vol: mom/vol20 归一化(A6)
    返回 dict 含决策区切片后的全部矩阵 + 全量日期 + pre_days"""
    codes = [ALL_CODES[n] for n in etf_names] + ["000001.SH", "399006.SZ"]
    warmup = max(mom_period + 15, 320)
    bt_start_dt = _dt.datetime.strptime(start.replace("-", ""), "%Y%m%d")
    fetch_start = (bt_start_dt - _dt.timedelta(days=warmup)).strftime("%Y%m%d")
    dfs = FEED.get_dataframe(codes, fetch_start, end.replace("-", ""))
    fetch_ts = pd.Timestamp(fetch_start); end_ts = pd.Timestamp(end.replace("-", ""))
    for code in list(dfs.keys()):
        df = dfs[code]
        df = df[(df["trade_date"] >= fetch_ts) & (df["trade_date"] <= end_ts)]
        if len(df) == 0: del dfs[code]
        else: dfs[code] = df.reset_index(drop=True)
    all_full = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
    all_full = [d for d in all_full if fetch_ts <= d <= end_ts]
    all_dates = [d for d in all_full if pd.Timestamp(start) <= d]
    n_full = len(all_full); pre_days = n_full - len(all_dates)

    def bp(code):
        s = dfs[code].set_index("trade_date")["close"]
        return np.array([s.get(d, np.nan) for d in all_full])

    prices = {nm: bp(ALL_CODES[nm]) for nm in etf_names if ALL_CODES[nm] in dfs}
    etf_names = [n for n in etf_names if n in prices]
    sh_price = bp("000001.SH")
    cyb_p = bp("399006.SZ") if "399006.SZ" in dfs else np.full(n_full, np.nan)
    price_mat = np.array([prices[nm] for nm in etf_names])

    def mom(arr, w):
        m = np.full(n_full, np.nan)
        m[w:] = (arr[w:] / arr[:-w] - 1) * 100
        return m
    mom10_mat = np.array([mom(prices[nm], max(5, mom_period // 2)) for nm in etf_names])
    mom20_mat = np.array([mom(prices[nm], mom_period) for nm in etf_names])
    sh_mom = mom(sh_price, sh_window or mom_period)

    if norm_vol:  # A6: 排名/阈值动量按各标的 20 日波动率归一 (z-score 化)
        vol = calc_vol20(price_mat)
        with np.errstate(divide="ignore", invalid="ignore"):
            mom20_mat = np.where(vol > 0, mom20_mat / np.where(vol > 0, vol, 1.0), np.nan)
        sh_mom = sh_mom  # 上证保持原始口径(空仓判定仍跨资产比较, 口径需一致→不归一)

    return dict(etf_names=etf_names, price_mat=price_mat, mom20_mat=mom20_mat,
                mom10_mat=mom10_mat, sh_mom=sh_mom, cyb_p=cyb_p, sh_price=sh_price,
                n_full=n_full, pre_days=pre_days, all_full=all_full, all_dates=all_dates)

def run_cfg(cfg, mats):
    """在准备好的 mats 上跑 compute_signals, 返回 actions/nav/nav_log(用户窗口对齐)"""
    pre = mats["pre_days"]
    pre_mats = {"accel_th": calc_accel_thresholds(mats["mom10_mat"], cfg)[:, pre:]}
    actions, final_nav, nav_log = compute_signals(
        cfg, mats["etf_names"], mats["price_mat"][:, pre:],
        mats["mom20_mat"][:, pre:], mats["mom10_mat"][:, pre:],
        mats["sh_mom"][pre:], mats["cyb_p"], mats["sh_price"],
        mats["n_full"] - pre, pre_days=pre, pre_mats=pre_mats)
    vs_user = (mats["n_full"] - pre) - len(nav_log)
    user_nav = [1.0] * vs_user + list(nav_log)
    return actions, final_nav, user_nav

def metrics(user_nav, n_days):
    nav = np.array(user_nav)
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(n_days / 252, 0.01)
    ann = (nav[-1] ** (1 / years) - 1) * 100 if nav[-1] > 0 else -100.0
    dr = np.diff(nav) / nav[:-1]
    dr = dr[~np.isnan(dr)]
    sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if len(dr) > 10 and np.std(dr) > 1e-9 else 0.0
    return {"total_ret": round((nav[-1] - 1) * 100, 1), "ann": round(ann, 2),
            "mdd": round(mdd, 2), "calmar": round(ann / abs(mdd), 2) if mdd else 0,
            "sharpe": round(sharpe, 2)}

def base_cfg(**over):
    c = SignalConfig()
    for k, v in over.items(): setattr(c, k, v)
    return c

def trades_from_actions(actions, mats, user_nav):
    """把 actions 展开成逐笔交易账本 (含日期/标的/盈亏/持有天数/原因)"""
    pre = mats["pre_days"]; names = mats["etf_names"]
    trades = []; entry = None
    for day_idx, old_p, new_p, reason in actions:
        full_day = min(pre + day_idx, mats["n_full"] - 1)
        date = str(mats["all_full"][full_day])[:10]
        uday = min(day_idx, len(user_nav) - 1)
        nav_now = user_nav[uday]
        if old_p >= 0 and entry is not None:
            trades.append({**entry, "exit_date": date, "exit_nav": nav_now,
                           "ret": (nav_now / entry["entry_nav"] - 1) * 100,
                           "days": full_day - entry["entry_full_day"],
                           "exit_reason": reason, "switch_to": names[new_p] if (new_p is not None and new_p >= 0) else "空仓"})
        if new_p is not None and new_p >= 0:
            entry = {"asset": names[new_p], "entry_date": date, "entry_nav": nav_now,
                     "entry_full_day": full_day, "entry_reason": reason}
        elif new_p is None or new_p < 0:
            entry = None
    if entry is not None:
        trades.append({**entry, "exit_date": str(mats["all_full"][-1])[:10],
                       "exit_nav": user_nav[-1],
                       "ret": (user_nav[-1] / entry["entry_nav"] - 1) * 100,
                       "days": mats["n_full"] - 1 - entry["entry_full_day"],
                       "exit_reason": "期末", "switch_to": "-"})
    return trades

def save(obj, name):
    os.makedirs("/root/etf_strategy/audit/out", exist_ok=True)
    with open(f"/root/etf_strategy/audit/out/{name}.json", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    print(f"saved out/{name}.json")
