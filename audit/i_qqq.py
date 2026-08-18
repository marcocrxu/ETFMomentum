"""I 组 · 纳指动量信号源审计: QQQ(美元价×USDCNY) vs 513100 ETF — 2026-08-17

问题: 513100 是 QDII ETF, 价格含溢价噪声 (C6/D2 已确认); 改用 QQQ 作为纳指动量信号,
      执行仍在 513100。排名/阈值/上证比较用 QQQ-CNY 动量, 净值/止损仍按 513100 价格。
数据: Yahoo Finance chart API (QQQ + USDCNY=X), 缓存 audit/out/qqq_usdcny.csv。
对齐: CN 交易日 D 的收盘决策, 只用美国日期 ≤ D-1 的已收盘段 (无前视)。
输出: audit/out/i_qqq.json
"""
import sys, os, json, time
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd, requests

from harness import load_mats, run_cfg, base_cfg, metrics, save, trades_from_actions

CACHE = "/root/etf_strategy/audit/out/qqq_usdcny.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_yahoo(sym, period1="2013-06-01"):
    p1 = int(pd.Timestamp(period1).timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%s" % sym,
                     params={"period1": p1, "period2": p2, "interval": "1d"},
                     headers=UA, timeout=30)
    d = r.json()["chart"]["result"][0]
    ts = d["timestamp"]
    close = d["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s", utc=True).date,
                       sym: close}).dropna()
    return df


def load_qqq_cny():
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["date"])
    else:
        q = fetch_yahoo("QQQ")
        fx = fetch_yahoo("USDCNY=X")
        df = q.merge(fx, on="date", how="inner").sort_values("date")
        df.to_csv(CACHE, index=False)
        time.sleep(1)
    df["qqq_cny"] = df["QQQ"] * df["USDCNY=X"]
    return df


def align_us_to_cn(cn_dates, us_df):
    """CN 日期 D → 美国日期 ≤ D-1 的最后收盘 (无前视)"""
    us = us_df.set_index("date")["qqq_cny"]
    us_idx = pd.DatetimeIndex(us.index)
    out = np.full(len(cn_dates), np.nan)
    for i, d in enumerate(cn_dates):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        pos = us_idx.searchsorted(dt, side="right") - 1
        if pos >= 0:
            out[i] = us.iloc[pos]
    return out


def mom(arr, w):
    m = np.full(len(arr), np.nan)
    m[w:] = (arr[w:] / arr[:-w] - 1) * 100
    return m


def main():
    usdf = load_qqq_cny()
    print("QQQ-CNY 数据: %s ~ %s, %d 行" % (usdf["date"].min(), usdf["date"].max(), len(usdf)))
    out = {"qqq_range": [str(usdf["date"].min()), str(usdf["date"].max())]}

    for wname, s, e in (("默认窗", "2016-07-07", "2026-08-12"),
                        ("长窗", "2015-01-05", "2026-08-12")):
        mats = load_mats("纳指,黄金,创业".split(","), start=s, end=e)
        names = mats["etf_names"]; nz_i = names.index("纳指")
        cn_dates = mats["all_full"]
        qq = align_us_to_cn(cn_dates, usdf)

        # ── 信号相关性 ──
        etf_mom = mom(mats["price_mat"][nz_i], 20)
        qq_mom = mom(qq, 20)
        valid = ~np.isnan(etf_mom) & ~np.isnan(qq_mom)
        corr = float(np.corrcoef(etf_mom[valid], qq_mom[valid])[0, 1])
        diff = np.nanmean(np.abs(etf_mom[valid] - qq_mom[valid]))

        # ── 基线 ──
        cfg0 = base_cfg()
        a0, _, u0 = run_cfg(cfg0, mats)
        m0 = metrics(u0, len(u0))

        # ── 混合: 纳指动量换 QQQ-CNY (mom20/mom10 全换; 价格仍 513100) ──
        pm_h = mats["price_mat"].copy()
        m20_h = mats["mom20_mat"].copy()
        m10_h = mats["mom10_mat"].copy()
        m20_h[nz_i] = qq_mom
        m10_h[nz_i] = mom(qq, 10)
        pre = mats["pre_days"]
        from strategy.signal import calc_accel_thresholds, compute_signals
        pre_mats = {"accel_th": calc_accel_thresholds(m10_h, cfg0)[:, pre:]}
        a1, nav1, nav_log = compute_signals(
            cfg0, names, pm_h[:, pre:], m20_h[:, pre:], m10_h[:, pre:],
            mats["sh_mom"][pre:], mats["cyb_p"], mats["sh_price"],
            mats["n_full"] - pre, pre_days=pre, pre_mats=pre_mats)
        vs_user = (mats["n_full"] - pre) - len(nav_log)
        u1 = [1.0] * vs_user + list(nav_log)
        m1 = metrics(u1, len(u1))

        # ── top1 翻转统计 (与 C6 同口径) ──
        vc = np.sum(~np.isnan(m20_h), axis=0)
        flips = 0
        for t in range(1, mats["n_full"]):
            a_t, b_t = m20_h[:, t].copy(), mats["mom20_mat"][:, t].copy()
            a_t[np.isnan(a_t)] = -np.inf; b_t[np.isnan(b_t)] = -np.inf
            if int(np.argmax(a_t)) != int(np.argmax(b_t)):
                flips += 1

        out[wname] = {
            "corr_mom20": round(corr, 4), "abs_diff_pp": round(diff, 3),
            "top1_flip_days": flips,
            "baseline": m0, "hybrid_qqq": m1,
            "delta": {k: round(m1[k] - m0[k], 2) for k in ("total_ret", "mdd", "calmar")},
        }
        print("%s: corr=%.4f |平均动量差=%.2fpp |top1翻转=%d天" % (wname, corr, diff, flips))
        print("  基线    :", m0)
        print("  QQQ混合 :", m1, "  delta:", out[wname]["delta"])

    save(out, "i_qqq")


if __name__ == "__main__":
    main()
