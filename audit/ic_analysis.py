"""ETF 动量因子 IC 分析
数据源: /home/ecs-user/backtest/cache/etf_momentum/*.csv (与实盘/回测同源)
口径: 复权收盘 = close * adj_factor / 最新 adj_factor (与 data/feed.py 一致)

因子: mom_w[t] = P[t]/P[t-w] - 1
未来收益: fwd_k[t] = P[t+k]/P[t] - 1  (t 收盘买入, t+k 收盘卖出, 无前视)
IC: 每个交易日 t, 横截面上 Spearman(rank) 相关
"""
import glob
import os

import numpy as np
import pandas as pd

CACHE = "/home/ecs-user/backtest/cache/etf_momentum"

NAME = {
    "513100.SH": "纳指", "159915.SZ": "创业", "518880.SH": "黄金",
    "588080.SH": "科创50", "510300.SH": "沪深300", "510050.SH": "上证50",
    "563300.SH": "中证2000", "513120.SH": "港股创新药", "512400.SH": "有色",
    "512890.SH": "红利低波", "510500.SH": "中证500", "511880.SH": "货基",
}
LIVE_POOL = ["纳指", "创业", "黄金"]
XS_POOL = [v for k, v in NAME.items() if v != "货基"]


def load_panel():
    px = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "*.csv"))):
        code = os.path.basename(f).replace("_", ".").replace(".csv", "")
        if code not in NAME:
            continue
        df = pd.read_csv(f, parse_dates=["trade_date"]).sort_values("trade_date")
        adj = df["adj_factor"].ffill().bfill().fillna(1.0)
        df["close_adj"] = df["close"] * adj / adj.iloc[-1]
        px[NAME[code]] = df.set_index("trade_date")["close_adj"]
    return pd.DataFrame(px).sort_index()


def rank_ic(a, b):
    """一行(横截面)的 Spearman IC, a/b 为等长有效值"""
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().values.astype(float)
    rb = pd.Series(b).rank().values.astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def nw_t(v, lag):
    """Newey-West 调整 t 值 (重叠窗口的 IC 序列自相关极高, 朴素 t 严重虚高)"""
    v = np.asarray(v, float)
    n = len(v)
    x = v - v.mean()
    s = (x * x).sum() / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).sum() / n
    se = np.sqrt(max(s, 0) / n)
    return v.mean() / se if se > 0 else np.nan


def ic_stats(ics, label, nw_lag=0):
    v = np.array([x for x in ics if not np.isnan(x)])
    if len(v) == 0:
        print("  %-30s 无有效样本" % label)
        return None
    mean, sd, n = v.mean(), v.std(ddof=1), len(v)
    icir = mean / sd if sd > 0 else np.nan
    t = icir * np.sqrt(n)
    extra = ""
    if nw_lag:
        tnw = nw_t(v, nw_lag)
        extra = "  NW t %+5.2f" % tnw
    print("  %-30s n=%4d  IC %+.4f  std %.3f  ICIR %+.3f  t %+5.2f%s  IC>0 %4.1f%%"
          % (label, n, mean, sd, icir, t, extra, (v > 0).mean() * 100))
    return dict(n=int(n), ic=float(mean), sd=float(sd), icir=float(icir),
                t=float(t), win=float((v > 0).mean()))


def cross_ic(mom_df, fwd_df, cols, min_n=5, k=0):
    """逐日横截面 IC (numpy 预取, 避免 pandas 逐行开销)"""
    M = mom_df[cols].values
    R = fwd_df[cols].values
    n_days = M.shape[0]
    out = []
    for i in range(n_days - k):
        m = ~(np.isnan(M[i]) | np.isnan(R[i]))
        if m.sum() >= min_n:
            out.append(rank_ic(M[i][m], R[i][m]))
    return out


def main():
    panel = load_panel()
    print("价格面板: %d 标的 × %d 交易日 (%s ~ %s)"
          % (panel.shape[1], panel.shape[0], panel.index[0].date(), panel.index[-1].date()))
    print("标的:", ", ".join(panel.columns))
    P = panel.values
    dates = panel.index

    for w, fname in [(20, "mom20 (策略主因子)"), (10, "mom10"), (60, "mom60")]:
        mom = panel / panel.shift(w) - 1
        print("\n" + "=" * 80)
        print("因子 %s  →  未来 k 日收益 | 横截面 Rank IC (池 %d 只 ETF)" % (fname, len(XS_POOL)))
        print("=" * 80)
        for k in [1, 5, 10, 20]:
            fwd = panel.shift(-k) / panel - 1
            ics = cross_ic(mom, fwd, XS_POOL, k=k)
            ic_stats(ics, "k=%d 日" % k, nw_lag=k)
        print("  --- 实盘池(纳指/创业/黄金, N=3) ---")
        for k in [5, 20]:
            fwd = panel.shift(-k) / panel - 1
            ics = cross_ic(mom, fwd, LIVE_POOL, min_n=3, k=k)
            ic_stats(ics, "k=%d 日" % k, nw_lag=k)
        print("  --- 分年度 (k=20) ---")
        fwd = panel.shift(-20) / panel - 1
        M = mom[XS_POOL].values; R = fwd[XS_POOL].values
        by_year = {}
        for i in range(len(M) - 20):
            m = ~(np.isnan(M[i]) | np.isnan(R[i]))
            if m.sum() >= 5:
                by_year.setdefault(dates[i].year, []).append(rank_ic(M[i][m], R[i][m]))
        for y in sorted(by_year):
            ic_stats(by_year[y], "%d 年" % y)

    print("\n" + "=" * 80)
    print("逐标的: 时序 Rank IC (mom20 vs 自身未来20日收益) 与绝对动量条件收益")
    print("=" * 80)
    mom20 = (panel / panel.shift(20) - 1).values
    fwd20 = (panel.shift(-20) / panel - 1).values
    for j, nm in enumerate(panel.columns):
        a, b = mom20[:, j], fwd20[:, j]
        m = ~(np.isnan(a) | np.isnan(b)) & ~np.isnan(P[:, j])
        if m.sum() < 100:
            continue
        hold = b[m][a[m] > 0]; flat = b[m][a[m] <= 0]
        print("  %-8s n=%4d  时序IC %+.3f   mom20>0 未来20日均值 %+.2f%%   mom20<=0 均值 %+.2f%%"
              % (nm, m.sum(), rank_ic(a[m], b[m]),
                 np.mean(hold) * 100 if len(hold) else np.nan,
                 np.mean(flat) * 100 if len(flat) else np.nan))

    print("\n" + "=" * 80)
    print("实盘期核对 (2026-09-10 ~ %s)" % dates[-1].date())
    print("=" * 80)
    seg = panel.loc["2026-09-10":]
    print(seg[LIVE_POOL].round(4).to_string())
    m = (panel / panel.shift(20) - 1).loc["2026-09-10":][LIVE_POOL]
    print("\n20日动量 (%):")
    print((m * 100).round(2).to_string())


if __name__ == "__main__":
    main()
