"""参数寻优: ETF 动量轮动 (窗口 2016-01-01 ~ 2025-12-31)
数据: /home/ecs-user/backtest/cache/etf_momentum/*.csv
口径: t 日收盘调仓, 持有 R 日, 单边成本 0.05%, 无候选停泊货基(缺失则现金 0%)

网格:
  因子   mom20 / mom60 / mom120 / mom250
  top_k  1..5
  调仓R  10 / 20 / 40 / 60
  过滤   abs(绝对动量) on/off
  动量上限 cap  None / 50% / 75% / 100% / 150%
  权重   等权 / 逆波动率
共 4×5×4×2×5×2 = 1600 组

稳健性: 2016-2021 (样本内) vs 2022-2025 (样本外) + 2026 (留出)
"""
import glob
import itertools
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
CASH = "货基"
COST = 0.0005


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


def build(panel):
    P = panel.values.astype(float)
    names = list(panel.columns)
    n_a = len(names)
    pool_mask = np.array([n != CASH for n in names])
    cash_i = names.index(CASH)
    fac = {}
    for w in [20, 60, 120, 250]:
        fac["mom%d" % w] = (panel / panel.shift(w) - 1).values
    vol = (panel.pct_change().rolling(60).std() * np.sqrt(252)).values
    return P, names, n_a, pool_mask, cash_i, fac, vol


def bt(P, fac_mat, vol, pool_mask, cash_i, top_k, R, abs_filter, cap, inv_vol,
       cost=COST, start=None, end=None, dates=None):
    n_days, n_a = P.shape
    reb = list(range(250, n_days - R, R))
    if start is not None:
        reb = [i for i in reb if dates[i] >= start]
    if end is not None:
        reb = [i for i in reb if dates[i] <= end]
    if len(reb) < 6:
        return None
    w = np.zeros(n_a)
    nav = 1.0
    log = np.empty(len(reb))
    turns = np.empty(len(reb))
    for k, i in enumerate(reb):
        S = fac_mat[i]
        cand = np.where(pool_mask & ~np.isnan(S) & ~np.isnan(P[i]))[0]
        if abs_filter:
            cand = cand[S[cand] > 0]
        if cap is not None:
            cand = cand[S[cand] <= cap]
        w_new = np.zeros(n_a)
        if cand.size:
            order = cand[np.argsort(-S[cand])][:top_k]
            if inv_vol:
                v = np.array([vol[i][j] for j in order])
                v = np.where(np.isnan(v) | (v < 0.05), 0.05, v)
                iw = 1.0 / v
                iw /= iw.sum()
                for t, j in enumerate(order):
                    w_new[j] = iw[t]
            else:
                w_new[order] = 1.0 / len(order)
        else:
            w_new[cash_i] = 1.0
        turn = float(np.abs(w_new - w).sum())
        turns[k] = turn
        nav *= (1 - cost * turn)
        p0, p1 = P[i], P[i + R]
        r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, 0.0)
        nav *= 1 + float((w_new * r).sum())
        log[k] = nav
        w = w_new
    navs = pd.Series(log, index=[dates[i] for i in reb])
    return navs, turns


def stats(navs, turns, R, ppy=None):
    r = navs.pct_change().dropna().values
    eq = navs.values
    yrs = (navs.index[-1] - navs.index[0]).days / 365.25
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1
    ppy = ppy or 252 / R
    vol = r.std(ddof=1) * np.sqrt(ppy)
    peak = np.maximum.accumulate(eq)
    mdd = (eq / peak - 1).min()
    return dict(cagr=cagr, vol=vol, sharpe=cagr / vol if vol else 0, mdd=mdd,
                calmar=cagr / abs(mdd) if mdd else 0, total=eq[-1] - 1,
                turn=np.mean(turns) * ppy, n=len(navs), yrs=yrs)


def main():
    panel = load_panel()
    P, names, n_a, pool_mask, cash_i, fac, vol = build(panel)
    dates = panel.index
    S0 = pd.Timestamp("2016-01-01"); S1 = pd.Timestamp("2025-12-31")

    grid = list(itertools.product(
        ["mom20", "mom60", "mom120", "mom250"],   # 因子
        [1, 2, 3, 4, 5],                          # top_k
        [10, 20, 40, 60],                         # 调仓周期
        [True, False],                            # 绝对动量过滤
        [None, 0.5, 0.75, 1.0, 1.5],              # 动量上限
        [False, True],                            # 逆波动率
    ))
    print("网格 %d 组, 窗口 %s ~ %s, 单边成本 %.2f%%"
          % (len(grid), S0.date(), S1.date(), COST * 100))
    rows = []
    for fname, top_k, R, absf, cap, iv in grid:
        out = bt(P, fac[fname], vol, pool_mask, cash_i, top_k, R, absf, cap, iv,
                 start=S0, end=S1, dates=dates)
        if out is None:
            continue
        s = stats(*out, R)
        rows.append(dict(factor=fname, top_k=top_k, R=R, absf=absf,
                         cap=(cap if cap else 0), inv_vol=iv, **s))
    res = pd.DataFrame(rows)
    res.to_csv("audit/out/param_grid_2016_2025.csv", index=False)
    print("结果已存 audit/out/param_grid_2016_2025.csv (%d 组)\n" % len(res))

    cols = ["factor", "top_k", "R", "absf", "cap", "inv_vol",
            "cagr", "vol", "sharpe", "mdd", "calmar", "turn"]
    for obj, asc in [("sharpe", False), ("calmar", False), ("cagr", False)]:
        print("=" * 118)
        print("按 %s 排序 Top10" % obj.upper())
        print("=" * 118)
        t = res.sort_values(obj, ascending=asc).head(10)
        print("  %-8s %4s %4s %5s %5s %4s | %8s %7s %7s %8s %7s %7s"
              % ("因子", "topk", "R", "absf", "cap", "iVol", "年化", "波动", "夏普", "最大回撤", "Calmar", "年换手"))
        for _, r in t.iterrows():
            print("  %-8s %4d %4d %5s %5.2f %4s | %+7.2f%% %6.1f%% %7.2f %7.1f%% %7.2f %6.0f%%"
                  % (r.factor, r.top_k, r.R, r.absf, r.cap, r.inv_vol,
                     r.cagr * 100, r.vol * 100, r.sharpe, r.mdd * 100, r.calmar, r.turn * 100))
        print()

    # 参数敏感性: 固定最佳因子, 看 (top_k × R) 与 cap 的夏普曲面
    best_f = res.sort_values("sharpe", ascending=False).iloc[0]
    print("=" * 118)
    print("参数高原: 因子=%s, 绝对动量=%s, 逆波动率=%s —— 夏普 (行=top_k, 列=R)"
          % (best_f.factor, best_f.absf, best_f.inv_vol))
    print("=" * 118)
    for cap in [0.5, 0.75, 1.0, 1.5, 0]:
        sub = res[(res.factor == best_f.factor) & (res.absf == best_f.absf)
                  & (res.inv_vol == best_f.inv_vol) & (res.cap == cap)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="top_k", columns="R", values="sharpe")
        print("  cap=%s" % ("无" if cap == 0 else "%.0f%%" % (cap * 100)))
        print("     R:      " + "".join("%9d" % c for c in piv.columns))
        for idx, row in piv.iterrows():
            print("     top%-2d   " % idx + "".join("%9.2f" % v for v in row.values))
        print()

    # 样本内/样本外
    print("=" * 118)
    print("样本内(2016-2021) vs 样本外(2022-2025): 夏普 Top10 配置")
    print("=" * 118)
    top10 = res.sort_values("sharpe", ascending=False).head(10)
    print("  %-8s %4s %4s %5s %5s %4s | %9s %9s | %9s" %
          ("因子", "topk", "R", "absf", "cap", "iVol", "IS夏普", "OOS夏普", "2026年化"))
    oos_rows = []
    for _, r in top10.iterrows():
        a = bt(P, fac[r.factor], vol, pool_mask, cash_i, r.top_k, int(r.R), bool(r.absf),
               (r.cap if r.cap else None), bool(r.inv_vol),
               start=pd.Timestamp("2016-01-01"), end=pd.Timestamp("2021-12-31"), dates=dates)
        b = bt(P, fac[r.factor], vol, pool_mask, cash_i, r.top_k, int(r.R), bool(r.absf),
               (r.cap if r.cap else None), bool(r.inv_vol),
               start=pd.Timestamp("2022-01-01"), end=pd.Timestamp("2025-12-31"), dates=dates)
        c = bt(P, fac[r.factor], vol, pool_mask, cash_i, r.top_k, int(r.R), bool(r.absf),
               (r.cap if r.cap else None), bool(r.inv_vol),
               start=pd.Timestamp("2026-01-01"), end=pd.Timestamp("2026-12-31"), dates=dates)
        sa = stats(*a, int(r.R)); sb = stats(*b, int(r.R)); sc = stats(*c, int(r.R))
        oos_rows.append((r, sa, sb, sc))
        print("  %-8s %4d %4d %5s %5.2f %4s | %9.2f %9.2f | %+8.1f%%"
              % (r.factor, r.top_k, r.R, r.absf, r.cap, r.inv_vol,
                 sa["sharpe"], sb["sharpe"], sc["cagr"] * 100))
    # IS/OOS 相关性 (全部网格)
    allc = []
    for _, r in res.iterrows():
        b = bt(P, fac[r.factor], vol, pool_mask, cash_i, r.top_k, int(r.R), bool(r.absf),
               (r.cap if r.cap else None), bool(r.inv_vol),
               start=pd.Timestamp("2022-01-01"), end=pd.Timestamp("2025-12-31"), dates=dates)
        if b:
            allc.append(stats(*b, int(r.R))["sharpe"])
        else:
            allc.append(np.nan)
    res["sharpe_oos"] = allc
    m = res.dropna(subset=["sharpe_oos"])
    print("\n全网格: IS 夏普 与 OOS 夏普 相关系数 = %.2f  (越低说明样本内排名越不可信)"
          % m["sharpe"].corr(m["sharpe_oos"]))
    print("OOS 夏普 > 0 的配置占比: %.0f%%  (共 %d 组)"
          % ((m["sharpe_oos"] > 0).mean() * 100, len(m)))
    # 稳健推荐: IS 与 OOS 都进前 20%
    q1 = m["sharpe"].quantile(0.8); q2 = m["sharpe_oos"].quantile(0.8)
    rob = m[(m["sharpe"] >= q1) & (m["sharpe_oos"] >= q2)].sort_values("sharpe", ascending=False)
    print("\n两段都进前 20%% 的稳健配置 (Top8):")
    print("  %-8s %4s %4s %5s %5s %4s | %8s %8s %8s %8s %8s"
          % ("因子", "topk", "R", "absf", "cap", "iVol", "IS夏普", "OOS夏普", "全窗夏普", "年化", "最大回撤"))
    for _, r in rob.head(8).iterrows():
        print("  %-8s %4d %4d %5s %5.2f %4s | %8.2f %8.2f %8.2f %+7.2f%% %7.1f%%"
              % (r.factor, r.top_k, r.R, r.absf, r.cap, r.inv_vol, r.sharpe,
                 r.sharpe_oos, r.sharpe, r.cagr * 100, r.mdd * 100))
    rob.to_csv("audit/out/param_robust_2016_2025.csv", index=False)


if __name__ == "__main__":
    os.makedirs("audit/out", exist_ok=True)
    main()
