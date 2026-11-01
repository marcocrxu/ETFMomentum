"""A股个股 反转策略回测 (前复权日线, baostock)
池: 沪深300 + 中证500 成分股 (今日成分, 有幸存者偏差)
信号: 过去 L 日收益, 买跌得最多的 N 只 (反转) / 买涨得最多的 (动量)
调仓: 每 k 个交易日, 等权, 单边成本 0.15% (佣金+印花税+滑点)
过滤: 剔除 ST / 停牌 / 上市不足 250 日 / 价格<2元
"""
import glob
import os

import numpy as np
import pandas as pd

CACHE = "/home/ecs-user/backtest/cache/stocks"
COST = 0.0015
S0, S1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31")


def load():
    px, st, ts = {}, {}, {}
    files = sorted(glob.glob(os.path.join(CACHE, "*.csv")))
    for f in files:
        code = os.path.basename(f)[:-4]
        try:
            df = pd.read_csv(f, usecols=["date", "close", "tradestatus", "isST"],
                             parse_dates=["date"])
        except Exception:
            continue
        if len(df) < 300:
            continue
        df = df.sort_values("date").drop_duplicates("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        px[code] = df.set_index("date")["close"]
        ts[code] = pd.to_numeric(df.set_index("date")["tradestatus"], errors="coerce")
        st[code] = pd.to_numeric(df.set_index("date")["isST"], errors="coerce")
    P = pd.DataFrame(px).sort_index()
    T = pd.DataFrame(ts).reindex(P.index)
    S = pd.DataFrame(st).reindex(P.index)
    return P, T, S


def run(P, T, S, L=20, k=20, n_pos=50, cost=COST, pick="rev", rng=None,
        start=S0, end=S1, use_filter=True, ret_detail=False):
    dates = P.index
    V = P.values
    Tv = T.values
    Sv = S.values
    n_stock = V.shape[1]
    reb = list(range(260, len(dates) - k, k))
    reb = [i for i in reb if start <= dates[i] <= end]
    hist_ok = np.zeros(n_stock)
    for j in range(n_stock):
        col = V[:, j]
        hist_ok[j] = np.sum(~np.isnan(col[:300]))
    w = np.zeros(n_stock)
    nav = 1.0
    rets, turns, hold = [], [], []
    for i in reb:
        F = V[i] / V[i - L] - 1
        ok = ~np.isnan(F) & ~np.isnan(V[i])
        if use_filter:
            ok &= (Tv[i] == 1)
            ok &= (Sv[i] == 0)
            ok &= (V[i] >= 2.0)
            # 上市满 250 个交易日
            listed = np.array([np.sum(~np.isnan(V[max(0, i - 300):i + 1, j])) >= 250 for j in range(n_stock)])
            ok &= listed
        # 涨停/跌停不可成交 (主板 ±10%, 创业/科创 ±20%)
        pct = V[i] / V[i - 1] - 1
        lim = np.where(np.array([c.startswith(("sz_30", "sh_68")) for c in P.columns]), 0.195, 0.095)
        tradable = ~((pct > lim) | (pct < -lim))
        cand = np.where(ok)[0]
        w_new = np.zeros(n_stock)
        if cand.size > n_pos:
            if pick == "rev":
                sel = cand[np.argsort(F[cand])][:n_pos]
            elif pick == "mom":
                sel = cand[np.argsort(-F[cand])][:n_pos]
            else:
                sel = rng.choice(cand, size=n_pos, replace=False)
        else:
            sel = cand
        if sel.size:
            w_new[sel] = 1.0 / len(sel)
        turn = float(np.abs(w_new - w).sum())
        turns.append(turn)
        p0, p1 = V[i], V[i + k]
        r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, 0.0)
        g = float((w_new * r).sum())
        rets.append(g - cost * turn)
        nav *= (1 - cost * turn) * (1 + g)
        hold.append(int(sel.size))
        w = w_new
    r = np.array(rets)
    yrs = len(r) / (252 / k)
    cagr = (np.prod(1 + r)) ** (1 / yrs) - 1
    vola = r.std(ddof=1) * np.sqrt(252 / k)
    eq = np.cumprod(1 + r)
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    out = dict(cagr=cagr, vol=vola, sharpe=cagr / vola if vola else 0, mdd=mdd,
               turn=np.mean(turns) * 252 / k, n=len(r), avg_hold=np.mean(hold),
               win=(r > 0).mean())
    if ret_detail:
        return pd.Series(r, index=[dates[i] for i in reb]), out
    return out


def main():
    P, T, S = load()
    print("个股池: %d 只 × %d 交易日 (%s ~ %s)" %
          (P.shape[1], P.shape[0], P.index[0].date(), P.index[-1].date()))

    # 1) 反转 vs 动量 vs 随机 vs 等权
    print("\n" + "=" * 100)
    print("【1】20日反转 vs 动量 vs 随机 vs 全池等权 (月频调仓, 单边成本0.15%%, 2016-2025)")
    print("=" * 100)
    print("  %-34s %9s %8s %8s %9s %9s %7s" % ("策略", "年化", "波动", "夏普", "最大回撤", "年换手", "月胜率"))
    eq = run(P, T, S, L=20, k=20, n_pos=10 ** 6, cost=0.0, use_filter=True)
    print("  %-34s %+8.2f%% %7.1f%% %8.2f %8.1f%% %8.0f%% %6.1f%%" %
          ("全池等权(基准, 无成本)", eq["cagr"] * 100, eq["vol"] * 100, eq["sharpe"],
           eq["mdd"] * 100, eq["turn"] * 100, eq["win"] * 100))
    for n_pos in [30, 50, 80]:
        for pick, lab in [("rev", "买跌最多%d只(反转)"), ("mom", "买涨最多%d只(动量)")]:
            s = run(P, T, S, L=20, k=20, n_pos=n_pos, pick=pick)
            print("  %-34s %+8.2f%% %7.1f%% %8.2f %8.1f%% %8.0f%% %6.1f%%" %
                  (lab % n_pos, s["cagr"] * 100, s["vol"] * 100, s["sharpe"], s["mdd"] * 100,
                   s["turn"] * 100, s["win"] * 100))
    rng = np.random.default_rng(0)
    rs = [run(P, T, S, L=20, k=20, n_pos=50, pick="rnd", rng=rng) for _ in range(200)]
    sh = np.array([x["sharpe"] for x in rs]); cg = np.array([x["cagr"] for x in rs])
    print("  %-34s %+8.2f%% %7s %8.2f (5%%~95%%: %.2f~%.2f) %8s %6.1f%%" %
          ("随机选50只(200次均值)", cg.mean() * 100, "—", sh.mean(),
           np.percentile(sh, 5), np.percentile(sh, 95), "—", np.mean([x["win"] for x in rs]) * 100))
    base = run(P, T, S, L=20, k=20, n_pos=50, pick="rev")
    print("  → 反转夏普 %.2f 在随机分布中的分位: %.0f%%" % (base["sharpe"], (sh < base["sharpe"]).mean() * 100))

    # 2) 五分组
    print("\n" + "=" * 100)
    print("【2】五分组 (按过去20日收益分5组, 等权持有20日, 不含成本)")
    print("=" * 100)
    dates = P.index; V = P.values; Tv = T.values; Sv = S.values
    reb = [i for i in range(260, len(dates) - 20, 20) if S0 <= dates[i] <= S1]
    groups = {q: [] for q in range(1, 6)}
    for i in reb:
        F = V[i] / V[i - 20] - 1
        ok = (~np.isnan(F) & ~np.isnan(V[i]) & (Tv[i] == 1) & (Sv[i] == 0) & (V[i] >= 2))
        cand = np.where(ok)[0]
        if cand.size < 50:
            continue
        order = cand[np.argsort(F[cand])]
        p0, p1 = V[i], V[i + 20]
        for q in range(1, 6):
            part = order[int((q - 1) * len(order) / 5):int(q * len(order) / 5)]
            r = np.where((p0[part] > 0) & ~np.isnan(p1[part]), p1[part] / p0[part] - 1, 0.0)
            groups[q].append(r.mean())
    print("  %-10s %10s %8s" % ("分组", "年化", "月胜率"))
    for q in range(1, 6):
        g = np.array(groups[q]); yrs = len(g) / 12
        c = (np.prod(1 + g)) ** (1 / yrs) - 1
        tag = " ← 跌最多(反转买这个)" if q == 1 else (" ← 涨最多" if q == 5 else "")
        print("  Q%d%-8s %+9.2f%% %7.1f%%%s" % (q, "", c * 100, (g > 0).mean() * 100, tag))
    g1 = np.array(groups[1]); g5 = np.array(groups[5]); ls = g1 - g5
    print("  Q1-Q5 多空价差: 年化 %+.2f%%  胜率 %.1f%%" %
          (((np.prod(1 + ls)) ** (1 / (len(ls) / 12)) - 1) * 100, (ls > 0).mean() * 100))

    # 3) 不同回看期
    print("\n" + "=" * 100)
    print("【3】不同回看期 (买跌最多50只, 月频, 成本0.15%)")
    print("=" * 100)
    print("  %-10s %10s %8s %8s %9s" % ("回看L", "年化", "夏普", "最大回撤", "年换手"))
    for L in [5, 10, 20, 60, 120, 250]:
        s = run(P, T, S, L=L, k=20, n_pos=50, pick="rev")
        print("  %-10s %+9.2f%% %8.2f %8.1f%% %8.0f%%" % ("%d日" % L, s["cagr"] * 100, s["sharpe"], s["mdd"] * 100, s["turn"] * 100))

    # 4) 成本敏感性
    print("\n" + "=" * 100)
    print("【4】成本敏感性 (20日反转, 买跌最多50只)")
    print("=" * 100)
    for c in [0.0, 0.0005, 0.001, 0.0015, 0.003]:
        s = run(P, T, S, L=20, k=20, n_pos=50, cost=c)
        print("  单边 %.2f%%: 年化 %+8.2f%%  夏普 %5.2f  最大回撤 %7.1f%%" %
              (c * 100, s["cagr"] * 100, s["sharpe"], s["mdd"] * 100))

    # 5) 分年度
    r, _ = run(P, T, S, L=20, k=20, n_pos=50, pick="rev", ret_detail=True)
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    print("\n  分年度(反转): " + "  ".join("%d %+.1f%%" % (y, v * 100) for y, v in yr.items()))


if __name__ == "__main__":
    main()
