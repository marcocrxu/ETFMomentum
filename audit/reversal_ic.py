"""反转(reversal)因子 IC 与组合检验 — 与 audit/ic_analysis.py 同源同口径
数据: /home/ecs-user/backtest/cache/etf_momentum/*.csv

1) IC 期限结构: 回看 L 日动量 → 未来 k 日收益 的横截面 Rank IC
2) 组合检验: 每 20 日调仓, 买输家(bottom3) / 买赢家(top3) / 多空, 计费前后
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
POOL = [v for v in NAME.values() if v != "货基"]


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
    ra = pd.Series(a).rank().values.astype(float)
    rb = pd.Series(b).rank().values.astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def nw_t(v, lag):
    v = np.asarray(v, float); n = len(v); x = v - v.mean()
    s = (x * x).sum() / n
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).sum() / n
    se = np.sqrt(max(s, 0) / n)
    return v.mean() / se if se > 0 else np.nan


def ic_term_structure(panel, lookbacks, horizons):
    print("=" * 96)
    print("IC 期限结构: 回看 L 日收益 → 未来 k 日收益  (横截面 Rank IC, 池 %d 只 ETF, %s~%s)"
          % (len(POOL), panel.index[0].date(), panel.index[-1].date()))
    print("=" * 96)
    hdr = "  回看L \\ 未来k " + "".join("%12s" % ("k=%d" % k) for k in horizons)
    print(hdr)
    out = {}
    M_ALL = {}
    for L in lookbacks:
        fac = panel / panel.shift(L) - 1
        M_ALL[L] = fac
        row = "%11s日 " % L
        for k in horizons:
            fwd = panel.shift(-k) / panel - 1
            M = fac[POOL].values; R = fwd[POOL].values
            ics = []
            for i in range(len(M) - k):
                ok = ~(np.isnan(M[i]) | np.isnan(R[i]))
                if ok.sum() >= 5:
                    ics.append(rank_ic(M[i][ok], R[i][ok]))
            v = np.array(ics)
            row += "%6.3f(%4.1f)" % (v.mean(), nw_t(v, k))
            out[(L, k)] = dict(ic=float(v.mean()), nw_t=float(nw_t(v, k)), win=float((v > 0).mean()))
        print(row + "   ← 括号内为 Newey-West t")
    return out, M_ALL


def portfolio_test(panel, L=10, k=20, n_side=3, cost=0.0005):
    """每 k 日调仓: 买回看 L 日跌最多的 n_side 只(反转) / 涨最多的 n_side 只(动量)"""
    P = panel[POOL].values
    cols = POOL
    dates = panel.index
    reb = list(range(L, len(P) - k, k))
    loser, winner, mkt = [], [], []
    turn_l = turn_w = 0.0
    prev_l, prev_w = set(), set()
    for i in reb:
        f = P[i] / P[i - L] - 1
        r = P[i + k] / P[i] - 1
        ok = ~(np.isnan(f) | np.isnan(r))
        idx = np.where(ok)[0]
        if len(idx) < 2 * n_side + 1:
            continue
        order = idx[np.argsort(f[idx])]
        lo, hi = list(order[:n_side]), list(order[-n_side:])
        rl, rw = r[lo].mean(), r[hi].mean()
        rm = r[idx].mean()
        tl = 1 - len(set(lo) & prev_l) / n_side
        tw = 1 - len(set(hi) & prev_w) / n_side
        loser.append(rl - cost * 2 * tl)
        winner.append(rw - cost * 2 * tw)
        mkt.append(rm)
        turn_l += tl; turn_w += tw
        prev_l, prev_w = set(lo), set(hi)
    def summ(x, lab, periods_per_year=252 / k):
        x = np.array(x)
        eq = np.cumprod(1 + x)
        yrs = len(x) / periods_per_year
        cagr = eq[-1] ** (1 / yrs) - 1
        vol = x.std(ddof=1) * np.sqrt(periods_per_year)
        peak = np.maximum.accumulate(eq)
        mdd = (eq / peak - 1).min()
        print("  %-28s 年化 %+6.2f%%  波动 %5.1f%%  夏普 %5.2f  最大回撤 %6.1f%%  胜率 %4.0f%%  调仓%d次"
              % (lab, cagr * 100, vol * 100, cagr / vol if vol else 0, mdd * 100,
                 (x > 0).mean() * 100, len(x)))
        return cagr
    print("\n" + "=" * 96)
    print("组合检验: 回看 %d 日, 每 %d 日调仓, 各取 %d 只 (单边成本 %.2f%%)"
          % (L, k, n_side, cost * 100))
    print("=" * 96)
    summ(mkt, "等权全池(基准, 无成本)")
    summ(winner, "买赢家 top%d (动量)" % n_side)
    summ(loser, "买输家 bottom%d (反转)" % n_side)
    n = min(len(loser), len(winner))
    ls = [loser[i] - winner[i] for i in range(n)]
    summ(ls, "多空: 输家-赢家 (含成本)")
    print("  平均单边换手: 输家腿 %.0f%%  赢家腿 %.0f%% (反转腿换手 %s)"
          % (turn_l / len(loser) * 100, turn_w / len(winner) * 100,
             "更低" if turn_l < turn_w else "更高"))


def main():
    panel = load_panel()
    print("面板 %d 标的 × %d 日 (%s ~ %s)\n"
          % (panel.shape[1], panel.shape[0], panel.index[0].date(), panel.index[-1].date()))
    ic_term_structure(panel,
                      lookbacks=[1, 2, 3, 5, 10, 20, 40, 60, 120, 250],
                      horizons=[1, 5, 20])
    for L in [5, 10, 20]:
        portfolio_test(panel, L=L, k=20, n_side=3)


if __name__ == "__main__":
    main()
