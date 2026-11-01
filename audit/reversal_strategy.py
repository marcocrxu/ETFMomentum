"""反转策略参数寻优 + 有效性检验
窗口 2016-01-01 ~ 2025-12-31, 单边成本 0.05%, 数据同 param_search

网格:
  回看 L     1 / 2 / 3 / 5 日
  持有 k     3 / 5 / 10 / 20 日
  持仓数     2 / 3 / 4
  过滤       none / mom250>0(只在涨势里抄底) / mom250>-20%(不接深跌) / 排除高波动
  权重       等权 / 逆波动率
共 4×4×3×4×2 = 384 组
输出: 净夏普排序 + 多空价差 + 随机化检验 + 成本敏感性
"""
import importlib.util
import itertools
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "audit/param_search.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)

COST = 0.0005
S0, S1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31")

panel = ps.load_panel()
P, names, n_a, pm, cash_i, fac, vol300 = ps.build(panel)
dates = panel.index
mom250 = fac["mom250"]
VOL = (panel.pct_change().rolling(60).std() * np.sqrt(252)).values


def bt(L, k, n_pos, filt="none", wt="eq", cost=COST, pick="rev", rng=None,
       start=S0, end=S1, ret_period=False):
    reb = [i for i in range(250, len(P) - k, k)]
    reb = [i for i in reb if start <= dates[i] <= end]
    w = np.zeros(n_a); nav = 1.0; rets = []; turns = []
    for i in reb:
        F = P[i] / P[i - L] - 1
        M = mom250[i]
        V = VOL[i]
        ok = pm & ~np.isnan(F) & ~np.isnan(P[i])
        if filt == "trend":
            ok &= (M > 0)
        elif filt == "notdeep":
            ok &= (M > -0.20)
        elif filt == "lowvol":
            ok &= ~np.isnan(V)
        cand = np.where(ok)[0]
        if filt == "lowvol" and cand.size > n_pos:
            cand = cand[np.argsort(V[cand])][:max(n_pos, cand.size // 2)]
        w_new = np.zeros(n_a)
        if cand.size:
            if pick == "rev":
                sel = cand[np.argsort(F[cand])][:n_pos]
            elif pick == "mom":
                sel = cand[np.argsort(-F[cand])][:n_pos]
            else:  # random
                sel = rng.choice(cand, size=min(n_pos, cand.size), replace=False)
            if wt == "iv":
                v = np.array([max(V[j], 0.05) if not np.isnan(V[j]) else 0.20 for j in sel])
                iw = 1 / v; iw /= iw.sum()
                for t, j in enumerate(sel):
                    w_new[j] = iw[t]
            else:
                w_new[sel] = 1.0 / len(sel)
        else:
            w_new[cash_i] = 1.0
        turn = float(np.abs(w_new - w).sum())
        turns.append(turn)
        p0, p1 = P[i], P[i + k]
        r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, 0.0)
        g = float((w_new * r).sum())
        rets.append(g - cost * turn)
        nav *= (1 - cost * turn) * (1 + g)
        w = w_new
    if ret_period:
        return np.array(rets), np.array(turns)
    r = np.array(rets); yrs = len(r) / (252 / k)
    net = (np.prod(1 + r)) ** (1 / yrs) - 1
    vola = r.std(ddof=1) * np.sqrt(252 / k)
    peak = np.cumprod(1 + r); dd = (peak / np.maximum.accumulate(peak) - 1).min()
    return dict(cagr=net, vol=vola, sharpe=net / vola if vola else 0, mdd=dd,
                turn=np.mean(turns) * 252 / k, n=len(r))


def main():
    rows = []
    for L, k, npos, filt, wt in itertools.product(
            [1, 2, 3, 5], [3, 5, 10, 20], [2, 3, 4],
            ["none", "trend", "notdeep", "lowvol"], ["eq", "iv"]):
        s = bt(L, k, npos, filt, wt)
        rows.append(dict(L=L, k=k, npos=npos, filt=filt, wt=wt, **s))
    res = pd.DataFrame(rows)
    res.to_csv("audit/out/reversal_grid.csv", index=False)
    print("=" * 108)
    print("反转网格 %d 组 (2016-2025, 成本0.05%%)  按净夏普 Top12" % len(res))
    print("=" * 108)
    print("  %3s %3s %4s %-8s %4s | %8s %7s %7s %8s %8s" %
          ("L", "k", "持仓", "过滤", "权重", "净年化", "波动", "夏普", "最大回撤", "年换手"))
    for _, r in res.sort_values("sharpe", ascending=False).head(12).iterrows():
        print("  %3d %3d %4d %-8s %4s | %+7.2f%% %6.1f%% %7.2f %7.1f%% %7.0f%%" %
              (r.L, r.k, r.npos, r.filt, r.wt, r.cagr * 100, r.vol * 100, r.sharpe, r.mdd * 100, r.turn * 100))
    print("\n  对照 等权全池: 年化 +10.74% 夏普 0.70 最大回撤 -28.4%")
    print("  对照 12月动量top3: 年化 +23.82% 夏普 1.60 最大回撤 -16.0%")

    best = res.sort_values("sharpe", ascending=False).iloc[0]
    print("\n" + "=" * 108)
    print("最优反转配置: L=%d日, 持有%d日, %d只, 过滤=%s, 权重=%s" %
          (best.L, best.k, best.npos, best.filt, best.wt))
    print("=" * 108)
    # 多空价差 (纯因子, 无市场beta)
    lo = bt(best.L, best.k, best.npos, best.filt, best.wt, pick="rev", ret_period=True)[0]
    hi = bt(best.L, best.k, best.npos, best.filt, best.wt, pick="mom", ret_period=True)[0]
    n = min(len(lo), len(hi)); ls = lo - hi
    yrs = n / (252 / best.k)
    print("  多空价差(买跌最多 - 买涨最多): 年化 %+.2f%%  胜率 %.0f%%  夏普 %.2f" %
          (((np.prod(1 + ls)) ** (1 / yrs) - 1) * 100, (ls > 0).mean() * 100,
           ((np.prod(1 + ls)) ** (1 / yrs) - 1) / (ls.std(ddof=1) * np.sqrt(252 / best.k))))
    # 随机化
    rng = np.random.default_rng(0)
    sh = []
    for _ in range(400):
        sh.append(bt(best.L, best.k, best.npos, best.filt, best.wt, pick="rnd", rng=rng)["sharpe"])
    sh = np.array(sh)
    print("  随机选%d只(400次): 夏普 均值 %.2f (5%%~95%%: %.2f~%.2f) → 实际 %.2f 分位 %.0f%%" %
          (best.npos, sh.mean(), np.percentile(sh, 5), np.percentile(sh, 95), best.sharpe,
           (sh < best.sharpe).mean() * 100))
    # 成本敏感性
    print("  成本敏感性:")
    for c in [0.0, 0.0002, 0.0005, 0.001, 0.002]:
        s = bt(best.L, best.k, best.npos, best.filt, best.wt, cost=c)
        print("     单边%.2f%%: 净年化 %+7.2f%%  夏普 %5.2f  最大回撤 %6.1f%%" %
              (c * 100, s["cagr"] * 100, s["sharpe"], s["mdd"] * 100))
    # 分年度
    r, _ = bt(best.L, best.k, best.npos, best.filt, best.wt, ret_period=True)
    reb = [i for i in range(250, len(P) - best.k, best.k)]
    reb = [i for i in reb if S0 <= dates[i] <= S1]
    sr = pd.Series(r, index=[dates[i] for i in reb])
    yr = sr.groupby(sr.index.year).apply(lambda x: (1 + x).prod() - 1)
    print("  分年度: " + "  ".join("%d %+.1f%%" % (y, v * 100) for y, v in yr.items()))
    # A股池
    sub = np.array([n not in ["纳指", "黄金", "有色", "港股创新药"] for n in names])
    global pm
    old = pm.copy(); pm = pm & sub
    s2 = bt(best.L, best.k, best.npos, best.filt, best.wt)
    rng = np.random.default_rng(1)
    sh2 = np.array([bt(best.L, best.k, best.npos, best.filt, best.wt, pick="rnd", rng=rng)["sharpe"] for _ in range(400)])
    print("  仅A股池: 净年化 %+.2f%% 夏普 %.2f 最大回撤 %.1f%%  | 随机 夏普均值 %.2f (5%%~95%%: %.2f~%.2f) 分位 %.0f%%" %
          (s2["cagr"] * 100, s2["sharpe"], s2["mdd"] * 100, sh2.mean(),
           np.percentile(sh2, 5), np.percentile(sh2, 95), (sh2 < s2["sharpe"]).mean() * 100))
    pm = old


if __name__ == "__main__":
    main()
