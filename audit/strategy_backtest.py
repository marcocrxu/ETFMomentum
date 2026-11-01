"""ETF 轮动策略回测 — 基于 IC 检验结论 (12月动量 + 短期反转过滤 + 绝对动量空仓)
数据: /home/ecs-user/backtest/cache/etf_momentum/*.csv  (2013-02 ~ 2026-09-30)
成交: t 日收盘调仓, t+R 日收盘再平衡, 单边成本默认 0.05% (佣金+滑点)
现金停泊: 货基 ETF 511880 (无候选时)

对照: 等权全池 / mom20 top-k (当前生产策略的简化代理) / 纯 mom250 / mom250+反转过滤
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
CASH = "货基"
POOL = [v for v in NAME.values() if v != CASH]


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


def run(panel, score="mom250", top_k=3, R=20, abs_filter=True, rev_lambda=0.5,
        cost=0.0005, start=None, end=None, ret_all=False):
    """返回 (stats_dict, nav_series)"""
    P = panel.values
    names = list(panel.columns)
    idx = {n: i for i, n in enumerate(names)}
    dates = panel.index
    cash_i = idx[CASH]
    mom250 = panel / panel.shift(250) - 1
    mom20 = panel / panel.shift(20) - 1
    mom5 = panel / panel.shift(5) - 1
    M250, M20, M5 = mom250.values, mom20.values, mom5.values

    t0 = 250
    reb = list(range(t0, len(P) - R, R))
    if start:
        reb = [i for i in reb if dates[i] >= pd.Timestamp(start)]
    if end:
        reb = [i for i in reb if dates[i] <= pd.Timestamp(end)]

    w = np.zeros(len(names))
    nav = 1.0
    nav_log, pos_log = [], []
    turnovers = []
    for i in reb:
        # ── 打分 (仅用 t 及之前数据) ──
        cand = [j for j in range(len(names)) if names[j] in POOL
                and not np.isnan(M250[i][j]) and not np.isnan(P[i][j])]
        if abs_filter:
            cand = [j for j in cand if M250[i][j] > 0]
        if score == "mom250":
            key = {j: M250[i][j] for j in cand}
        elif score == "mom20":
            k2 = [j for j in cand if not np.isnan(M20[i][j])]
            key = {j: M20[i][j] for j in k2}
        elif score == "mom250_rev":   # 12月动量排名 减 λ×5日涨幅排名
            k2 = [j for j in cand if not np.isnan(M5[i][j])]
            r250 = pd.Series({j: M250[i][j] for j in k2}).rank()
            r5 = pd.Series({j: M5[i][j] for j in k2}).rank()
            key = (r250 - rev_lambda * r5).to_dict()
        else:
            raise ValueError(score)
        sel = sorted(key, key=lambda j: -key[j])[:top_k] if key else []

        # ── 目标权重 ──
        w_new = np.zeros(len(names))
        if sel:
            for j in sel:
                w_new[j] = 1.0 / len(sel)
        else:
            w_new[cash_i] = 1.0   # 空仓 -> 停泊货基
        turn = float(np.abs(w_new - w).sum())
        turnovers.append(turn)
        nav *= (1 - cost * turn)

        # ── 持有到 i+R ──
        p0 = P[i]; p1 = P[i + R]
        r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, 0.0)
        nav *= 1 + float((w_new * r).sum())
        nav_log.append(nav); pos_log.append(sel)
        w = w_new

    navs = pd.Series(nav_log, index=[dates[i] for i in reb])
    if ret_all:
        return navs, turnovers, pos_log
    return navs, turnovers


def stats(navs, periods_per_year=252 / 20, label=""):
    r = navs.pct_change().dropna().values
    eq = navs.values
    yrs = (navs.index[-1] - navs.index[0]).days / 365.25
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    peak = np.maximum.accumulate(eq)
    mdd = (eq / peak - 1).min()
    return dict(label=label, cagr=cagr, vol=vol, sharpe=cagr / vol if vol else 0,
                mdd=mdd, calmar=cagr / abs(mdd) if mdd else 0,
                total=eq[-1] - 1, yrs=yrs,
                win=(r > 0).mean())


def fmt(s, extra=""):
    return ("  %-30s 年化 %+6.2f%%  波动 %5.1f%%  夏普 %5.2f  最大回撤 %6.1f%%  "
            "Calmar %4.2f  累计 %6.2fx%s"
            % (s["label"], s["cagr"] * 100, s["vol"] * 100, s["sharpe"],
               s["mdd"] * 100, s["calmar"], 1 + s["total"], extra))


def yearly(navs):
    r = navs.pct_change().dropna()
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def main():
    panel = load_panel()
    print("面板 %d 标的 × %d 日 (%s ~ %s)"
          % (panel.shape[1], panel.shape[0], panel.index[0].date(), panel.index[-1].date()))
    avail = {c: panel[c].dropna().index[0].date() for c in panel.columns}
    print("上市日:", ", ".join("%s %s" % (k, v) for k, v in avail.items()))

    runs = {}
    print("\n" + "=" * 104)
    print("全样本 (2013-11 ~ 2026-09, 每 20 交易日调仓, 单边成本 0.05%)")
    print("=" * 104)
    # 基准: 等权全池
    eq_nav = (1 + (panel[POOL].pct_change().mean(axis=1).fillna(0))).cumprod()
    eq_nav = eq_nav.loc[eq_nav.index >= panel.index[250]]
    print(fmt(stats(eq_nav, 252, "等权全池 (买入持有, 无成本)")))
    # 纳指 / 黄金 单持
    for nm in ["纳指", "黄金"]:
        s = panel[nm].dropna()
        s = s.loc[s.index >= panel.index[250]]
        print(fmt(stats(s, 252, "单持 " + nm + " (买入持有, 无成本)")))

    configs = [
        ("mom20 top3 (当前生产代理)", dict(score="mom20", top_k=3, abs_filter=False)),
        ("mom20 top3 + 绝对动量", dict(score="mom20", top_k=3, abs_filter=True)),
        ("mom250 top1", dict(score="mom250", top_k=1)),
        ("mom250 top2", dict(score="mom250", top_k=2)),
        ("mom250 top3", dict(score="mom250", top_k=3)),
        ("mom250+反转过滤 top3 (λ=0.5)", dict(score="mom250_rev", top_k=3, rev_lambda=0.5)),
        ("mom250+反转过滤 top3 (λ=1.0)", dict(score="mom250_rev", top_k=3, rev_lambda=1.0)),
        ("mom250+反转过滤 top5 (λ=0.5)", dict(score="mom250_rev", top_k=5, rev_lambda=0.5)),
    ]
    for lab, cfg in configs:
        navs, turns = run(panel, **cfg)
        s = stats(navs, 252 / 20, lab)
        turn_yr = np.mean(turns) * (252 / 20)
        print(fmt(s, "  年换手 %.0f%%" % (turn_yr * 100)))
        runs[lab] = navs

    # 稳健性: 成本敏感性 / 调仓频率 / 分段
    print("\n" + "=" * 104)
    print("稳健性检验")
    print("=" * 104)
    for c in [0.0, 0.0002, 0.0005, 0.001, 0.002]:
        navs, _ = run(panel, score="mom250_rev", top_k=3, rev_lambda=0.5, cost=c)
        print(fmt(stats(navs, 252 / 20, "12月动量+反转 top3 成本%.2f%%" % (c * 100))))
    for R in [10, 20, 60]:
        navs, turns = run(panel, score="mom250_rev", top_k=3, rev_lambda=0.5, R=R)
        print(fmt(stats(navs, 252 / R, "调仓周期 %d 日" % R),
                  "  年换手 %.0f%%" % (np.mean(turns) * 252 / R * 100)))
    print("\n  分段 (12月动量+反转 top3, 成本0.05%):")
    for lo, hi, lab in [("2013-11-01", "2019-12-31", "前半段 2013-11~2019-12"),
                        ("2020-01-01", "2026-09-30", "后半段 2020-01~2026-09")]:
        navs, _ = run(panel, score="mom250_rev", top_k=3, rev_lambda=0.5, start=lo, end=hi)
        print(fmt(stats(navs, 252 / 20, lab)))
        navs2, _ = run(panel, score="mom250", top_k=3, start=lo, end=hi)
        print(fmt(stats(navs2, 252 / 20, "  对照: 纯 mom250 top3")))

    print("\n  分年度收益 (12月动量+反转 top3):")
    navs, _, _ = run(panel, score="mom250_rev", top_k=3, rev_lambda=0.5, ret_all=True)
    yy = yearly(navs)
    print("    " + "  ".join("%d %+.1f%%" % (y, v * 100) for y, v in yy.items()))
    print("  分年度收益 (等权全池):")
    yy2 = yearly(eq_nav)
    print("    " + "  ".join("%d %+.1f%%" % (y, v * 100) for y, v in yy2.items()))

    # 导出净值
    out = pd.DataFrame({"mom250_top3": runs["mom250 top3"],
                        "mom250_rev_top3": runs["mom250+反转过滤 top3 (λ=0.5)"],
                        "mom20_top3": runs["mom20 top3 (当前生产代理)"],
                        "equal_weight": eq_nav.reindex(runs["mom250 top3"].index).ffill()})
    os.makedirs("audit/out", exist_ok=True)
    out.to_csv("audit/out/strategy_nav.csv")
    print("\n净值已导出: audit/out/strategy_nav.csv")


if __name__ == "__main__":
    main()
