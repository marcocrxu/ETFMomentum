"""有效性检验: mom250·top3·cap75%·等权 到底有没有 edge
1) 随机化检验: 同样结构但随机选 3 只 (排除/保留 cap 过滤), 500 次 -> 分布与 p 值
2) 对照: 等权全池 / 等权候选池 / 单持黄金 / 单持纳指
3) 配对检验: 每期收益 减 等权全池, Newey-West t
4) 去后见之明: 只用 A股标的 (剔除 纳指/黄金/有色/港创药) 重跑
"""
import importlib.util
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ps", "audit/param_search.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)

COST, R, TOP, CAP = 0.0005, 10, 3, 0.75
S0, S1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31")

panel = ps.load_panel()
P, names, n_a, pm, cash_i, fac, vol = ps.build(panel)
dates = panel.index
mom = fac["mom250"]
idx = {n: i for i, n in enumerate(names)}


def run(sub_mask=None, use_cap=True, pick="mom", rng=None, cost=COST,
        start=S0, end=S1, ret_series=False):
    mask = pm.copy() if sub_mask is None else (pm & sub_mask)
    reb = [i for i in range(250, len(P) - R, R)]
    reb = [i for i in reb if start <= dates[i] <= end]
    w = np.zeros(n_a); nav = 1.0; log = []
    for i in reb:
        S = mom[i]
        cand = np.where(mask & ~np.isnan(S) & ~np.isnan(P[i]))[0]
        if use_cap:
            cand = cand[S[cand] <= CAP]
        w_new = np.zeros(n_a)
        if cand.size:
            if pick == "mom":
                o = cand[np.argsort(-S[cand])][:TOP]
            else:
                o = rng.choice(cand, size=min(TOP, cand.size), replace=False)
            w_new[o] = 1.0 / len(o)
        else:
            w_new[cash_i] = 1.0
        nav *= (1 - cost * float(np.abs(w_new - w).sum()))
        p0, p1 = P[i], P[i + R]
        r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, 0.0)
        gross = float((w_new * r).sum())
        nav *= (1 + gross)
        log.append((dates[i], gross - cost * float(np.abs(w_new - w).sum()), nav))
        w = w_new
    navs = pd.Series([x[2] for x in log], index=[x[0] for x in log])
    if ret_series:
        return np.array([x[1] for x in log]), navs
    return navs


def stat(navs):
    r = navs.pct_change().dropna().values
    yrs = (navs.index[-1] - navs.index[0]).days / 365.25
    cagr = navs.iloc[-1] ** (1 / yrs) - 1
    v = r.std(ddof=1) * np.sqrt(252 / R)
    peak = np.maximum.accumulate(navs.values)
    return cagr, v, cagr / v, (navs.values / peak - 1).min()


print("=" * 92)
print("窗口 2016-01 ~ 2025-12  (%d 次调仓, 单边成本 0.05%%)" % len(run()))
print("=" * 92)
base = run()
c, v, s, mdd = stat(base)
print("① 基准(实际策略 mom250+cap75%%): 年化 %+.2f%%  波动 %.1f%%  夏普 %.2f  最大回撤 %.1f%%" % (c * 100, v * 100, s, mdd * 100))

for lab, use_cap in [("随机选3只 + cap75%%", True), ("随机选3只 · 无cap", False)]:
    rng = np.random.default_rng(42)
    res = [stat(run(use_cap=use_cap, pick="rnd", rng=rng)) for _ in range(500)]
    cg = np.array([x[0] for x in res]); sh = np.array([x[2] for x in res]); dd = np.array([x[3] for x in res])
    p = (sh >= s).mean()
    print("\n② %s (500 次随机):" % lab)
    print("   年化 均值 %+.2f%% (5%%~95%%: %+.2f%%~%+.2f%%)  夏普 均值 %.2f (5%%~95%%: %.2f~%.2f)  最大回撤 中位 %.1f%%"
          % (cg.mean() * 100, np.percentile(cg, 5) * 100, np.percentile(cg, 95) * 100,
             sh.mean(), np.percentile(sh, 5), np.percentile(sh, 95), np.median(dd) * 100))
    print("   实际策略夏普 %.2f 在随机分布中的分位: %.1f%%  (p 值 = %.3f)" % (s, (sh < s).mean() * 100, p))

# 等权对照
reb = [i for i in range(250, len(P) - R, R)]
reb = [i for i in reb if S0 <= dates[i] <= S1]
eq = []; nav = 1.0; eqr = []
for i in reb:
    p0, p1 = P[i], P[i + R]
    r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, np.nan)
    m = ~np.isnan(r) & pm
    g = r[m].mean()
    nav *= (1 + g); eq.append(nav); eqr.append(g)
eq = pd.Series(eq, index=[dates[i] for i in reb])
c2, v2, s2, m2 = stat(eq)
print("\n③ 等权全池(每10日再平衡, 无成本): 年化 %+.2f%%  波动 %.1f%%  夏普 %.2f  最大回撤 %.1f%%" % (c2 * 100, v2 * 100, s2, m2 * 100))
for nm in ["黄金", "纳指", "有色"]:
    pp = panel[nm].dropna(); pp = pp[(pp.index >= dates[reb[0]]) & (pp.index <= dates[reb[-1] + R])]
    cc, vv, ss, mm = stat(pp)
    print("   单持 %-4s 年化 %+.2f%%  波动 %.1f%%  夏普 %.2f  最大回撤 %.1f%%" % (nm, cc * 100, vv * 100, ss, mm * 100))

# 配对检验
rr, _ = run(ret_series=True)
excess = rr - np.array(eqr)
mean_ex = excess.mean(); n = len(excess)
x = excess - mean_ex
s0 = (x * x).sum() / n
for l in range(1, R + 1):
    s0 += 2 * (1 - l / (R + 1)) * (x[l:] * x[:-l]).sum() / n
se = np.sqrt(s0 / n)
print("\n④ 每期超额(策略 - 等权全池): 均值 %+.3f%%/期 (年化 %+.2f%%), 胜率 %.0f%%, Newey-West t = %+.2f"
      % (mean_ex * 100, mean_ex * (252 / R) * 100, (excess > 0).mean() * 100, mean_ex / se))

# 去后见之明: 只用 A股标的
drop = ["纳指", "黄金", "有色", "港股创新药"]
sub = np.array([n not in drop for n in names])
print("\n⑤ 只用 A股标的(剔除 %s):" % "/".join(drop))
b2 = run(sub_mask=sub); c3, v3, s3, m3 = stat(b2)
print("   mom250+cap75%%: 年化 %+.2f%%  夏普 %.2f  最大回撤 %.1f%%" % (c3 * 100, s3, m3 * 100))
rng = np.random.default_rng(7)
res = [stat(run(sub_mask=sub, pick="rnd", rng=rng)) for _ in range(500)]
sh = np.array([x[2] for x in res]); cg = np.array([x[0] for x in res])
print("   随机3只: 年化均值 %+.2f%%  夏普均值 %.2f (5%%~95%%: %.2f~%.2f)  → 实际夏普分位 %.1f%%"
      % (cg.mean() * 100, sh.mean(), np.percentile(sh, 5), np.percentile(sh, 95), (sh < s3).mean() * 100))
eqs = []
nav = 1.0
for i in reb:
    p0, p1 = P[i], P[i + R]
    r = np.where((p0 > 0) & ~np.isnan(p0) & ~np.isnan(p1), p1 / p0 - 1, np.nan)
    m = ~np.isnan(r) & sub
    if m.sum() >= 2:
        nav *= (1 + r[m].mean())
    eqs.append(nav)
e2 = pd.Series(eqs, index=[dates[i] for i in reb])
c4, v4, s4, m4 = stat(e2)
print("   等权A股池: 年化 %+.2f%%  夏普 %.2f  最大回撤 %.1f%%" % (c4 * 100, s4, m4 * 100))
