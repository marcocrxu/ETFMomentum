"""K 组补遗2 · 被拦交易逐笔明细 + VIX/SKEW 管道自检 — 2026-08-27"""
import sys, os
sys.path.insert(0, "/home/ecs-user/etf_strategy"); sys.path.insert(0, "/home/ecs-user/etf_strategy/audit")
import numpy as np, pandas as pd

from harness import load_mats, base_cfg, metrics, trades_from_actions
from data.usfeed import load_qqq_cny, align_to_cn, align_fear_to_cn
from strategy.signal import calc_accel_thresholds, compute_signals


def run_variant(cfg, mats, extra_pre=None, sh_full=None):
    pre = mats["pre_days"]
    m20 = mats["mom20_mat"].copy(); m10 = mats["mom10_mat"].copy()
    if getattr(cfg, "nz_mom_qqq_on", True) and "纳指" in mats["etf_names"]:
        _w = min(max(float(getattr(cfg, "nz_mom_qqq_w", 0.7)), 0.0), 1.0)
        _qq = align_to_cn(mats["all_full"], load_qqq_cny())
        _qq = pd.Series(_qq).ffill().values
        _nz = mats["etf_names"].index("纳指")
        def momw(a, w):
            m = np.full(len(a), np.nan); m[w:] = (a[w:] / a[:-w] - 1) * 100
            return m
        m20[_nz] = (1 - _w) * m20[_nz] + _w * momw(_qq, 20)
        m10[_nz] = (1 - _w) * m10[_nz] + _w * momw(_qq, 10)
    pre_mats = {"accel_th": calc_accel_thresholds(m10, cfg)[:, pre:]}
    if extra_pre:
        pre_mats.update(extra_pre)
    sh = sh_full[pre:] if sh_full is not None else mats["sh_mom"][pre:]
    actions, final_nav, nav_log = compute_signals(
        cfg, mats["etf_names"], mats["price_mat"][:, pre:],
        m20[:, pre:], m10[:, pre:], sh, mats["cyb_p"], mats["sh_price"],
        mats["n_full"] - pre, pre_days=pre, pre_mats=pre_mats)
    vs_user = (mats["n_full"] - pre) - len(nav_log)
    return actions, [1.0] * vs_user + list(nav_log)


mats = load_mats("纳指,黄金,创业".split(","), start="2016-07-07", end="2026-08-12")
pre = mats["pre_days"]
names = mats["etf_names"]
nz = names.index("纳指")

dgs = pd.read_csv("/home/ecs-user/etf_strategy/audit/out/dgs.csv", parse_dates=["date"])
dgs = dgs[dgs["DGS10"].notna()].reset_index(drop=True)   # 剔除美股假日行
dgs["d10c20"] = dgs["DGS10"].diff(20)
dgs["d10d20"] = -dgs["DGS10"].diff(20)
dgs["d10abs20"] = dgs["DGS10"].diff(20).abs()


def align_feat(col):
    s = dgs.set_index("date")[col].dropna()
    idx = pd.DatetimeIndex(s.index)
    r = np.full(len(mats["all_full"]), np.nan)
    for i, d in enumerate(mats["all_full"]):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        p = idx.searchsorted(dt, side="right") - 1
        if p >= 0:
            r[i] = s.iloc[p]
    return r


feat = {k: align_feat(k) for k in ("d10c20", "d10d20", "d10abs20")}

# VIX/SKEW 管道自检
vix_full = align_fear_to_cn(mats["all_full"], "vix")
skew_full = align_fear_to_cn(mats["all_full"], "skew")
vw = vix_full[pre:]; sw = skew_full[pre:]
print("VIX>25 天数: %d | SKEW>150 天数: %d (决策区)" % ((vw > 25).sum(), (sw > 150).sum()))

cfg0 = base_cfg()
a0, u0 = run_variant(cfg0, mats)
m0 = metrics(u0, len(u0))
print("基线: %s" % m0)

cfg_vs = base_cfg(vix_gate=25, skew_gate=150)
a_vs, u_vs = run_variant(cfg_vs, mats, extra_pre={"vix": vw, "skew": sw})
print("VIX25+SKEW150: %s | actions %d -> %d" % (metrics(u_vs, len(u_vs)), len(a0), len(a_vs)))

print("\n========== 逐笔被拦明细 (默认窗) ==========")
for tag, fk, th in [("C:|10Y20dΔ|>0.4(已入引擎)", "d10abs20", 0.4),
                    ("A:10Y20d涨幅>0.4", "d10c20", 0.4),
                    ("B:10Y20d急降>0.4", "d10d20", 0.4)]:
    cfg = base_cfg(vix_gate=th)
    a1, u1 = run_variant(cfg, mats, extra_pre={"vix": feat[fk][pre:]})
    m1 = metrics(u1, len(u1))
    t0 = trades_from_actions(a0, mats, u0)
    t1 = trades_from_actions(a1, mats, u1)
    buys0 = {(x["entry_date"], x["asset"]) for x in t0 if x["asset"] == "纳指"}
    buys1 = {(x["entry_date"], x["asset"]) for x in t1 if x["asset"] == "纳指"}
    blocked = sorted(d for d, a in (buys0 - buys1))
    print("\n[%s] Δtotal %+.1fpp Δcalmar %+.2f | 被拦 %d 笔" % (tag, m1["total_ret"] - m0["total_ret"], m1["calmar"] - m0["calmar"], len(blocked)))
    for ed in blocked:
        hits = [x for x in t0 if x["asset"] == "纳指" and x["entry_date"] == ed]
        if not hits:
            print("   %s: (账本未找到?)" % (ed,))
            continue
        x = hits[0]
        # variant 里同一日附近的替代动作: 找 variant 中该日前后首个买入
        print("   %s 买入 → %s 出场, 基线该笔收益 %+.1f%% (%d天, %s)"
              % (ed, x["exit_date"], x["ret"], x["days"], x["exit_reason"]))
    # 变体中新出现的纳指入场 (延迟买入)
    extra = sorted(d for d, a in (buys1 - buys0))
    for ed in extra:
        hits = [x for x in t1 if x["asset"] == "纳指" and x["entry_date"] == ed]
        if hits:
            print("   [+] 变体延迟买入 %s (基线无), 该笔收益 %+.1f%%" % (ed, hits[0]["ret"]))

# 门触发日期区间 (决策区)
print("\n========== 门触发时段 ==========")
for tag, fk, th in [("A>0.4", "d10c20", 0.4), ("B>0.4", "d10d20", 0.4), ("C>0.4", "d10abs20", 0.4)]:
    f = feat[fk][pre:]
    dates = [str(d)[:10] for d in mats["all_full"][pre:]]
    mask = ~np.isnan(f) & (f > th)
    print("%s: %d 天" % (tag, mask.sum()))
    if mask.any():
        dd = [d for d, m in zip(dates, mask) if m]
        # 聚类为连续段
        runs = []
        st = dd[0]; last = dd[0]
        for d in dd[1:]:
            if (pd.Timestamp(d) - pd.Timestamp(last)).days <= 5:
                last = d
            else:
                runs.append((st, last)); st = d; last = d
        runs.append((st, last))
        for a, b in runs:
            print("    %s ~ %s" % (a, b))
