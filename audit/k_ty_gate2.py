"""K 组补遗 · 美债利率冲击门细网格 + 被拦交易逐笔 + 与 VIX/SKEW 组合 — 2026-08-27

初筛 (k_ty_gate.py) 两个正候选:
  A: 10Y 20个美股交易日涨幅 > 0.4pp → 禁新买/切向纳指  (三窗口 +327/+576/+104pp)
  B: 10Y 20个美股交易日急降 > 0.4pp → 禁新买/切向纳指  (三窗口 +539/+181/+33pp)
本脚本:
  1. A/B 细网格 (0.3~0.6) × 三窗口, 检验平台 vs 尖峰
  2. |20dΔ| 双向往返冲击门 (利率波动率=压力) 候选
  3. A/B 全空仓语义 0.4 档
  4. 默认窗被拦交易逐笔 (baseline 有而 variant 无的纳指买入, 记后续收益)
  5. 与 VIX>25/SKEW>150 组合 (I-8 最强门)
输出: audit/out/k_ty_gate2.json
"""
import sys, os, json
sys.path.insert(0, "/home/ecs-user/etf_strategy"); sys.path.insert(0, "/home/ecs-user/etf_strategy/audit")
import numpy as np, pandas as pd

from harness import load_mats, base_cfg, metrics, save, trades_from_actions
from data.usfeed import load_qqq_cny, align_to_cn, align_fear_to_cn
from strategy.signal import calc_accel_thresholds, compute_signals

WINDOWS = [
    ("默认窗", "2016-07-07", "2026-08-12"),
    ("2015长窗", "2015-01-05", "2026-08-12"),
    ("2020短窗", "2020-01-01", "2026-08-12"),
]


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


def main():
    dgs = pd.read_csv("/home/ecs-user/etf_strategy/audit/out/dgs.csv", parse_dates=["date"])
    dgs = dgs[dgs["DGS10"].notna()].reset_index(drop=True)   # 剔除美股假日行, diff 才=真实交易日
    dgs["d10c20"] = dgs["DGS10"].diff(20)
    dgs["d10d20"] = -dgs["DGS10"].diff(20)
    dgs["d10abs20"] = dgs["DGS10"].diff(20).abs()
    out = {"windows": {}}

    def align_feat(cn_dates, col):
        s = dgs.set_index("date")[col].dropna()
        idx = pd.DatetimeIndex(s.index)
        r = np.full(len(cn_dates), np.nan)
        for i, d in enumerate(cn_dates):
            dt = pd.Timestamp(d) - pd.Timedelta(days=1)
            p = idx.searchsorted(dt, side="right") - 1
            if p >= 0:
                r[i] = s.iloc[p]
        return r

    for wname, s, e in WINDOWS:
        mats = load_mats("纳指,黄金,创业".split(","), start=s, end=e)
        pre = mats["pre_days"]
        aligned = {k: align_feat(mats["all_full"], k) for k in
                   ("d10c20", "d10d20", "d10abs20")}
        win = {k: aligned[k][pre:] for k in aligned}
        sh_full = mats["sh_mom"].copy()

        cfg0 = base_cfg()
        a0, u0 = run_variant(cfg0, mats)
        m0 = metrics(u0, len(u0))
        print("\n===== %s (%s ~ %s) | 基线 %s =====" % (wname, s, e, m0))
        wres = {"baseline": m0, "grid": [], "combos": [], "blocked_trades": []}

        # ── 1/2/3 细网格 ──
        for tag, fk in [("A:10Y20d涨幅>X", "d10c20"),
                        ("B:10Y20d急降>X", "d10d20"),
                        ("C:|10Y20dΔ|>X", "d10abs20")]:
            for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55):
                cfg = base_cfg(vix_gate=th)
                a1, u1 = run_variant(cfg, mats, extra_pre={"vix": win[fk]})
                m1 = metrics(u1, len(u1))
                mask = ~np.isnan(win[fk]) & (win[fk] > th)
                rec = {"tag": tag.replace("X", str(th)), "th": th, "gate_days": int(mask.sum()),
                       **m1, "d_total": round(m1["total_ret"] - m0["total_ret"], 1),
                       "d_calmar": round(m1["calmar"] - m0["calmar"], 2),
                       "d_dd": round(m1["mdd"] - m0["mdd"], 2), "n_actions": len(a1)}
                wres["grid"].append(rec)
                print("  %-26s 门%4d天 | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                      % (rec["tag"], rec["gate_days"], m1["total_ret"], rec["d_total"],
                         m1["mdd"], m1["calmar"], rec["d_calmar"]))

        # ── 4 全空仓语义 0.4 ──
        for tag, fk in [("A0.4全空仓", "d10c20"), ("B0.4全空仓", "d10d20")]:
            mask = ~np.isnan(aligned[fk]) & (aligned[fk] > 0.4)
            shf = np.where(mask, 999.0, sh_full)
            a1, u1 = run_variant(base_cfg(), mats, sh_full=shf)
            m1 = metrics(u1, len(u1))
            rec = {"tag": tag, "gate_days": int(mask[pre:].sum()), **m1,
                   "d_total": round(m1["total_ret"] - m0["total_ret"], 1),
                   "d_calmar": round(m1["calmar"] - m0["calmar"], 2),
                   "d_dd": round(m1["mdd"] - m0["mdd"], 2)}
            wres["grid"].append(rec)
            print("  %-26s 门%4d天 | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                  % (tag, rec["gate_days"], m1["total_ret"], rec["d_total"],
                     m1["mdd"], m1["calmar"], rec["d_calmar"]))

        # ── 5 与 VIX25/SKEW150 组合 (默认窗/长窗才做) ──
        if wname != "2020短窗":
            vix_full = align_fear_to_cn(mats["all_full"], "vix")
            skew_full = align_fear_to_cn(mats["all_full"], "skew")
            vix_w = vix_full[pre:]; skew_w = skew_full[pre:]
            combos = [
                ("VIX25+SKEW150", {"vix": vix_w, "skew": skew_w}, base_cfg(vix_gate=25, skew_gate=150)),
                ("VIX25+SKEW150+A0.4", {"vix": vix_w, "skew": skew_w, }, None),
            ]
            # A0.4 与 VIX 共用 vix 槽位 → 组合需额外通道, 此处用 skew 槽位借道:
            # skew_gate 管道也是 arr[t]>gate, 把 A 特征灌进 skew, gate=0.4
            cfg_c1 = base_cfg(vix_gate=25, skew_gate=150)
            cfg_c1.vix_gate = 25
            # 组合1: VIX25 + SKEW150 (对照)
            a1, u1 = run_variant(cfg_c1, mats, extra_pre={"vix": vix_w, "skew": skew_w})
            m1 = metrics(u1, len(u1))
            wres["combos"].append({"tag": "VIX25+SKEW150", **m1,
                                   "d_total": round(m1["total_ret"] - m0["total_ret"], 1),
                                   "d_calmar": round(m1["calmar"] - m0["calmar"], 2)})
            print("  %-26s | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                  % ("VIX25+SKEW150", m1["total_ret"], wres["combos"][-1]["d_total"],
                     m1["mdd"], m1["calmar"], wres["combos"][-1]["d_calmar"]))
            # 组合2: VIX25 + A0.4 (A 借道 skew 槽)
            cfg_c2 = base_cfg(vix_gate=25, skew_gate=0.4)
            a2, u2 = run_variant(cfg_c2, mats, extra_pre={"vix": vix_w, "skew": win["d10c20"]})
            m2 = metrics(u2, len(u2))
            wres["combos"].append({"tag": "VIX25+A0.4", **m2,
                                   "d_total": round(m2["total_ret"] - m0["total_ret"], 1),
                                   "d_calmar": round(m2["calmar"] - m0["calmar"], 2)})
            print("  %-26s | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                  % ("VIX25+A0.4", m2["total_ret"], wres["combos"][-1]["d_total"],
                     m2["mdd"], m2["calmar"], wres["combos"][-1]["d_calmar"]))
            # 组合3: VIX25 + SKEW150 + A0.4 (借道: skew 特征 = SKEW 主 + A 副, 取 max 语义)
            skewA = np.where((win["d10c20"] > 0.4) | (~np.isnan(skew_w) & (skew_w > 150)), 151.0, np.nan)
            cfg_c3 = base_cfg(vix_gate=25, skew_gate=150)
            a3, u3 = run_variant(cfg_c3, mats, extra_pre={"vix": vix_w, "skew": skewA})
            m3 = metrics(u3, len(u3))
            wres["combos"].append({"tag": "VIX25+SKEW150+A0.4", **m3,
                                   "d_total": round(m3["total_ret"] - m0["total_ret"], 1),
                                   "d_calmar": round(m3["calmar"] - m0["calmar"], 2)})
            print("  %-26s | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                  % ("VIX25+SKEW150+A0.4", m3["total_ret"], wres["combos"][-1]["d_total"],
                     m3["mdd"], m3["calmar"], wres["combos"][-1]["d_calmar"]))
            # 组合4: VIX25 + SKEW150 + B0.4
            skewB = np.where((win["d10d20"] > 0.4) | (~np.isnan(skew_w) & (skew_w > 150)), 151.0, np.nan)
            cfg_c4 = base_cfg(vix_gate=25, skew_gate=150)
            a4, u4 = run_variant(cfg_c4, mats, extra_pre={"vix": vix_w, "skew": skewB})
            m4 = metrics(u4, len(u4))
            wres["combos"].append({"tag": "VIX25+SKEW150+B0.4", **m4,
                                   "d_total": round(m4["total_ret"] - m0["total_ret"], 1),
                                   "d_calmar": round(m4["calmar"] - m0["calmar"], 2)})
            print("  %-26s | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                  % ("VIX25+SKEW150+B0.4", m4["total_ret"], wres["combos"][-1]["d_total"],
                     m4["mdd"], m4["calmar"], wres["combos"][-1]["d_calmar"]))

        # ── 6 被拦交易逐笔 (仅默认窗, A0.4/B0.4) ──
        if wname == "默认窗":
            for tag, fk in [("A0.4", "d10c20"), ("B0.4", "d10d20")]:
                cfg = base_cfg(vix_gate=0.4)
                a1, u1 = run_variant(cfg, mats, extra_pre={"vix": win[fk]})
                t0 = trades_from_actions(a0, mats, u0)
                t1 = trades_from_actions(a1, mats, u1)
                # baseline 中"买入纳指"的交易在 variant 中被拦下的 = 基线有而变体无的纳指入场
                def buys(tl):
                    return {(x["entry_date"], x["asset"]) for x in tl if x["asset"] == "纳指"}
                blocked = sorted(buys(t0) - buys(t1))
                wres["blocked_trades"].append({"tag": tag, "n_blocked": len(blocked),
                                               "dates": blocked})
                print("  [%s] 被拦纳指入场 %d 笔: %s" % (tag, len(blocked), blocked[:12]))
                # 被拦交易的后续收益: 在 baseline 账本里找同入场日的纳指交易
                for ed in blocked:
                    for x in t0:
                        if x["asset"] == "纳指" and x["entry_date"] == ed:
                            print("     %s 买入 → %s 卖出 收益 %+.1f%% (%d天, %s)"
                                  % (ed, x["exit_date"], x["ret"], x["days"], x["exit_reason"]))
                            break
        out["windows"][wname] = wres

    save(out, "k_ty_gate2")


if __name__ == "__main__":
    main()
