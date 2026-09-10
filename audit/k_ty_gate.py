"""K 组 · 十年期/一年期美债收益率过滤器消融 — 2026-08-27

用户提议: 把 10Y/1Y 美债纳入过滤器。此前 I-7 测过 10Y-3M 倒挂禁纳指 → 否决(错过2023-24 AI行情);
10Y-1Y 是不同信号(利差更窄、更贴政策利率), 此处系统消融:
  利差门(倒挂/深度倒挂) / 水平门(10Y>X / 1Y>X) / 利率冲击门(20d/60d Δ) / 陡峭化门 / 急降门
  × 两种语义: (a) 禁新买/切向纳指 [复用 vix_gate 管道, 语义=arr[t]>gate]
             (b) 全池空仓 [sh_mom 置顶仿真上证空仓]
数据: FRED DGS1/DGS10 (免费CSV, 日频), 对齐 CN 交易日 (无前视: CN日D 用美国日期≤D-1 收盘)。
基线: 当前默认口径 (QQQ 0.7 混合开, accel_cooldown=3 黄金限定等全部默认)。
输出: audit/out/k_ty_gate.json
"""
import sys, os, json, time
sys.path.insert(0, "/home/ecs-user/etf_strategy"); sys.path.insert(0, "/home/ecs-user/etf_strategy/audit")
import numpy as np, pandas as pd

from harness import load_mats, base_cfg, metrics, save
from strategy.signal import calc_accel_thresholds, compute_signals
from data.usfeed import load_qqq_cny, align_to_cn

DGS_CSV = "/home/ecs-user/etf_strategy/audit/out/dgs.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
EPS = 0.001  # vix_gate 管道要求 gate>0, 用 epsilon 表示"严格大于0"

WINDOWS = [
    ("默认窗", "2016-07-07", "2026-08-12"),
    ("2015长窗", "2015-01-05", "2026-08-12"),
    ("2020短窗", "2020-01-01", "2026-08-12"),
    ("25/26窗", "2025-01-01", "2026-08-12"),
]


def load_dgs():
    if not os.path.exists(DGS_CSV):
        import requests
        frames = []
        for sym in ("DGS1", "DGS10"):
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                             params={"id": sym, "cosd": "2014-01-01"}, headers=UA, timeout=30)
            r.raise_for_status()
            lines = r.text.strip().splitlines()
            df = pd.DataFrame([l.split(",") for l in lines[1:]], columns=["date", sym])
            df[sym] = pd.to_numeric(df[sym], errors="coerce")
            frames.append(df.dropna())
        d = frames[0].merge(frames[1], on="date", how="outer").sort_values("date")
        d.to_csv(DGS_CSV, index=False)
    df = pd.read_csv(DGS_CSV, parse_dates=["date"])
    return df


def align_dgs(cn_dates, df, col):
    """CN 交易日 D → 美国日期 ≤ D-1 的最后收盘 (无前视)"""
    s = df.set_index("date")[col].dropna()
    idx = pd.DatetimeIndex(s.index)
    out = np.full(len(cn_dates), np.nan)
    for i, d in enumerate(cn_dates):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        p = idx.searchsorted(dt, side="right") - 1
        if p >= 0:
            out[i] = s.iloc[p]
    return out


def feats(dgs):
    """在美股日频上构建特征序列 (再对齐 CN 日); 先剔除假日行使 diff 为真实交易日间隔"""
    dgs = dgs[dgs["DGS10"].notna()].reset_index(drop=True)
    def fr(v):
        return pd.DataFrame({"date": dgs["date"], "v": v})
    return {
        "inv": fr(dgs["DGS1"] - dgs["DGS10"]),               # 10Y-1Y 倒挂度 (=-(10Y-1Y))
        "d10": fr(dgs["DGS10"]),
        "d1": fr(dgs["DGS1"]),
        "d10c20": fr(dgs["DGS10"].diff(20)),                 # 10Y 20个美股交易日涨幅
        "d10c60": fr(dgs["DGS10"].diff(60)),
        "stc60": fr((dgs["DGS10"] - dgs["DGS1"]).diff(60)),  # 熊陡: 利差60日走阔
        "d10d20": fr(-dgs["DGS10"].diff(20)),                # 利率急降(避险冲击)
    }


def run_variant(cfg, mats, extra_pre=None, sh_full=None):
    """harness.run_cfg 复刻 + extra_pre 注入 + sh_mom 可覆写 (全空仓仿真)"""
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
    dgs = load_dgs()
    print("DGS 数据: %s ~ %s, %d 行" % (dgs["date"].min(), dgs["date"].max(), len(dgs)))
    F = feats(dgs)
    out = {"dgs_range": [str(dgs["date"].min()), str(dgs["date"].max())], "windows": {}}

    for wname, s, e in WINDOWS:
        mats = load_mats("纳指,黄金,创业".split(","), start=s, end=e)
        pre = mats["pre_days"]
        # 特征对齐到 CN 全日历, 再切片到决策区
        aligned = {k: align_dgs(mats["all_full"], F[k], "v") for k in F}
        feats_win = {k: aligned[k][pre:] for k in aligned}

        cfg0 = base_cfg()
        a0, u0 = run_variant(cfg0, mats)
        m0 = metrics(u0, len(u0))
        print("\n===== %s (%s ~ %s) =====" % (wname, s, e))
        print("  基线(QQQ0.7默认): %s | %d actions" % (m0, len(a0)))

        # 全空仓仿真: sh_mom 置顶
        sh_full = mats["sh_mom"].copy()
        sh_boost = lambda mask: np.where(mask, 999.0, sh_full)

        variants = []
        # (标签, 语义, 特征key, 阈值, QQQ开关覆盖)
        # a) 禁新买/切向纳指 (vix_gate 管道)
        for tag, fk, th in [
            ("10Y-1Y倒挂<0 禁纳指", "inv", EPS),
            ("10Y-1Y<-0.25 禁纳指", "inv", 0.25),
            ("10Y-1Y<-0.5 禁纳指", "inv", 0.5),
            ("10Y>4.5% 禁纳指", "d10", 4.5),
            ("10Y>5.0% 禁纳指", "d10", 5.0),
            ("1Y>5.0% 禁纳指", "d1", 5.0),
            ("1Y>5.25% 禁纳指", "d1", 5.25),
            ("10Y 20dΔ>0.4pp 禁纳指", "d10c20", 0.4),
            ("10Y 20dΔ>0.6pp 禁纳指", "d10c20", 0.6),
            ("10Y 60dΔ>1.0pp 禁纳指", "d10c60", 1.0),
            ("利差60d走阔>0.5pp 禁纳指", "stc60", 0.5),
            ("10Y 20d急降>0.4pp 禁纳指", "d10d20", 0.4),
        ]:
            variants.append((tag, "gate", fk, th, None))
        # b) 全池空仓 (sh_mom 置顶)
        variants += [
            ("10Y-1Y倒挂 全空仓", "cash", "inv", EPS, None),
            ("10Y>5.0% 全空仓", "cash", "d10", 5.0, None),
            ("10Y 20dΔ>0.6pp 全空仓", "cash", "d10c20", 0.6, None),
        ]

        wres = {"baseline": m0, "variants": []}
        for tag, sem, fk, th, qqq_override in variants:
            arr_full = aligned[fk]
            mask = ~np.isnan(arr_full) & (arr_full > th)
            cfg = base_cfg()
            if qqq_override is not None:
                cfg.nz_mom_qqq_on = qqq_override
            extra = shf = None
            if sem == "gate":
                cfg.vix_gate = th
                extra = {"vix": feats_win[fk]}
            else:
                shf = sh_boost(mask)
            a1, u1 = run_variant(cfg, mats, extra_pre=extra, sh_full=shf)
            m1 = metrics(u1, len(u1))
            n_gate = int(np.sum(mask[pre:])) if sem == "gate" else int(np.sum(mask[pre:]))
            rec = {"tag": tag, "sem": sem, "gate_days": n_gate, **m1,
                   "d_total": round(m1["total_ret"] - m0["total_ret"], 1),
                   "d_calmar": round(m1["calmar"] - m0["calmar"], 2),
                   "d_dd": round(m1["mdd"] - m0["mdd"], 2),
                   "n_actions": len(a1)}
            wres["variants"].append(rec)
            print("  %-34s 门天数%4d | total %8.1f%% (Δ%+8.1f) dd %6.2f cal %4.2f (Δ%+.2f)"
                  % (tag, n_gate, m1["total_ret"], rec["d_total"], m1["mdd"], m1["calmar"], rec["d_calmar"]))

        # 交叉校验: 纯ETF口径 + 倒挂门, 对照 I-7 的 10Y-3M 倒挂结果 (2541.8/1.86)
        cfg_off = base_cfg(nz_mom_qqq_on=False)
        a_off, u_off = run_variant(cfg_off, mats)
        m_off = metrics(u_off, len(u_off))
        cfg_off.vix_gate = EPS
        a_i, u_i = run_variant(cfg_off, mats, extra_pre={"vix": feats_win["inv"]})
        m_i = metrics(u_i, len(u_i))
        wres["crosscheck_qqqoff"] = {"baseline": m_off, "inv_gate": m_i}
        print("  [QQQ关交叉校验] 基线 %s | 倒挂门 %s" % (m_off, m_i))

        out["windows"][wname] = wres

    save(out, "k_ty_gate")


if __name__ == "__main__":
    main()
