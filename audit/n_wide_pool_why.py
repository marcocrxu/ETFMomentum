# -*- coding: utf-8 -*-
"""N-15: 为什么"她的钱来自池外标的", 而我们扩池却是负收益? —— 把亏损拆成两半

N-11 已知: 我方宽池7 总对数 3.423 vs 窄池3 4.481 = **−1.058**。
本脚本把这个 −1.058 严格拆成两部分:
  A. **池外标的自身贡献** (我们持有它们时赚/亏了多少)
  B. **稀释成本** (为持有它们而放弃的 创业/纳指/黄金 交易日价值)
并对每只池外标的做三方对照: 我方排名器 vs 买入持有 vs 她 V12-R2。

用法: .venv/bin/python audit/n_wide_pool_why.py
"""
import json
import os
import sys
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import web.bt_runner as br  # noqa: E402

START, END = "2014-01-02", "2026-09-24"
CACHE = "/home/ecs-user/backtest/cache/etf_momentum"
PEER = "/home/ecs-user/peer_artifacts/v12_r2"

NARROW = ["纳指", "黄金", "创业"]
WIDE = NARROW + ["有色", "红利低波", "科创50", "中证2000"]
CODE = {"纳指": "513100.SH", "黄金": "518880.SH", "创业": "159915.SZ",
        "有色": "512400.SH", "红利低波": "512890.SH", "科创50": "588080.SH",
        "中证2000": "563300.SH"}
PEER_CODE = {"512400": "有色", "512890": "红利低波", "588080": "科创50",
             "563300": "中证2000", "510500": "中证500", "513120": "港股创新药",
             "510300": "沪深300", "159915": "创业", "513100": "纳指",
             "518880": "黄金", "511880": "现金"}


def adj_close(code):
    fn = code.replace(".", "_") + ".csv"
    df = pd.read_csv(os.path.join(CACHE, fn), parse_dates=["trade_date"])
    s = pd.Series((df.close * df.adj_factor).values, index=df.trade_date)
    return s


def run(universe):
    r = br.run_backtest(dict(universe=",".join(universe), start_date=START, end_date=END))
    dates = list(r["dates"])
    nav = np.array(r["nav_log"], dtype=float)
    rets = np.zeros(len(nav)); rets[1:] = nav[1:] / nav[:-1] - 1.0
    pos = np.array(r["position_series"], dtype=int)
    labels = r["position_labels"]
    return dates, rets, pos, labels, nav


def attribute(dates, rets, pos, label_of, extra_only=None):
    acc = {}
    for t in range(1, len(rets)):
        k = label_of(int(pos[t - 1]))
        if extra_only is not None and k not in extra_only:
            continue
        a = acc.setdefault(k, {"days": 0, "logret": 0.0})
        a["days"] += 1
        r = float(rets[t])
        a["logret"] += np.log1p(r) if r > -1 else -np.inf
    return acc


def main():
    nd, n_rets, n_pos, n_lab, n_nav = run(NARROW)
    wd, w_rets, w_pos, w_lab, w_nav = run(WIDE)
    lab_n = lambda i: ("空仓" if i < 0 else n_lab[i + 1].split("(")[0])
    lab_w = lambda i: ("空仓" if i < 0 else w_lab[i + 1].split("(")[0])
    an = attribute(nd, n_rets, n_pos, lab_n)
    aw = attribute(wd, w_rets, w_pos, lab_w)

    tot_n = sum(v["logret"] for v in an.values())
    tot_w = sum(v["logret"] for v in aw.values())
    print("=== 总量 ===")
    print("  我方窄池3 总对数 %.3f (累计 %+.1f%%) | 我方宽池7 总对数 %.3f (累计 %+.1f%%) | 差 %+.3f"
          % (tot_n, (np.exp(tot_n) - 1) * 100, tot_w, (np.exp(tot_w) - 1) * 100, tot_w - tot_n))

    extra = {"有色", "红利低波", "科创50", "中证2000"}
    shared = {"纳指", "黄金", "创业", "空仓"}
    ae = sum(v["logret"] for k, v in aw.items() if k in extra)
    print("\n=== 把 −%.3f 拆成两半 ===" % (tot_n - tot_w))
    print("  A. 池外 4 只标的自身贡献（我方排名器持有时）: %+.3f  (天数 %d)"
          % (ae, sum(v["days"] for k, v in aw.items() if k in extra)))
    print("  B. 共同标的的稀释损失: %+.3f" % (
        sum(aw.get(k, {}).get("logret", 0) - an.get(k, {}).get("logret", 0) for k in shared)))
    print("\n  逐标的 (我方窄池 → 我方宽池):")
    for k in ("创业", "纳指", "黄金", "空仓") + tuple(sorted(extra)):
        a, b = an.get(k, {"days": 0, "logret": 0.0}), aw.get(k, {"days": 0, "logret": 0.0})
        print("    %-8s 天数 %4d→%4d  对数 %+7.3f→%+7.3f  差 %+7.3f"
              % (k, a["days"], b["days"], a["logret"], b["logret"], b["logret"] - a["logret"]))

    # ── 每只池外标的: 我方排名器 vs 买入持有 vs 她 ──
    meta = json.load(open(os.path.join(PEER, "results", "path_metadata.json")))
    assets = json.load(open(os.path.join(PEER, "results", "feature_metadata.json")))["assets"]
    pdates = list(meta["dates"])
    with np.load(os.path.join(PEER, "results", "ref_paths.npz"), allow_pickle=False) as z:
        p_rets = z["v12r2_08b522c2e1aa291f054d__close_1bp__returns"]
        p_hold = z["v12r2_08b522c2e1aa291f054d__close_1bp__holdings"]
    pid = {d: i for i, d in enumerate(pdates)}
    wi = {d: i for i, d in enumerate(wd)}

    print("\n=== 池外标的逐只对照 (2014-01-02 ~ %s) ===" % END)
    print("  %-10s %-28s %-28s %-28s" % ("标的", "我方排名器(持有期)", "买入持有(全区间)", "她 V12-R2(持有期)"))
    rows = {}
    for code, name in (("512400", "有色"), ("588080", "科创50"),
                       ("563300", "中证2000"), ("512890", "红利低波"),
                       ("510500", "中证500"), ("513120", "港股创新药")):
        s = adj_close(code + ".SH")
        s = s[(s.index >= START) & (s.index <= END)]
        sdates = [d.strftime("%Y-%m-%d") for d in s.index]
        # 我方: 持有该标的的日期 (position 索引 = WIDE.index(name)+1... 用标签匹配)
        ours_days = [wd[t] for t in range(1, len(w_rets)) if lab_w(int(w_pos[t - 1])) == name]
        oc = sum(np.log1p(w_rets[t]) for t in range(1, len(w_rets))
                 if lab_w(int(w_pos[t - 1])) == name and w_rets[t] > -1)
        bh = float(np.log(s.iloc[-1] / s.iloc[0]))
        # 她: 该 code 在 assets 里的索引
        idx = assets.index(code) if code in assets else None
        hd, hc = 0, 0.0
        if idx is not None:
            for t in range(1, len(p_rets)):
                if int(p_hold[t - 1]) == idx:
                    hd += 1
                    if p_rets[t] > -1:
                        hc += float(np.log1p(p_rets[t]))
        rows[name] = dict(code=code, our_days=len(ours_days), our_log=oc, bh_log=bh,
                          peer_days=hd, peer_log=hc, first=sdates[0], last=sdates[-1])
        print("  %-10s %4d天 %+7.3f (%5.1fbp/天)   %+7.3f (%s起)  %4d天 %+7.3f (%5.1fbp/天)"
              % (name, len(ours_days), oc, oc / max(len(ours_days), 1) * 1e4,
                 bh, sdates[0], hd, hc, hc / max(hd, 1) * 1e4))

    # ── 机会成本: 持有池外标的的那些日子, 创业/纳指 在干什么 ──
    print("\n=== 机会成本: 我方持有池外标的的日子, 共同标的同期表现 ===")
    cyb = adj_close("159915.SZ"); nz = adj_close("513100.SH")
    cyb = cyb[(cyb.index >= START) & (cyb.index <= END)]
    nz = nz[(nz.index >= START) & (nz.index <= END)]
    cdates = [d.strftime("%Y-%m-%d") for d in cyb.index]
    ci = {d: i for i, d in enumerate(cdates)}
    nzi = {d: i for i, d in enumerate([d.strftime("%Y-%m-%d") for d in nz.index])}
    ndates_nz = [d.strftime("%Y-%m-%d") for d in nz.index]
    hold_extra = [wd[t] for t in range(1, len(w_rets)) if lab_w(int(w_pos[t - 1])) in extra]
    def bh_on(series, idxmap, days):
        tot = 0.0
        for d in days:
            i = idxmap.get(d)
            if i is None or i == 0:
                continue
            r = series[i] / series[i - 1] - 1
            if np.isfinite(r) and r > -1:
                tot += float(np.log1p(r))
        return tot
    print("  持有池外标的共 %d 天; 这些日子里 创业 买入持有 %+.3f, 纳指 %+.3f"
          % (len(hold_extra), bh_on(cyb.values, ci, hold_extra), bh_on(nz.values, nzi, hold_extra)))
    print("  (同期我方在池外标的上实际赚到 %+.3f)" % ae)

    # ── 我们的排名器在这些标的上是否跑输买入持有? ──
    print("\n=== 结论指标: 排名器 vs 买入持有 (只算**我方池里有**的 4 只) ===")
    hold4 = ("有色", "科创50", "中证2000", "红利低波")
    tot_our = sum(rows[k]["our_log"] for k in hold4)
    tot_bh = sum(rows[k]["bh_log"] for k in hold4)
    tot_peer = sum(rows[k]["peer_log"] for k in hold4)
    tot_peer_bh = sum(rows[k]["bh_log"] for k in hold4)   # 同区间同标的买入持有
    print("  我方排名器 %+.3f | 买入持有 %+.3f | **排名器超额 %+.3f**" % (tot_our, tot_bh, tot_our - tot_bh))
    print("  她 V12-R2   %+.3f | 买入持有 %+.3f | **她排名器超额 %+.3f**" % (tot_peer, tot_peer_bh, tot_peer - tot_peer_bh))

    print("\n=== 稀释损失的来源: 每个共同标的『被放弃的那些交易日』值多少 ===")
    for k in ("创业", "纳指", "黄金", "空仓"):
        a, b = an.get(k, {"days": 0, "logret": 0.0}), aw.get(k, {"days": 0, "logret": 0.0})
        dd = a["days"] - b["days"]
        lost = a["logret"] - b["logret"]
        per = lost / dd * 1e4 if dd else 0.0
        print("  %-6s 让出 %4d 天, 这些天平均 %+6.1fbp/天, 合计 %+.3f"
              % (k, dd, per, -lost))
    print("  换来的 868 天池外标的平均 %+.1fbp/天 (合计 %+.3f)"
          % (ae / 868 * 1e4, ae))

    out = dict(total_narrow=tot_n, total_wide=tot_w, extra_assets_log=ae,
               dilution_log=tot_w - tot_n - ae, per_asset=rows,
               shared_narrow={k: an.get(k) for k in shared},
               shared_wide={k: aw.get(k) for k in shared},
               opportunity_cost=dict(days=len(hold_extra),
                                     cyb_log=bh_on(cyb.values, ci, hold_extra),
                                     nz_log=bh_on(nz.values, nzi, hold_extra)),
               ranker_vs_bh=dict(assets=list(hold4), ours=tot_our, buyhold=tot_bh,
                                 excess=tot_our - tot_bh, peer=tot_peer,
                                 peer_excess=tot_peer - tot_peer_bh))
    path = os.path.join(HERE, "audit", "out", "n_wide_pool_why.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
