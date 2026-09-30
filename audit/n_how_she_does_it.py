# -*- coding: utf-8 -*-
"""N-17: 她到底"怎么"做到的 —— 持仓段(leg)级别的机制对比

N-16 证明差距来自"在好日子里持有"的效率 (创业板 16.0x vs 我方 9.2x)。
本脚本拆到"每一段持有的进出场"这一层, 回答 HOW:
  1. 双方持仓段长度/收益分布对比
  2. 我方每一次退出的理由分布, 以及"下车后标的还涨了多少"(下车机会成本)
  3. 她自己每一段的进出场特征(进场时的 mom20、出场时的 mom5), 反推她的实际规则
  4. 关键牛市段的逐日归属对比

用法: .venv/bin/python audit/n_how_she_does_it.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import web.bt_runner as br  # noqa: E402

START, END = "2014-01-02", "2026-09-24"
CACHE = "/home/ecs-user/backtest/cache/etf_momentum"
PEER = "/home/ecs-user/peer_artifacts/v12_r2"
NAME = {"159915": "创业板", "588080": "科创50", "510300": "沪深300", "510500": "中证500",
        "563300": "中证2000", "512400": "有色", "512890": "红利低波", "513100": "纳指",
        "513120": "港股创新药", "518880": "黄金", "511880": "货币"}
CODE2NAME = {"513100.SH": "纳指", "518880.SH": "黄金", "159915.SZ": "创业板"}


def adj(code):
    suf = "_SZ" if code in ("159915",) else "_SH"
    df = pd.read_csv(os.path.join(CACHE, code + suf + ".csv"), parse_dates=["trade_date"])
    s = pd.Series((df.close * df.adj_factor).values, index=df.trade_date)
    return s[(s.index >= START) & (s.index <= END)]


def legs(dates, hold_label):
    """把 (日期, 持仓标签) 压成连续持仓段。"""
    out, cur = [], None
    for d, lab in zip(dates, hold_label):
        if cur is None or lab != cur["label"]:
            if cur:
                out.append(cur)
            cur = dict(label=lab, start=d, days=1)
        else:
            cur["days"] += 1
            cur["end"] = d
    if cur:
        out.append(cur)
    return out


def main():
    # ── 我方 ──
    r = br.run_backtest(dict(universe="纳指,黄金,创业", start_date=START, end_date=END))
    od = list(r["dates"])
    opos = np.array(r["position_series"], dtype=int)
    olab = r["position_labels"]
    _NM = {"创业": "创业板"}
    olab_of = lambda i: ("空仓" if i < 0 else _NM.get(olab[i + 1].split("(")[0],
                                                       olab[i + 1].split("(")[0]))
    o_hold = [olab_of(int(p)) for p in opos]
    dec = r["decisions"]

    # ── 她 ──
    meta = json.load(open(os.path.join(PEER, "results", "path_metadata.json")))
    assets = json.load(open(os.path.join(PEER, "results", "feature_metadata.json")))["assets"]
    pdates = list(meta["dates"])
    with np.load(os.path.join(PEER, "results", "ref_paths.npz"), allow_pickle=False) as z:
        pr = z["v12r2_08b522c2e1aa291f054d__close_1bp__returns"]
        ph = z["v12r2_08b522c2e1aa291f054d__close_1bp__holdings"]
    p_lab_by_idx = {i: NAME.get(assets[i], assets[i]) for i in range(len(assets))}
    # holdings[t] 是调仓后仓位 → 当日持仓为 holdings[t-1]
    p_hold = ["空仓"] + [p_lab_by_idx[int(ph[t - 1])] for t in range(1, len(ph))]
    pi = {d: i for i, d in enumerate(pdates)}
    common = [d for d in od if d in pi]
    oi = {d: i for i, d in enumerate(od)}
    oc = np.array([o_hold[oi[d]] for d in common], dtype=object)
    pc = np.array([p_hold[pi[d]] for d in common], dtype=object)
    o_r = np.array([r["nav_log"][oi[d]] for d in common]) if False else None

    print("=== 1. 持仓段 (leg) 分布, 共同区间 %s ~ %s (%d 天) ===" % (common[0], common[-1], len(common)))
    for tag, hold in (("我方", oc), ("她 V12-R2", pc)):
        L = legs(common, hold)
        # 每段收益: 用标的复权价
        rets = []
        for lg in L:
            if lg["label"] == "空仓":
                continue
            s = adj("159915" if lg["label"] == "创业板" else
                    "513100" if lg["label"] == "纳指" else "518880")
            sd = [d for d in s.index.strftime("%Y-%m-%d")]
            a, b = sd.index(lg["start"]), sd.index(lg.get("end", lg["start"]))
            rets.append(dict(label=lg["label"], days=lg["days"],
                             ret=float(s.iloc[b] / s.iloc[a] - 1),
                             start=lg["start"]))
        df = pd.DataFrame(rets)
        print("\n  --- %s: 共 %d 段 ---" % (tag, len(df)))
        print("  %-8s %5s %8s %8s %9s %9s %9s" % ("标的", "段数", "中位天数", "均天数", "中位收益", "均收益", "胜率"))
        for lab, g in df.groupby("label"):
            print("  %-8s %5d %8.0f %8.1f %9.2f%% %9.2f%% %8.0f%%"
                  % (lab, len(g), g.days.median(), g.days.mean(),
                     g.ret.median() * 100, g.ret.mean() * 100, (g.ret > 0).mean() * 100))
        print("  %-8s %5d %8.0f %8.1f %9.2f%% %9.2f%% %8.0f%%"
              % ("合计", len(df), df.days.median(), df.days.mean(),
                 df.ret.median() * 100, df.ret.mean() * 100, (df.ret > 0).mean() * 100))

    # ── 2. 我方退出理由 + 下车后标的还涨多少 ──
    print("\n=== 2. 我方退出理由分布 + 下车机会成本 (卖出后该标 20/60 日收益) ===")
    cnt = Counter()
    rows = []
    for d in dec:
        act = d.get("action", "")
        rsn = d.get("reason", "")
        cnt[rsn.split(":")[0].split("(")[0]] += 1
        if act.startswith("卖出") or act.startswith("切换"):
            body = act.replace("卖出", "").replace("切换", "").strip()
            name = body.split("→")[0].strip() if "→" in body else body
            if name not in ("纳指", "黄金", "创业"):
                continue
            code = {"纳指": "513100", "黄金": "518880", "创业": "159915"}[name]
            s = adj(code); sd = list(s.index.strftime("%Y-%m-%d"))
            if d["date"] not in sd:
                continue
            i = sd.index(d["date"])
            f20 = float(s.iloc[i + 20] / s.iloc[i] - 1) if i + 20 < len(s) else np.nan
            f60 = float(s.iloc[i + 60] / s.iloc[i] - 1) if i + 60 < len(s) else np.nan
            rows.append(dict(date=d["date"], asset=name, reason=rsn[:24], f20=f20, f60=f60))
    for k, v in cnt.most_common():
        print("  %-28s %3d 次" % (k, v))
    dfx = pd.DataFrame(rows)
    print("\n  卖出后 20/60 日该标的收益 (正=卖早了):")
    for lab, g in dfx.groupby("asset"):
        print("  %-6s n=%3d  20日 中位 %+6.2f%% 均值 %+7.2f%% | 60日 中位 %+7.2f%% 均值 %+8.2f%% | 20日为正比例 %.0f%%"
              % (lab, len(g), g.f20.median() * 100, g.f20.mean() * 100,
                 g.f60.median() * 100, g.f60.mean() * 100, (g.f20 > 0).mean() * 100))
    print("\n  按退出理由看 (只看卖后 60 日):")
    for lab, g in dfx.groupby("reason"):
        if len(g) >= 4:
            print("  %-26s n=%3d  60日均值 %+8.2f%%  中位 %+7.2f%%" %
                  (lab, len(g), g.f60.mean() * 100, g.f60.median() * 100))

    # ── 3. 关键: 创业板 大牛段谁在场 ──
    print("\n=== 3. 创业板大牛段逐日归属 (谁在场) ===")
    cyb = adj("159915"); csd = list(cyb.index.strftime("%Y-%m-%d"))
    for a, b, tag in (("2019-01-01", "2021-02-28", "2019~2021结构牛"),
                      ("2024-10-01", "2026-09-24", "2024-10~2026第二轮牛"),
                      ("2015-01-01", "2015-06-12", "2015杠杆牛")):
        idx = [i for i, d in enumerate(common) if a <= d <= b]
        if not idx:
            continue
        sa = cyb.index[cyb.index <= pd.Timestamp(b)][-1]
        cyb_ret = float(cyb.loc[sa] / cyb.loc[cyb.index[cyb.index >= pd.Timestamp(a)][0]] - 1)
        oh = sum(1 for i in idx if oc[i] == "创业板")
        ph_ = sum(1 for i in idx if pc[i] == "创业板")
        onz = sum(1 for i in idx if oc[i] == "纳指"); pnz = sum(1 for i in idx if pc[i] == "纳指")
        og = sum(1 for i in idx if oc[i] == "黄金"); pg = sum(1 for i in idx if pc[i] == "黄金")
        print("  %-22s 创业板涨 %+7.1f%%  %4d天 | 我方持创业板 %3d天 她 %3d天 | 纳指 我%3d/她%3d | 黄金 我%3d/她%3d"
              % (tag, cyb_ret * 100, len(idx), oh, ph_, onz, pnz, og, pg))


    # ── 4. 她的创业板 32 段 vs 我们 77 段: 她抢跑了哪些 ──
    print("\n=== 4. 创业板逐段对照: 她进场时我们在哪 ===")
    def legs_of(dates, hold, target):
        out, cur = [], None
        for d, lab in zip(dates, hold):
            if lab == target:
                if cur is None:
                    cur = dict(start=d, days=1, end=d)
                else:
                    cur["days"] += 1; cur["end"] = d
            elif cur is not None:
                out.append(cur); cur = None
        if cur: out.append(cur)
        return out
    ol = legs_of(common, oc, "创业板")
    pl = legs_of(common, pc, "创业板")
    print("  我方 %d 段, 她 %d 段" % (len(ol), len(pl)))
    csd2 = {d: i for i, d in enumerate(cyb.index.strftime("%Y-%m-%d"))}
    print("\n  她每一段: 进场日 / 天数 / 段收益 / 该日我们持什么")
    oi2 = {d: i for i, d in enumerate(common)}
    sum_ret, sum_days, ours_elsewhere = 0.0, 0, Counter()
    for lg in pl:
        a, b = lg["start"], lg["end"]
        if a not in csd2 or b not in csd2:
            continue
        ret = float(cyb.iloc[csd2[b]] / cyb.iloc[csd2[a]] - 1)
        sum_ret += ret; sum_days += lg["days"]
        ours_elsewhere[oc[oi2[a]]] += 1
    print("    她的创业板合计: %d 段, %d 天, 段收益之和 %.1f%% (复利 %.1f%%)"
          % (len(pl), sum_days, sum_ret * 100, (np.prod([1 + float(cyb.iloc[csd2[l["end"]]] / cyb.iloc[csd2[l["start"]]] - 1)
                                                          for l in pl if l["start"] in csd2 and l["end"] in csd2]) - 1) * 100))
    print("    她进场那天我们持: %s" % dict(ours_elsewhere))
    print("\n  我方创业板每一段:")
    sum_ret2, sum_days2 = 0.0, 0
    for lg in ol:
        a, b = lg["start"], lg["end"]
        if a not in csd2 or b not in csd2:
            continue
        sum_ret2 += float(cyb.iloc[csd2[b]] / cyb.iloc[csd2[a]] - 1); sum_days2 += lg["days"]
    print("    我方创业板合计: %d 段, %d 天, 段收益之和 %.1f%%" % (len(ol), sum_days2, sum_ret2 * 100))
    print("    => 她以 %.0f%% 的持有天数拿到段收益之和的 %.0f%%"
          % (100 * sum_days / max(sum_days2, 1), 100 * sum_ret / max(sum_ret2, 1e-9)))

    # 她持有创业板的日子 vs 我们持有创业板的日子: 这些天创业板分别涨多少
    def day_stats(hold, target):
        # ⚠️ 2026-09-29 修正: 区间 (t-1 → t) 的收益由**当日持仓** hold[t] 赚取, 不是 hold[t-1]。
        # 原写法 hold[t-1] 是 off-by-one, 曾导致 N-17 §4 误报"0.49 对数数据疑点"(实为我方脚本 bug)。
        tot, n = 0.0, 0
        for t in range(1, len(common)):
            if hold[t] == target:
                d0, d1 = common[t - 1], common[t]
                if d0 in csd2 and d1 in csd2:
                    tot += float(np.log(cyb.iloc[csd2[d1]] / cyb.iloc[csd2[d0]])); n += 1
        return n, tot
    no, lo_ = day_stats(oc, "创业板")
    np_, lp_ = day_stats(pc, "创业板")
    print("\n  持有创业板期间: 我方 %d 天 对数 %+.3f (%.1fbp/天) | 她 %d 天 对数 %+.3f (%.1fbp/天)"
          % (no, lo_, lo_ / no * 1e4, np_, lp_, lp_ / max(np_, 1) * 1e4))

    out = dict(common=[common[0], common[-1]], days=len(common))
    path = os.path.join(HERE, "audit", "out", "n_how_she_does_it.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
