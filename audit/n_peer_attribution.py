# -*- coding: utf-8 -*-
"""N-11: 收益归因 —— 同行 V12-R2 的钱到底从哪些标的/哪些时段赚出来

方法 (与双方引擎的计时约定一致, 见 v12/native.cpp:31,185):
  日 t 的收益由**当日持有的仓位**赚取, 而 holdings[t] 记录的是 t 日收盘调仓**之后**的仓位,
  故 returns[t] 归属 holdings[t-1]。
  归因量 = 持仓期日对数收益之和 (可加), 与我方同口径并列。

时段: 全期 2014-01-02~2026-09-24 / 长熊 2021-03-01~2024-09-30 / 第二轮牛 2024-10-01~2026-09-24
对手: V12-R2 (部署主推送), 参照 V10-H; 我方: bt_runner 默认基线。

用法: .venv/bin/python audit/n_peer_attribution.py
"""
import json
import os
import sys
from bisect import bisect_left, bisect_right

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.bt_runner import run_backtest  # noqa: E402

PEER = "/home/ecs-user/peer_artifacts/v12_r2"
START, END = "2014-01-02", "2026-09-24"
NAMES = {"159915": "创业板", "588080": "科创50", "510300": "沪深300", "510500": "中证500",
         "563300": "中证2000", "512400": "有色", "512890": "红利低波", "513100": "纳指",
         "513120": "港股创新药", "518880": "黄金", "511880": "货币/现金"}
PERIODS = [("全期", "2014-01-02", "2026-09-24"),
           ("长熊", "2021-03-01", "2024-09-30"),
           ("第二轮牛", "2024-10-01", "2026-09-24")]
ROLES = {"V12-R2": "v12r2_08b522c2e1aa291f054d", "V10-H": "v12r2_2b63e64ab0b9e24b8cab"}


def peer_series():
    meta = json.load(open(os.path.join(PEER, "results", "path_metadata.json")))
    assets = json.load(open(os.path.join(PEER, "results", "feature_metadata.json")))["assets"]
    dates = list(meta["dates"])
    out = {}
    with np.load(os.path.join(PEER, "results", "ref_paths.npz"), allow_pickle=False) as z:
        for role, cid in ROLES.items():
            out[role] = (z["%s__close_1bp__returns" % cid], z["%s__close_1bp__holdings" % cid])
    return dates, assets, out


def our_series(universe="纳指,黄金,创业"):
    r = run_backtest(dict(universe=universe, start_date=START, end_date=END))
    dates = list(r["dates"])
    nav = np.array(r["nav_log"], dtype=float)
    rets = np.zeros(len(nav)); rets[1:] = nav[1:] / nav[:-1] - 1.0
    pos = np.array(r["position_series"], dtype=int)          # -1=空仓, i=etf_names[i]
    labels = r["position_labels"]
    return dates, rets, pos, labels


def attribute(dates, rets, pos, label_of):
    """returns[t] 归 pos[t-1] (当日持仓赚当日收益)。返回按标的汇总。"""
    acc = {}
    for t in range(1, len(rets)):
        key = label_of(int(pos[t - 1]))
        a = acc.setdefault(key, {"days": 0, "logret": 0.0, "switches": 0, "win_days": 0})
        a["days"] += 1
        r = float(rets[t])
        a["logret"] += np.log1p(r) if r > -1 else -np.inf
        if r > 0:
            a["win_days"] += 1
        if int(pos[t]) != int(pos[t - 1]):
            a["switches"] += 1
    for k, a in acc.items():
        a["contrib_pct"] = (np.exp(a["logret"]) - 1) * 100
        a["avg_daily_bp"] = a["logret"] / max(a["days"], 1) * 1e4
        a["win_rate"] = a["win_days"] / max(a["days"], 1)
    return acc


def slice_period(dates, series, a, b):
    lo, hi = bisect_left(dates, a), bisect_right(dates, b)
    return series[lo:hi]


def report(name, dates, rets, pos, label_of, a, b):
    idx = [i for i, d in enumerate(dates) if a <= d <= b]
    d2 = [dates[i] for i in idx]
    r2, p2 = rets[idx], pos[idx]
    acc = attribute(d2, r2, p2, label_of)
    total_log = sum(v["logret"] for v in acc.values())
    rows = sorted(acc.items(), key=lambda kv: -kv[1]["logret"])
    print("\n--- %s | %s (%s ~ %s) 总对数收益 %.3f (累计 %+.1f%%) ---"
          % (name, [p[0] for p in PERIODS if p[1] == a][0], a, b,
             total_log, (np.exp(total_log) - 1) * 100))
    print("  %-12s %6s %8s %12s %10s %8s %8s" %
          ("标的", "天数", "占比", "对数收益贡献", "折合收益", "日均bp", "胜率"))
    for k, v in rows:
        print("  %-12s %6d %7.1f%% %12.3f %9.1f%% %8.2f %7.1f%%"
              % (k, v["days"], 100 * v["days"] / max(sum(x["days"] for x in acc.values()), 1),
                 v["logret"], v["contrib_pct"], v["avg_daily_bp"], v["win_rate"] * 100))
    return acc


def main():
    pd_, assets, peers = peer_series()
    od, o_rets, o_pos, o_labels = our_series()
    common = sorted(set(od) & set(pd_))
    oi = {d: i for i, d in enumerate(od)}
    pi = {d: i for i, d in enumerate(pd_)}
    # 重采样到共同日期
    o_r = np.array([o_rets[oi[d]] for d in common])
    o_p = np.array([o_pos[oi[d]] for d in common])
    # position_series: -1=空仓, 0/1/2=etf_names[0/1/2]; position_labels = ["空仓"] + etf_names
    # 前端口径见 web/static/js/charts.js:270 (labels[p + 1])
    lab_our = lambda i: ("空仓" if i < 0 else o_labels[i + 1].split("(")[0])
    res = {}
    for a, b in [(p[1], p[2]) for p in PERIODS]:
        res["ours@" + a] = report("我方窄池3", common, o_r, o_p, lab_our, a, b)
        for role in ROLES:
            rr, hh = peers[role]
            r = np.array([rr[pi[d]] for d in common])
            h = np.array([hh[pi[d]] for d in common])
            lab = lambda i, _a=assets: (NAMES.get(_a[i], _a[i]))
            res["%s@%s" % (role, a)] = report(role, common, r, h, lab, a, b)

    # 集中度: 最好 N 天占全部正对数收益的比例
    print("\n===== 收益集中度 (全期, 对数收益) =====")
    for nm, r in [("我方", o_r)] + [(k, np.array([peers[k][0][pi[d]] for d in common]))
                                    for k in ROLES]:
        lg = np.log1p(r)
        top10 = np.sort(lg)[-10:].sum()
        pos_sum = lg[lg > 0].sum()
        print("  %-8s 总对数 %7.3f | 前10日 %7.3f (占正收益 %.1f%%) | 交易日胜率 %.1f%%"
              % (nm, lg.sum(), top10, 100 * top10 / pos_sum, 100 * (lg > 0).mean()))

    # ── 我方宽池 (F4 已否决) 的同口径归因: 那些"我们没有的标的"在我们引擎里是赚是亏? ──
    print("\n===== 对照: 我方引擎跑宽池7 (无条件扩池, F4 已否决) 的归因 =====")
    wd, w_rets, w_pos, w_labels = our_series("纳指,黄金,创业,有色,红利低波,科创50,中证2000")
    lab_w = lambda i: ("空仓" if i < 0 else w_labels[i + 1].split("(")[0])
    res["ours_wide7@2014-01-02"] = report("我方宽池7", wd, w_rets, w_pos, lab_w,
                                          "2014-01-02", "2026-09-24")
    res["delta_ours_narrow_vs_wide"] = {}
    for k in set(res["ours@2014-01-02"]) | set(res["ours_wide7@2014-01-02"]):
        a = res["ours@2014-01-02"].get(k, {}).get("logret", 0.0)
        b = res["ours_wide7@2014-01-02"].get(k, {}).get("logret", 0.0)
        res["delta_ours_narrow_vs_wide"][k] = b - a
    path = "/home/ecs-user/etf_strategy/audit/out/n_peer_attribution.json"
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
