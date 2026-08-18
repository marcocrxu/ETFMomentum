"""J 组 · yanyan 标的池最优组合穷举 — 2026-08-18

问题: 10 只排名标的中选哪几只组合最好? (2^10 = 1024 子集穷举)
方法: ①基准窗(2016-07-07~2026-08-12)穷举全部子集, 按 Calmar 排名
     ②防过拟合: 训练窗(2016-2021)排名 → 测试窗(2021-2026)评估 top 候选
     ③输出全量结果 + top 表。数据: out/j_yanyan_combo.json
"""
import sys, os, json, itertools, time
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd

from data.feed import TushareFeed
from web.config_local import TUSHARE_TOKEN
from strategy.yanyan import YanyanConfig, UNIVERSE, compute_yanyan
from harness import save

FEED = TushareFeed(TUSHARE_TOKEN, cache_dir="/root/backtest/cache/etf_momentum")
names = [u[0] for u in UNIVERSE]
groups = [u[2] for u in UNIVERSE]
codes = [u[1] for u in UNIVERSE]

W_FULL = ("2016-07-07", "2026-08-12")
W_TR = ("2016-07-07", "2021-06-30")
W_TE = ("2021-07-01", "2026-08-12")


def load():
    dfs = FEED.get_dataframe(codes, "20150301", "20260812")
    _all = sorted(set().union(*[set(df["trade_date"]) for df in dfs.values()]))
    _all = [d for d in _all if pd.Timestamp("20150301") <= d <= pd.Timestamp("20260812")]

    def bp(code):
        s = dfs[code].set_index("trade_date")["close"]
        return np.array([s.get(d, np.nan) for d in _all])
    pm = np.array([bp(c) for c in codes])
    hs = pm[names.index("沪深300")]
    return pm, hs, _all


def metrics_from_log(nav_log):
    nav = np.array(nav_log)
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(len(nav) / 252, 0.01)
    ann = (nav[-1] ** (1 / years) - 1) * 100 if nav[-1] > 0 else -100.0
    calmar = round(ann / abs(mdd), 2) if mdd else 0.0
    return {"total_ret": round((nav[-1] - 1) * 100, 1),
            "ann": round(ann, 2), "mdd": round(mdd, 2), "calmar": calmar,
            "n_days": len(nav)}


def eval_subset(idx_list, pm, hs, s0, e0):
    """idx_list: 排名标的行索引; s0/e0: 日期字符串 → 切片 [s0-210, e0] 保证 MA200 预热"""
    cash_i = names.index("货币")
    day0 = next(i for i, d in enumerate(range_dates) if str(d)[:10] >= s0)
    day1 = next(i for i, d in enumerate(range_dates) if str(d)[:10] >= e0)
    lo = max(0, day0 - 210)
    sub_idx = idx_list + [cash_i]
    pm_sub = pm[sub_idx, lo:day1 + 1]
    hs_sub = hs[lo:day1 + 1]
    names_sub = [names[i] for i in idx_list] + ["货币"]
    groups_sub = [groups[i] for i in idx_list] + ["C"]
    cfg = YanyanConfig()
    actions, nav, log = compute_yanyan(cfg, names_sub, groups_sub, pm_sub, hs_sub,
                                       pm_sub.shape[1])
    # nav_log 从 vs 起算: 用 vs 之后第一个完整日对齐窗口起点 (近似: 取全部 log)
    return metrics_from_log(log)


def main():
    global range_dates
    pm, hs, range_dates = load()
    combos = []
    rank_idx = [i for i, g in enumerate(groups) if g != "C"]
    t0 = time.time()
    for k in range(2, len(rank_idx) + 1):
        for combo in itertools.combinations(rank_idx, k):
            m = eval_subset(list(combo), pm, hs, W_FULL[0], W_FULL[1])
            combos.append({"combo": [names[i] for i in combo],
                           **m})
    combos.sort(key=lambda c: -c["calmar"])
    out = {"n": len(combos), "top_calmar": combos[:20],
           "top_total": sorted(combos, key=lambda c: -c["total_ret"])[:10]}
    save(out, "j_yanyan_combo")
    print("穷举 %d 子集完成, 耗时 %.0f 秒" % (len(combos), time.time() - t0))
    print()
    print("=== 全窗 Calmar TOP 12 ===")
    for c in combos[:12]:
        print("%-28s total %+7.1f%% mdd %+6.2f calmar %5.2f" % (
            "+".join(c["combo"]), c["total_ret"], c["mdd"], c["calmar"]))
    print()
    print("=== 全窗 total TOP 10 ===")
    for c in sorted(combos, key=lambda c: -c["total_ret"])[:10]:
        print("%-28s total %+7.1f%% mdd %+6.2f calmar %5.2f" % (
            "+".join(c["combo"]), c["total_ret"], c["mdd"], c["calmar"]))

    # ── 前向验证: 训练窗 top20 在测试窗的表现 ──
    print()
    print("=== Walk-forward: 训练 2016-2021 选 top → 测试 2021-2026 ===")
    tr_rows = []
    for c in combos:
        idx = [names.index(n) for n in c["combo"]]
        tr_rows.append((c["combo"], eval_subset(idx, pm, hs, W_TR[0], W_TR[1])))
    tr_rows.sort(key=lambda x: -x[1]["calmar"])
    print("训练窗 top8 (按 Calmar):")
    for combo, m in tr_rows[:8]:
        idx = [names.index(n) for n in combo]
        te = eval_subset(idx, pm, hs, W_TE[0], W_TE[1])
        print("  %-26s 训练 cal %5.2f → 测试 total %+7.1f%% mdd %+6.2f cal %5.2f" % (
            "+".join(combo), m["calmar"], te["total_ret"], te["mdd"], te["calmar"]))
    # 全池对照
    te_all = eval_subset(rank_idx, pm, hs, W_TE[0], W_TE[1])
    tr_all = eval_subset(rank_idx, pm, hs, W_TR[0], W_TR[1])
    print("  [全池对照]        训练 cal %5.2f → 测试 total %+7.1f%% mdd %+6.2f cal %5.2f" % (
        tr_all["calmar"], te_all["total_ret"], te_all["mdd"], te_all["calmar"]))


if __name__ == "__main__":
    main()
