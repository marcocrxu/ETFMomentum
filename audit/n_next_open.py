# -*- coding: utf-8 -*-
"""N-22: 次日开盘成交回测 —— 第 t 日收盘出信号, 第 t+1 日**开盘价**成交

与 N-13 的 lag1 不同: lag1 是"推迟一个收盘成交"(仍是收盘价), 本脚本用**真实开盘价**,
日内拆成两段: 旧仓吃 open[t+1]/close[t], 付双边费, 新仓吃 close[t+1]/open[t+1]。

数据: data/feed.py 只对 close 做复权 (open 是原始价), 本脚本按同一公式自行复权:
      open_adj = open_raw × adj_factor / 最新adj_factor
缺失开盘价时**延后换仓**(保留持仓), 与对方"缺行延后"语义一致。

自检 (强校验): 令 open_mat := close_mat 时, 本重放必须与 N-13 的 "actions 位移一天 + close 重放" **逐位一致**
—— 因为开=收时, 旧仓整天吃 close[t+1]/close[t], 新仓当日收益为 0, 正是"推迟一个收盘成交"。

用法: .venv/bin/python audit/n_next_open.py
"""
import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
import web.bt_runner as br  # noqa: E402
from conftest import replay_nav  # noqa: E402
from audit.n_lag1_stress import run_capture, shift_actions  # noqa: E402

START, END = "2014-01-02", "2026-09-24"
CACHE = "/home/ecs-user/backtest/cache/etf_momentum"
SESS = 244.0


def adj_open(code):
    """复权开盘价 (与 feed.py 同一公式)"""
    df = pd.read_csv(os.path.join(CACHE, code.replace(".", "_") + ".csv"),
                     parse_dates=["trade_date"])
    af = df.adj_factor.ffill().bfill().fillna(1.0)
    s = pd.Series((df.open * af / af.iloc[-1]).values, index=df.trade_date)
    return s


def replay_next_open(actions, close_mat, open_mat, names, cfg, lag=1, max_bar=6):
    """lag=1: 第 t 日收盘的动作在**第 t+1 日开盘**执行。返回全长时间轴 nav。"""
    c = cfg.commission + cfg.slippage
    na = len(names)
    n = close_mat.shape[1]
    act_map = {}
    for a in actions:
        act_map.setdefault(a[0], []).append(a)
    # 待执行队列: (触发日, old_p, new_p, reason)
    pending = []
    pos = -1
    nav = 1.0
    out = np.ones(n)
    for t in range(n):
        # 1) 本日待执行的动作 (来自 t-lag 日)
        todo = [a for a in pending if a[0] == t - lag]
        # 2) 旧仓吃 (t-1 → 今开)
        if t > 0:
            if todo:
                ex = open_mat[:, t]
                p0 = close_mat[pos, t - 1] if pos >= 0 else np.nan
                p1 = ex[pos] if pos >= 0 else np.nan
                if pos >= 0 and np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0:
                    nav *= p1 / p0
                elif pos >= 0 and not (np.isfinite(p0) and np.isfinite(p1)):
                    todo = []          # 旧仓无开盘价 → 本日不换仓, 退回普通口径
            if not todo:
                p0 = close_mat[pos, t - 1] if pos >= 0 else np.nan
                p1 = close_mat[pos, t] if pos >= 0 else np.nan
                if pos >= 0 and np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                    nav *= p1 / p0
        # 3) 执行换仓 (付费)
        for _, old_p, new_p, reason in todo:
            if old_p is not None and old_p >= 0:
                nav *= (1 - c)
            if new_p is not None and new_p >= 0:
                nav *= (1 - c)
            pos = new_p if new_p is not None else pos
        # 4) 新仓吃 (今开 → 今收)
        if t > 0 and todo:
            p0 = open_mat[pos, t] if pos >= 0 else np.nan
            p1 = close_mat[pos, t] if pos >= 0 else np.nan
            if pos >= 0 and np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                nav *= p1 / p0
        # 5) 入队本日新动作
        if t in act_map and t < n - lag:
            for a in act_map[t]:
                pending.append(a)
        out[t] = nav
    return out


def stats(nav):
    nav = np.asarray(nav, float); nav = nav / nav[0]
    r = nav[1:] / nav[:-1] - 1
    dd = float((nav / np.maximum.accumulate(nav) - 1).min())
    ann = float(nav[-1] ** (SESS / len(r)) - 1)
    return dict(total=float(nav[-1] - 1), ann=ann * 100, dd=dd * 100,
                cal=(ann / abs(dd) if dd < 0 else float("nan")))


def main():
    res, cap = run_capture(dict(universe="纳指,黄金,创业", start_date=START, end_date=END))
    cfg, acts, cmat, names = cap["cfg"], cap["actions"], cap["price_mat"], cap["names"]
    dates = list(res["dates"])
    assert cmat.shape[1] == len(dates) == len(res["nav_log"])
    assert abs(cfg.crash_alloc - 1.0) < 1e-9, "本脚本只支持 crash_alloc=100% 的简单路径"
    print("标的:", names, "| 天数", len(dates), "| 动作", len(acts))

    # 复权开盘价矩阵
    code_of = br.ALL_CODES
    omat = np.full_like(cmat, np.nan)
    for i, nm in enumerate(names):
        s = adj_open(code_of[nm])
        s = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(END))]
        lut = {d.strftime("%Y-%m-%d"): v for d, v in s.items()}
        omat[i] = [lut.get(d, np.nan) for d in dates]
    cover = np.isfinite(omat).mean() * 100
    print("开盘价覆盖率 %.1f%% (NaN 日会延后换仓)" % cover)

    # ── 自检: open=close 时必须等于 N-13 的 lag1(位移一天 + close 重放) ──
    ref_lag1 = replay_nav(shift_actions(acts, cmat.shape[1]), cmat, names, cfg)
    chk = replay_next_open(acts, cmat, cmat, names, cfg, lag=1)
    err = float(np.max(np.abs(chk - ref_lag1)))
    print("自检 open=close 与 N-13 lag1 的最大差异 = %.3g  (%s)"
          % (err, "OK" if err < 1e-9 else "❌ 不自洽"))
    if err >= 1e-9:
        sys.exit(1)

    # ── 另外两个自检: 开盘价不得有前视 / 覆盖率 ──
    print("开盘价示例 (513100):", [round(float(omat[0, k]), 4) for k in (0, 100, 2500)])

    out = {}
    scen = [("close_1bp  引擎口径(当日收盘成交)", cmat, cmat, 0, 0.0),
            ("close_11bp 当日收盘 + 10bp", cmat, cmat, 0, 0.0010),
            ("lag1_1bp   推迟一个收盘", cmat, cmat, 1, 0.0),
            ("lag1_11bp  推迟一个收盘 + 10bp", cmat, cmat, 1, 0.0010),
            ("nextopen_1bp  次日开盘成交", cmat, omat, 1, 0.0),
            ("nextopen_11bp 次日开盘 + 10bp", cmat, omat, 1, 0.0010)]
    print("\n%-34s %9s %9s %8s %11s" % ("情景", "年化(244)", "回撤", "Calmar", "累计"))
    for tag, cm, om, lag, slip in scen:
        cc = dataclasses.replace(cfg, slippage=slip)
        if lag == 0:
            nav = replay_nav(acts, cm, names, cc)
        else:
            nav = replay_next_open(acts, cm, om, names, cc, lag=lag)
        st = stats(nav)
        out[tag] = st
        print("%-34s %8.2f%% %8.2f%% %8.2f %10.1f%%"
              % (tag, st["ann"], st["dd"], st["cal"], st["total"] * 100))

    b = out["close_1bp  引擎口径(当日收盘成交)"]
    n1 = out["nextopen_1bp  次日开盘成交"]
    n11 = out["nextopen_11bp 次日开盘 + 10bp"]
    print("\n=== 结论数字 ===")
    print("  当日收盘 → 次日开盘:   年化 %+.2fpp, 回撤 %+.2fpp, Calmar %+.2f"
          % (n1["ann"] - b["ann"], n1["dd"] - b["dd"], n1["cal"] - b["cal"]))
    print("  当日收盘 → 次日开盘+10bp: 年化 %+.2fpp, 回撤 %+.2fpp, Calmar %+.2f"
          % (n11["ann"] - b["ann"], n11["dd"] - b["dd"], n11["cal"] - b["cal"]))
    print("  次日开盘 vs 推迟一个收盘: 年化 %+.2fpp"
          % (n1["ann"] - out["lag1_1bp   推迟一个收盘"]["ann"]))

    p = os.path.join(HERE, "audit", "out", "n_next_open.json")
    json.dump({"scenarios": out, "open_coverage_pct": float(cover),
               "selfcheck_maxdiff_vs_lag1": err,
               "note": "第 t 日收盘信号, 第 t+1 日开盘成交; 开盘价按 feed 同一公式自行复权"},
              open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> %s" % p)


if __name__ == "__main__":
    main()
