# -*- coding: utf-8 -*-
"""N-12: 我方策略的"延迟一个收盘"压力测试 —— 补上 N-10 指出的最大缺口

背景: 同行 v12_r2 的四种情景里, `lag1_1bp` 比她自己的 `close_1bp` 掉 15.5pp 年化
(V12-R2 53.13%→37.61%), 说明她的策略对"信号当日收盘成交"这个理想假设高度依赖。
我方所有对外数字都建立在同一理想假设上, 但引擎**没有延迟成交档**, 因此无法回答同一问题。

本脚本用引擎自己产出的 actions 做**延迟一天执行的净值重放** (replay-shift):
  1. 跑一次默认基线, 中途截获 (actions, price_mat, cfg);
  2. 自检: 不位移的 replay 必须逐位复现引擎 nav_log (误差 0);
  3. 把每个动作的执行日 +1, 用同一 price_mat 重放 → lag1 净值;
  4. 成本档: slippage=0.0010 (单边 11bp)。
给出与我方/对方同名的四情景: close_1bp / close_11bp / lag1_1bp / lag1_11bp。

⚠️ 口径差异 (必须披露): 对方 `lag1` 是引擎级参数 (决策用 t-lag 日数据、t 日收盘成交, 状态机按
lag 重算); 本脚本是"决策序列取自原始路径、仅把执行推迟一天", 信息→执行的间隔同为 1 天,
但路径依赖 (延迟后的持仓会影响后续锁仓/冷却/年龄) 未重算。因此是**近似**, 不是逐位复刻。

用法: .venv/bin/python audit/n_lag1_stress.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import web.bt_runner as br  # noqa: E402
from conftest import replay_nav  # noqa: E402

START, END = "2014-01-02", "2026-09-24"
SESSIONS = 244.0


def run_capture(params):
    """跑 run_backtest, 截获 compute_signals 的 (actions, price_mat) 与实际 cfg。"""
    cap = {}
    orig_cs, orig_cfg = br.compute_signals, br.SignalConfig

    class Rec(orig_cfg):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            cap["cfg"] = self

    def wrap(cfg, names, price_mat, *a, **kw):
        acts, fn, nl = orig_cs(cfg, names, price_mat, *a, **kw)
        cap["actions"], cap["price_mat"], cap["names"] = acts, price_mat, names
        return acts, fn, nl

    br.SignalConfig, br.compute_signals = Rec, wrap
    try:
        res = br.run_backtest(dict(params))
    finally:
        br.SignalConfig, br.compute_signals = orig_cfg, orig_cs
    return res, cap


def shift_actions(actions, n_days):
    """把每个动作的执行日 +1 (超出末日的丢弃)。"""
    out = []
    for a in actions:
        if a[0] + 1 < n_days:
            out.append((a[0] + 1,) + tuple(a[1:]))
    return out


def stats(nav, dates):
    nav = np.asarray(nav, dtype=float)
    nav = nav / nav[0]
    r = nav[1:] / nav[:-1] - 1.0
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1.0).min())
    for sess, tag in ((244.0, "244"), (252.0, "252")):
        yrs = len(r) / sess
        cagr = float(nav[-1] ** (1 / yrs) - 1) if yrs > 0 else float("nan")
        if tag == "244":
            ann244, dd244 = cagr, dd
        else:
            ann252 = cagr
    return dict(total=float(nav[-1] - 1), ann244=ann244, ann252=ann252, max_dd=dd,
                calmar=ann244 / abs(dd) if dd < 0 else float("nan"))


def main():
    base = dict(universe="纳指,黄金,创业", start_date=START, end_date=END)
    print("=== 情景 1/4: close_1bp (基线, 理想同收盘) ===", flush=True)
    res, cap = run_capture(base)
    cfg = cap["cfg"]
    actions, pm, names = cap["actions"], cap["price_mat"], cap["names"]
    dates = list(res["dates"])
    assert len(dates) == pm.shape[1], (len(dates), pm.shape)

    rep = replay_nav(actions, pm, names, cfg)
    engine = np.array(res["nav_log"], dtype=float)
    err = float(np.max(np.abs(rep - engine)))
    print("  自检 replay(nav) vs 引擎 nav_log 最大误差 = %.3g  (%s)"
          % (err, "OK" if err < 1e-9 else "❌ 不自洽, 结果不可用"))
    if err >= 1e-9:
        sys.exit(1)

    out = {}
    for scen, lag, slip in (("close_1bp", 0, 0.0), ("close_11bp", 0, 0.0010),
                            ("lag1_1bp", 1, 0.0), ("lag1_11bp", 1, 0.0010)):
        c = dataclasses.replace(cfg, slippage=slip)
        acts = shift_actions(actions, pm.shape[1]) if lag else actions
        nav = replay_nav(acts, pm, names, c)
        out[scen] = stats(nav, dates)
        out[scen]["trades"] = int(res["trade_count"])
        print("  %-11s 年化(244) %6.2f%% | 年化(252) %6.2f%% | 回撤 %7.2f%% | Calmar %5.2f | 累计 %+9.1f%%"
              % (scen, out[scen]["ann244"] * 100, out[scen]["ann252"] * 100,
                 out[scen]["max_dd"] * 100, out[scen]["calmar"], out[scen]["total"] * 100), flush=True)

    print("\n=== 对照: 同行 V12-R2 同四情景 (冻结路径, 2014-01-02~2026-09-24) ===")
    P = "/home/ecs-user/peer_artifacts/v12_r2"
    meta = json.load(open(os.path.join(P, "results", "path_metadata.json")))
    peer = {}
    with np.load(os.path.join(P, "results", "ref_paths.npz"), allow_pickle=False) as z:
        for scen in ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp"):
            r = z["v12r2_08b522c2e1aa291f054d__%s__returns" % scen]
            nav = np.concatenate([[1.0], np.cumprod(1 + r)])
            peer[scen] = stats(nav, meta["dates"])
    for scen in ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp"):
        v = peer[scen]
        d = (out[scen]["ann244"] - v["ann244"]) * 100
        print("  %-11s 年化(244) %6.2f%% | 回撤 %7.2f%% | Calmar %5.2f  ｜ 我方 - 对方 = %+6.2fpp"
              % (scen, v["ann244"] * 100, v["max_dd"] * 100, v["calmar"], d))

    print("\n=== 延迟一天 + 加 10bp 滑点 对我方的边际伤害 ===")
    print("  我方: close_1bp %.2f%% → lag1_11bp %.2f%%  (%+.2fpp)"
          % (out["close_1bp"]["ann244"] * 100, out["lag1_11bp"]["ann244"] * 100,
             (out["lag1_11bp"]["ann244"] - out["close_1bp"]["ann244"]) * 100))
    print("  对方: close_1bp %.2f%% → lag1_11bp %.2f%%  (%+.2fpp)"
          % (peer["close_1bp"]["ann244"] * 100, peer["lag1_11bp"]["ann244"] * 100,
             (peer["lag1_11bp"]["ann244"] - peer["close_1bp"]["ann244"]) * 100))
    print("  回撤: 我方 %.2f%% → %.2f%% ; 对方 %.2f%% → %.2f%%"
          % (out["close_1bp"]["max_dd"] * 100, out["lag1_11bp"]["max_dd"] * 100,
             peer["close_1bp"]["max_dd"] * 100, peer["lag1_11bp"]["max_dd"] * 100))

    path = "/home/ecs-user/etf_strategy/audit/out/n_lag1_stress.json"
    json.dump({"ours": out, "peer_v12r2": peer,
               "caveat": "我方 lag1 为 replay-shift 近似(决策序列不重算), 对方为引擎级 lag 参数"},
              open(path, "w"), ensure_ascii=False, indent=1)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
