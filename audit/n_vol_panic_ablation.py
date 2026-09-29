# -*- coding: utf-8 -*-
"""N-7: 波动缩放急跌阈值消融 —— 复刻同行 V12-C 的 panic_mode='volatility'

背景: 2026-08-25 的 v9_learn_ablation 只测了**固定阈值**分支 (panic_drop=3/4/5%, 全窗负优化)。
同行真正在用的是**波动缩放**分支: thr=clip(K×VOL20, 2%, 10%), 且 V12-C 把 VOL 口径从
"当日 VOL20" 改成 "**上一实际报价日** max(VOL20, VOL60)" (risk_context=prior_max20_60,
prior_clock = previous actual quoted row of that same asset)。
本脚本把两支都跑一遍, 三个窗口。

用法: .venv/bin/python audit/n_vol_panic_ablation.py [quick|full]
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.bt_runner import run_backtest  # noqa: E402

WINDOWS = [
    ("2016-07-07", "2026-08-20", "默认窗(主样本)"),
    ("2015-01-05", "2026-08-20", "2015长窗"),
    ("2020-01-01", "2026-08-20", "2020短窗"),
]
BASE_UNIV = "纳指,黄金,创业"


def run(tag, start, end, **kw):
    p = {"universe": BASE_UNIV, "start_date": start, "end_date": end}
    p.update(kw)
    t0 = time.time()
    r = run_backtest(p)
    if "error" in r:
        print("%-52s ERROR %s" % (tag, r["error"]), flush=True)
        return None
    row = {"tag": tag, "ann": r["ann_ret"], "dd": r["max_dd"], "cal": r["calmar"],
           "sharpe": r.get("sharpe"), "total": r["total_ret"], "trades": r["trade_count"],
           "yearly": r.get("yearly")}
    print("%-52s ann %6.2f%% dd %6.2f%% cal %5.2f sharpe %4.2f trades %3d (%.0fs)"
          % (tag, row["ann"], row["dd"], row["cal"], row["sharpe"] or 0, row["trades"],
             time.time() - t0), flush=True)
    return row


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    out = []
    ks = (1.0, 1.25, 1.4, 1.5, 1.75, 2.0, 2.5) if mode == "full" else (1.25, 1.4, 1.5)
    priors = ("prior_max20_60", "prior20") if mode == "full" else ("prior_max20_60",)
    for ws, we, wtag in WINDOWS:
        print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we), flush=True)
        r = run("基线(急跌离场关)", ws, we)
        if r:
            out.append(dict(r, window=wtag, kind="baseline"))
        for pd_ in (3.0, 4.0, 5.0):
            r = run("固定阈值 panic_drop=%.0f%%" % pd_, ws, we, panic_drop=pd_)
            if r:
                out.append(dict(r, window=wtag, kind="fixed", k=pd_))
        for k in ks:
            r = run("波动缩放 当日VOL20 × %.2f" % k, ws, we,
                    panic_vol_on=True, panic_vol_k=k, panic_vol_src="current20")
            if r:
                out.append(dict(r, window=wtag, kind="vol_current20", k=k))
        for src in priors:
            for k in (ks if src == "prior_max20_60" else (1.4, 1.5, 2.0)):
                r = run("波动缩放 %s × %.2f" % (src, k), ws, we,
                        panic_vol_on=True, panic_vol_k=k, panic_vol_src=src)
                if r:
                    out.append(dict(r, window=wtag, kind="vol_" + src, k=k))
    path = "/home/ecs-user/etf_strategy/audit/out/n_vol_panic_ablation.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
