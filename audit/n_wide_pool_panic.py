# -*- coding: utf-8 -*-
"""N-8: 波动缩放急跌离场在**宽池**上的复测 —— 检验"宽池才有效"的假设

N-7 结论: 窄池 3 上 1.5%~5% 整条急跌阈值带全负 (含同行 V12-C 精确口径)。
同行在宽池 (11 只, 含科创/中证2000/有色等高波标的) 上测出急跌组件 +5.5pp,
推测机制是"宽池持仓常是高波标的, 急跌=趋势破坏; 窄池扛过去更优"。
本脚本在**我方宽池**上直接验证这个假设。

池:
  A 窄池3       纳指,黄金,创业                        (对照, N-7 已测)
  B 宽池7       纳指,黄金,创业,有色,红利低波,科创50,中证2000  (无条件扩池, F4 已否决)
  C 宽池7-低波  纳指,黄金,创业,有色,科创50,中证2000    (去掉低波的红利低波, 检验 2% 下限解释)
  D 条件池      同 B, 但 bull_pool_extra 仅牛市纳入   (最接近同行的 regime 条件池设计)

用法: .venv/bin/python audit/n_wide_pool_panic.py
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
EXTRA = "有色,红利低波,科创50,中证2000"
EXTRA_NOLOWVOL = "有色,科创50,中证2000"

POOLS = [
    ("A_窄池3", "纳指,黄金,创业", {}),
    ("B_宽池7", "纳指,黄金,创业," + EXTRA, {}),
    ("C_宽池6去低波", "纳指,黄金,创业," + EXTRA_NOLOWVOL, {}),
    ("D_条件池7", "纳指,黄金,创业," + EXTRA,
     {"bull_pool_extra": EXTRA, "bull_pool_rule": "sh_mom20"}),
]

# (标签, 参数字典)
CONFIGS = [
    ("基线(急跌关)", {}),
    ("固定 4%", {"panic_drop": 4.0}),
    ("固定 2.5%", {"panic_drop": 2.5}),
    ("波动 当日VOL20 × 1.4", {"panic_vol_on": True, "panic_vol_k": 1.4,
                              "panic_vol_src": "current20"}),
    ("波动 prior_max20_60 × 1.25 (=V12-C倍数)", {"panic_vol_on": True, "panic_vol_k": 1.25,
                                                  "panic_vol_src": "prior_max20_60"}),
    ("波动 prior_max20_60 × 1.40 (=V12-R2)", {"panic_vol_on": True, "panic_vol_k": 1.4,
                                               "panic_vol_src": "prior_max20_60"}),
    ("波动 prior_max20_60 × 2.50", {"panic_vol_on": True, "panic_vol_k": 2.5,
                                    "panic_vol_src": "prior_max20_60"}),
]


def main():
    out = []
    for pname, univ, extra in POOLS:
        for ws, we, wtag in WINDOWS:
            print("\n===== %s | %s (%s ~ %s) =====" % (pname, wtag, ws, we), flush=True)
            base_ann = None
            for cname, ckw in CONFIGS:
                p = dict(universe=univ, start_date=ws, end_date=we)
                p.update(extra)
                p.update(ckw)
                t0 = time.time()
                r = run_backtest(p)
                if "error" in r:
                    print("  %-40s ERROR %s" % (cname, r["error"]), flush=True)
                    continue
                if base_ann is None:
                    base_ann = r["ann_ret"]
                d = r["ann_ret"] - base_ann
                print("  %-40s ann %6.2f%% (%+6.2fpp) dd %6.2f%% cal %5.2f trades %3d (%.0fs)"
                      % (cname, r["ann_ret"], d, r["max_dd"], r["calmar"],
                         r["trade_count"], time.time() - t0), flush=True)
                out.append(dict(pool=pname, window=wtag, cfg=cname, ann=r["ann_ret"],
                                dd=r["max_dd"], cal=r["calmar"], sharpe=r.get("sharpe"),
                                trades=r["trade_count"], total=r["total_ret"],
                                d_ann=(None if cname.startswith("基线") else d)))
    path = "/home/ecs-user/etf_strategy/audit/out/n_wide_pool_panic.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
