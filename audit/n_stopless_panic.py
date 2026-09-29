# -*- coding: utf-8 -*-
"""N-9: 急跌离场是不是被我们自己的止损"顶掉"了？—— 关掉止损后再测

N-8 结论: 宽池上急跌离场比窄池**更负** —— "宽池才有效"的假设被否。
但两边还有一个根本差异: 她的 V12 配置里**没有止损**
(`trail_stop=0.0`, 无固定止损线), 她唯一的左尾保护就是急跌离场 + 抄底锁仓;
而我方引擎**默认就有** -5% 相对止损 + 绝对止损 + 加速止盈。
若急跌离场的信息已被我方止损覆盖, 那它在此处只会贡献"重复触发 + 甩下车"的噪声。

本脚本: 关掉我方止损 (stop_loss→-99 即失效, abs_stop_on=False) 后再测急跌,
看它是否从"负优化"变成"正贡献"。三种池 × 两个窗口。

用法: .venv/bin/python audit/n_stopless_panic.py
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.bt_runner import run_backtest  # noqa: E402

WINDOWS = [("2016-07-07", "2026-08-20", "默认窗(主样本)"),
           ("2015-01-05", "2026-08-20", "2015长窗")]
EXTRA = "有色,红利低波,科创50,中证2000"
POOLS = [
    ("A_窄池3", "纳指,黄金,创业", {}),
    ("B_宽池7", "纳指,黄金,创业," + EXTRA, {}),
    ("D_条件池7", "纳指,黄金,创业," + EXTRA,
     {"bull_pool_extra": EXTRA, "bull_pool_rule": "sh_mom20"}),
]
NOSTOP = {"stop_loss": -99.0, "abs_stop_on": False}
PANICS = [
    ("固定 4%", {"panic_drop": 4.0}),
    ("波动 当日VOL20 × 1.4", {"panic_vol_on": True, "panic_vol_k": 1.4,
                              "panic_vol_src": "current20"}),
    ("波动 prior_max20_60 × 1.25", {"panic_vol_on": True, "panic_vol_k": 1.25,
                                    "panic_vol_src": "prior_max20_60"}),
    ("波动 prior_max20_60 × 1.40", {"panic_vol_on": True, "panic_vol_k": 1.4,
                                    "panic_vol_src": "prior_max20_60"}),
    ("波动 prior_max20_60 × 2.50", {"panic_vol_on": True, "panic_vol_k": 2.5,
                                    "panic_vol_src": "prior_max20_60"}),
]


def main():
    out = []
    for pname, univ, extra in POOLS:
        for ws, we, wtag in WINDOWS:
            print("\n===== %s | %s =====" % (pname, wtag), flush=True)
            rows = {}
            for label, kw in [("止损ON 基线", {})] + [("止损ON " + n, d) for n, d in PANICS] + \
                             [("止损OFF 基线", NOSTOP)] + \
                             [("止损OFF " + n, dict(NOSTOP, **d)) for n, d in PANICS]:
                p = dict(universe=univ, start_date=ws, end_date=we)
                p.update(extra); p.update(kw)
                t0 = time.time()
                r = run_backtest(p)
                if "error" in r:
                    print("  %-34s ERROR %s" % (label, r["error"]), flush=True)
                    continue
                rows[label] = r
                print("  %-34s ann %6.2f%% dd %6.2f%% cal %5.2f trades %3d (%.0fs)"
                      % (label, r["ann_ret"], r["max_dd"], r["calmar"], r["trade_count"],
                         time.time() - t0), flush=True)
            for stopstate, basekey in (("止损ON", "止损ON 基线"), ("止损OFF", "止损OFF 基线")):
                b = rows.get(basekey)
                if not b:
                    continue
                for n, _ in PANICS:
                    r = rows.get(stopstate + " " + n)
                    if not r:
                        continue
                    out.append(dict(pool=pname, window=wtag, stop=stopstate, cfg=n,
                                    ann=r["ann_ret"], dd=r["max_dd"], cal=r["calmar"],
                                    trades=r["trade_count"],
                                    d_ann=r["ann_ret"] - b["ann_ret"],
                                    d_dd=r["max_dd"] - b["max_dd"],
                                    d_cal=r["calmar"] - b["calmar"]))
                out.append(dict(pool=pname, window=wtag, stop=stopstate, cfg="基线",
                                ann=b["ann_ret"], dd=b["max_dd"], cal=b["calmar"],
                                trades=b["trade_count"], d_ann=0.0, d_dd=0.0, d_cal=0.0))
    path = "/home/ecs-user/etf_strategy/audit/out/n_stopless_panic.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
