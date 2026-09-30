# -*- coding: utf-8 -*-
"""N-16: 同行到底"做对了什么"? —— 把她的 5.585 拆成"池子"和"选择"

N-15 已知: 扩池亏钱是我方排名器的两个失效模式(追高 + 误杀纳指)。
那她为什么没被同样的问题打中? 本脚本用**她自己那 11 只标的**构造几个"笨基准",
看她的 5.585 里有多少只是"池子好", 有多少才是"会轮动"。

基准 (2014-01-02~2026-09-24, 用我方缓存的乘法式复权价, 与她的标的相同):
  B1 等权日再平衡: 池内当日**有数据**的标的等权
  B2 等权日再平衡: 只含我们共同的 3 只(纳指/黄金/创业)
  B3 事后最优单只: 全期涨最多的那一只(不可实现, 上界参照)
  B4 每只标的"持有期日均 vs 无条件日均" —— 衡量她/我们的择时效率

用法: .venv/bin/python audit/n_peer_why_good.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import web.bt_runner as br  # noqa: E402

START, END = "2014-01-02", "2026-09-24"
CACHE = "/home/ecs-user/backtest/cache/etf_momentum"
PEER = "/home/ecs-user/peer_artifacts/v12_r2"

POOL11 = ["159915", "588080", "510300", "510500", "563300", "512400",
          "512890", "513100", "513120", "518880", "511880"]
NAME = {"159915": "创业板", "588080": "科创50", "510300": "沪深300", "510500": "中证500",
        "563300": "中证2000", "512400": "有色", "512890": "红利低波", "513100": "纳指",
        "513120": "港股创新药", "518880": "黄金", "511880": "货币"}
SHARE3 = ["513100", "518880", "159915"]


def load(code):
    df = pd.read_csv(os.path.join(CACHE, code.replace(".", "_") + ".csv")
                     if "." in code else os.path.join(CACHE, code + "_SH.csv" if code not in ("159915",) else code + "_SZ.csv"),
                     parse_dates=["trade_date"])
    s = pd.Series((df.close * df.adj_factor).values, index=df.trade_date)
    return s[(s.index >= START) & (s.index <= END)]


def load_all():
    out = {}
    for c in POOL11:
        suf = "_SZ" if c in ("159915",) else "_SH"
        df = pd.read_csv(os.path.join(CACHE, c + suf + ".csv"), parse_dates=["trade_date"])
        s = pd.Series((df.close * df.adj_factor).values, index=df.trade_date)
        out[c] = s[(s.index >= START) & (s.index <= END)]
    return out


def main():
    px = load_all()
    # 共同日期轴 = 沪深300 的交易日 (她的日历基准)
    cal = [d.strftime("%Y-%m-%d") for d in px["510300"].index]
    M = pd.DataFrame({c: px[c].reindex(px["510300"].index).ffill() for c in POOL11})
    R = M.pct_change()

    print("=== 各标的可得区间与买入持有 (2014-01-02 ~ %s) ===" % END)
    print("  %-10s %-12s %-12s %10s %12s" % ("标的", "首日", "末日", "买入持有对数", "日均bp"))
    bh = {}
    for c in POOL11:
        s = px[c]
        lg = float(np.log(s.iloc[-1] / s.iloc[0]))
        days = len(s)
        bh[c] = dict(first=s.index[0].strftime("%Y-%m-%d"), last=s.index[-1].strftime("%Y-%m-%d"),
                     log=lg, days=days, bp_per_day=lg / days * 1e4)
        print("  %-10s %-12s %-12s %+11.3f %11.1f" % (NAME[c], bh[c]["first"], bh[c]["last"],
                                                      lg, bh[c]["bp_per_day"]))

    # B1 等权日再平衡 (只含当日有数据的)
    avail = M.notna()
    w = avail.div(avail.sum(axis=1), axis=0)
    b1 = float(np.log1p((w.shift(1).fillna(0) * R.fillna(0)).sum(axis=1)).sum())
    # B2 共同 3 只等权
    M3 = M[SHARE3]; R3 = M3.pct_change(); a3 = M3.notna()
    w3 = a3.div(a3.sum(axis=1), axis=0)
    b2 = float(np.log1p((w3.shift(1).fillna(0) * R3.fillna(0)).sum(axis=1)).sum())
    # B3 事后最优单只 (从它自己的首日起)
    best_c = max(POOL11, key=lambda c: bh[c]["log"])
    b3 = bh[best_c]["log"]

    # 她的实际
    meta = json.load(open(os.path.join(PEER, "results", "path_metadata.json")))
    assets = json.load(open(os.path.join(PEER, "results", "feature_metadata.json")))["assets"]
    with np.load(os.path.join(PEER, "results", "ref_paths.npz"), allow_pickle=False) as z:
        pr = z["v12r2_08b522c2e1aa291f054d__close_1bp__returns"]
        ph = z["v12r2_08b522c2e1aa291f054d__close_1bp__holdings"]
    her = float(np.log1p(pr).sum())
    _nav = np.array(br.run_backtest(dict(universe="纳指,黄金,创业",
                                         start_date=START, end_date=END))["nav_log"], dtype=float)
    ours = float(np.log(_nav[-1] / _nav[0]))

    print("\n=== 笨基准 vs 实际 ===")
    print("  B1 她 11 只等权日再平衡  : %+.3f  (累计 %+.0f%%)" % (b1, (np.exp(b1) - 1) * 100))
    print("  B2 共同 3 只等权日再平衡  : %+.3f  (累计 %+.0f%%)" % (b2, (np.exp(b2) - 1) * 100))
    print("  B3 事后最优单只 (%s)   : %+.3f  (不可实现, 上界参照)" % (NAME[best_c], b3))
    print("  ----")
    print("  我方 窄池3 实际           : %+.3f" % ours)
    print("  她   V12-R2 实际          : %+.3f" % her)
    print("  ----")
    print("  她的轮动超额  = 她 − B1   : %+.3f   (池子之外的本事)" % (her - b1))
    print("  她的池子超额  = B1 − B2   : %+.3f   (换成 11 只等权能白拿多少)" % (b1 - b2))
    print("  我方轮动超额  = 我 − B2   : %+.3f" % (ours - b2))
    print("  两者轮动之差  = %+.3f" % ((her - b1) - (ours - b2)))

    # B4 择时效率: 持有期日均 vs 无条件日均
    print("\n=== B4 择时效率: 持有期日均收益 / 该标无条件日均 (越大=越会在好日子里持有) ===")
    print("  %-10s %14s %14s %10s" % ("标的", "她 持有期bp/天", "无条件bp/天", "倍数"))
    her_timing = {}
    for c in POOL11:
        i = assets.index(c)
        days = int((np.asarray(ph) == i).sum())
        lg = 0.0
        for t in range(1, len(pr)):
            if int(ph[t - 1]) == i and pr[t] > -1:
                lg += float(np.log1p(pr[t]))
        per = lg / days * 1e4 if days else 0.0
        uncond = bh[c]["bp_per_day"]
        mult = per / uncond if uncond > 0 else float("nan")
        her_timing[c] = dict(days=days, log=lg, bp_per_day=per, uncond=uncond, mult=mult)
        print("  %-10s %14.1f %14.1f %9.2fx" % (NAME[c], per, uncond, mult))

    out = dict(baselines=dict(B1_equal11=b1, B2_equal3=b2, B3_best_single=b3, best=best_c),
               ours=ours, hers=her, her_rotation=her - b1, her_pool=b1 - b2,
               our_rotation=ours - b2, rotation_gap=(her - b1) - (ours - b2),
               buyhold=bh, her_timing=her_timing)
    path = os.path.join(HERE, "audit", "out", "n_peer_why_good.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s" % path)


if __name__ == "__main__":
    main()
