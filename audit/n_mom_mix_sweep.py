# -*- coding: utf-8 -*-
"""N-24: 多周期动量加权全网格 —— 阶段 1 筛选 (1771 格 × 主窗)

严格执行 audit/n_mom_mix_protocol.md。step=5%, w5+w10+w15+w20=1.00。
输出全网格 CSV + 前 20 名清单, 供阶段 2 验证。

用法: .venv/bin/python audit/n_mom_mix_sweep.py
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import web.bt_runner as br  # noqa: E402


class _NoOp:
    """吞掉 matplotlib 调用 —— 只跳过绘图, 指标/决策完全不受影响 (N-24 批量消融用)。"""
    def __call__(self, *a, **k): return self
    def __getattr__(self, n): return self
    def __setitem__(self, k, v): pass
    def __setitem__(self, k, v): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


br.plt = _NoOp()          # 1.9x 加速; 已核对指标逐位一致

# 行情缓存: get_dataframe 每次 0.57~0.72s, 是主要瓶颈 (整个回测才 ~1s)。
# 同进程内窗口/代码固定, 结果只读复用 -> 再约 2x。仅本脚本使用, 不影响引擎。
_orig_gdf = br.FEED.get_dataframe
_gdf_cache = {}


def _cached_gdf(codes, start, end, *a, **k):
    key = (tuple(codes), start, end)
    if key not in _gdf_cache:
        _gdf_cache[key] = _orig_gdf(codes, start, end, *a, **k)
    return _gdf_cache[key]


br.FEED.get_dataframe = _cached_gdf

START = sys.argv[1] if len(sys.argv) > 1 else "2016-07-07"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-08-20"
TAG = sys.argv[3] if len(sys.argv) > 3 else "A"
BASE = (45.51, -18.08, 2.52, 253)
STEPS = 20            # 以 5% 为单位, 总和 20


def grid():
    for a in range(STEPS + 1):
        for b in range(STEPS + 1 - a):
            for c in range(STEPS + 1 - a - b):
                d = STEPS - a - b - c
                yield a / 20, b / 20, c / 20, d / 20


def spec(w5, w10, w15, w20):
    return "5:%.2f,10:%.2f,15:%.2f,20:%.2f" % (w5, w10, w15, w20)


def main():
    rows = []
    t0 = time.time()
    n = 0
    for w5, w10, w15, w20 in grid():
        kw = {} if abs(w20 - 1.0) < 1e-9 else dict(score_mom_mix=spec(w5, w10, w15, w20))
        r = br.run_backtest(dict(universe="纳指,黄金,创业", start_date=START, end_date=END, **kw))
        rows.append(dict(w5=w5, w10=w10, w15=w15, w20=w20, ann=r["ann_ret"], dd=r["max_dd"],
                         cal=r["calmar"], sharpe=r.get("sharpe"), trades=r["trade_count"],
                         total=r["total_ret"]))
        n += 1
        if n % 100 == 0:
            el = time.time() - t0
            print("  %4d/%d  %.0fs (剩 ~%.0fs)" % (n, 1771, el, el / n * (1771 - n)), flush=True)
    df = pd.DataFrame(rows)
    p = os.path.join(HERE, "audit", "out", "n_mom_mix_grid_%s.csv" % TAG)
    df.to_csv(p, index=False)
    print("\n[%s] 全网格完成 %d 格 (%s~%s), 用时 %.0fs" % (TAG, len(df), START, END, time.time() - t0))
    print("基线(20:1.0) ann=%.2f dd=%.2f cal=%.2f trades=%d"
          % (BASE[0], BASE[1], BASE[2], BASE[3]))
    print("\n=== 全网格分布 (年化%) ===")
    print(df.ann.describe(percentiles=[.05, .25, .5, .75, .95]).to_string())
    print("\n超过基线的格数: %d / %d (%.1f%%)"
          % ((df.ann > BASE[0]).sum(), len(df), 100 * (df.ann > BASE[0]).mean()))
    print("年化为正的格数: %d (%.1f%%)" % ((df.ann > 0).sum(), 100 * (df.ann > 0).mean()))
    print("\n=== 前 20 名 ===")
    top = df.sort_values("ann", ascending=False).head(20)
    print(top.to_string(index=False, float_format=lambda v: "%.4f" % v))
    # 权重形状: 前 20 名的平均权重 vs 全网格平均
    print("\n前 20 名平均权重: w5=%.3f w10=%.3f w15=%.3f w20=%.3f"
          % (top.w5.mean(), top.w10.mean(), top.w15.mean(), top.w20.mean()))
    print("全网格平均权重:   w5=%.3f w10=%.3f w15=%.3f w20=%.3f"
          % (df.w5.mean(), df.w10.mean(), df.w15.mean(), df.w20.mean()))
    print("\n=== 只看纯 mom20 (w5=w10=w15=0) 那一格 ===")
    print(df[(df.w5 == 0) & (df.w10 == 0) & (df.w15 == 0)].to_string(index=False))
    json.dump({"n": len(df), "base": BASE,
               "top20": top.to_dict("records"),
               "over_base": int((df.ann > BASE[0]).sum()),
               "best": df.loc[df.ann.idxmax()].to_dict()},
              open(os.path.join(HERE, "audit", "out", "n_mom_mix_sweep_%s.json" % TAG), "w"),
              ensure_ascii=False, indent=1, default=float)
    print("\nsaved -> %s" % p)


if __name__ == "__main__":
    main()
