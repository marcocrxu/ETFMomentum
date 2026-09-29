# -*- coding: utf-8 -*-
"""N 组: 同行 V12-R2 对照 —— 我方基线同窗重跑 + 对方冻结逐日路径读窗口指标

统一口径(与对方 v12_r2 一致): 244 观察日/年 年化, 日收益序列, 最大回撤按净值峰谷。

用法:
  .venv/bin/python audit/n_peer_window.py ours   # 只跑我方基线
  .venv/bin/python audit/n_peer_window.py peer   # 只读对方冻结 paths.npz
  .venv/bin/python audit/n_peer_window.py all
"""
import json
import os
import sys
from bisect import bisect_left, bisect_right

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PEER = "/home/ecs-user/peer_artifacts/v12_r2"
SESSIONS = 244.0

WINDOWS = [
    ("主样本窗_原", "2016-07-07", "2026-08-20"),
    ("主样本窗_对齐对方截止", "2016-07-07", "2026-09-24"),
    ("近三年窗", "2021-08-23", "2026-09-24"),
    ("对方全窗", "2014-01-02", "2026-09-24"),
]


def metrics(returns, sessions=SESSIONS):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    nav = np.cumprod(1.0 + r)
    years = r.size / sessions
    cagr = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 and nav[-1] > 0 else float("nan")
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1.0).min())
    sd = float(r.std(ddof=1)) if r.size > 1 else 0.0
    sharpe = float(r.mean() / sd * (sessions ** 0.5)) if sd > 0 else float("nan")
    return dict(days=int(r.size), total=float(nav[-1] - 1.0), cagr=cagr, max_dd=dd,
                sharpe=sharpe, calmar=(cagr / abs(dd) if dd < 0 else float("nan")))


def peer_metrics():
    """读对方 v12_r2/results/audit/reference_paths.npz (独立复算, receipt 逐值 exact)。

    该文件含 7 个角色 × 4 情景 × 3097 日的逐日收益, 与冻结 paths.npz 逐位一致
    (results/audit/receipt.json: max_error 全 0)。
    """
    meta = json.load(open(os.path.join(PEER, "results", "path_metadata.json")))
    reg = json.load(open(os.path.join(PEER, "registered_candidates.json")))
    sel = json.load(open(os.path.join(PEER, "results", "selection.json")))
    dates = np.array(meta["dates"])
    ctrl = reg["controls"]
    roles = {
        "C4(R2基座)": ctrl["c4"],
        "V12-C 对照": ctrl["c"],
        "V10-H 对照": ctrl["h"],
        "V9.2+(simple)": ctrl["simple"],
        "V12-R2(部署主推送=highest_cagr)": sel["exploratory"]["highest_cagr"],
    }
    sw = {}
    for line in open(os.path.join(PEER, "results", "comparison.csv")).read().splitlines()[1:]:
        role, cid, scen, cagr, dd, r21, r26, switches = line.split(",")
        sw[(cid, scen)] = int(switches)
    out = {}
    with np.load(os.path.join(PEER, "results", "ref_paths.npz"), allow_pickle=False) as z:
        for scen in ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp"):
            for label, cid in roles.items():
                rets = z["%s__%s__returns" % (cid, scen)]
                for wlabel, start, end in WINDOWS:
                    lo, hi = bisect_left(dates, start), bisect_right(dates, end)
                    m = metrics(rets[lo:hi])
                    if m is None:
                        continue
                    m["switches_full_period"] = sw.get((cid, scen))
                    out.setdefault(label, {}).setdefault(wlabel, {})[scen] = m
    return dict(candidate_ids=roles, windows=[w[0] for w in WINDOWS], results=out,
                peer_period=reg["period"], peer_grid=[reg["unique_count"], reg["selectable_count"]],
                qualified=sel["qualified_counts"],
                source="v12_r2/results/audit/reference_paths.npz (independent recompute, exact)")


def ours():
    from web.bt_runner import run_backtest
    out = {}
    for label, start, end in WINDOWS[:3]:
        res = run_backtest(dict(universe="纳指,黄金,创业", start_date=start, end_date=end))
        nav = np.array(res["nav_log"], dtype=float)
        rets = nav[1:] / nav[:-1] - 1.0
        m = metrics(rets)
        m.update(trade_count=res["trade_count"], data_range=res["data_range"],
                 checks=res["checks"],
                 engine_ann_252=res["ann_ret"], engine_dd=res["max_dd"],
                 engine_calmar=res["calmar"], engine_sharpe=res["sharpe"])
        out[label] = m
    return out


def sens_metrics():
    """v9/v9.1/v9.2 在 v12 修正数据上的逐日路径 (v12/results/sensitivity/paths.npz.gz)。"""
    import gzip
    import io
    base = "/home/ecs-user/peer_artifacts/v12_sens"
    meta = json.load(open(os.path.join(base, "path_metadata.json")))
    reg = json.load(open(os.path.join(base, "registered_candidates.json")))
    want = {"v9(我们存档的同版本)": "v12_10b7b1a6687fd049aebe",
            "v9.1": "v12_d1d9ea928d1692b16fb2", "v9.2": "v12_24552ec3549d227cc2e1"}
    ids, dates = meta["ids"], np.array(meta["dates"])
    role2id = {}
    for role, src in want.items():
        for rec in reg["records"]:
            if rec.get("source_id") == src or src in (rec.get("source_ids") or []):
                role2id[role] = rec["id"]
    with np.load(io.BytesIO(gzip.decompress(open(os.path.join(base, "paths.npz.gz"), "rb").read())),
                 allow_pickle=False) as z:
        out = {}
        for scen in ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp"):
            rets = z[scen + "__returns"]
            for role, cid in role2id.items():
                i = ids.index(cid)
                for wlabel, start, end in WINDOWS:
                    lo, hi = bisect_left(dates, start), bisect_right(dates, end)
                    m = metrics(rets[i, lo:hi])
                    if m:
                        out.setdefault(role, {}).setdefault(wlabel, {})[scen] = m
    return dict(candidate_ids=role2id, windows=[w[0] for w in WINDOWS], results=out,
                source="v12/results/sensitivity/paths.npz.gz (corrected total-return data)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if mode in ("peer", "all"):
        res["peer"] = peer_metrics()
    if mode in ("sens", "all"):
        res["sens"] = sens_metrics()
    if mode in ("ours", "all"):
        res["ours"] = ours()
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
