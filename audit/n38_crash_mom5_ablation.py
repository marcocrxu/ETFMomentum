# -*- coding: utf-8 -*-
"""N-38 恐慌抄底 × 5 日跌幅接入 消融（协议见 audit/n38_crash_mom5_protocol.md，跑前已注册）

三组互斥接入（见协议）:
  A add  : 原条件 AND (指数 MOM5 <= X%), X ∈ {-6,-8,-10,-12}
  B repl : (指数 MOM5 <= X%) AND 收盘<MA200, X ∈ {-6,-8,-10,-12}
  C zs   : (MOM5 的 z 值 <= Z) AND 收盘<MA200, Z ∈ {-1.5,-2.0,-2.5}

主窗 2016-07-07→2026-10-29 + 压力窗 2016-01-01 / 2018-01-01
逐窗自比基线；判定同时看 年化/回撤/Calmar 与 抄底触发次数，不只看年化。

用法: cd /home/ecs-user/etf_strategy && .venv/bin/python audit/n38_crash_mom5_ablation.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import web.bt_runner as br  # noqa: E402

UNIVERSE = "纳指,黄金,创业"
MAIN_START = "2016-07-07"
STRESS_STARTS = ["2016-01-01", "2018-01-01"]
STARTS = [MAIN_START] + STRESS_STARTS
END = "2026-10-29"
OUT = os.path.join(HERE, "audit", "out", "n38_crash_mom5.json")


def run(kw, start):
    r = br.run_backtest(dict(universe=UNIVERSE, start_date=start, end_date=END, **kw))
    if "error" in r:
        raise RuntimeError("%s %s" % (start, r["error"]))
    decs = r.get("decisions", [])
    crash_days = [d["date"] for d in decs if d.get("reason") == "恐慌抄底"]
    return {"ann": float(r["ann_ret"]), "dd": float(r["max_dd"]), "cal": float(r["calmar"]),
            "total": float(r["total_ret"]), "sharpe": float(r["sharpe"]),
            "trades": int(r["trade_count"]), "crash_days": crash_days}


def build_grid():
    g = [("基线(无5日门)", {})]
    for x in (-6.0, -8.0, -10.0, -12.0):
        g.append(("A 增量 MOM5<=%g%%" % x, dict(crash_mom5_mode="add", crash_mom5_pct=x)))
    for x in (-6.0, -8.0, -10.0, -12.0):
        g.append(("B 替代 MOM5<=%g%%" % x, dict(crash_mom5_mode="repl", crash_mom5_pct=x)))
    for z in (-1.5, -2.0, -2.5):
        g.append(("C z化 z<=%g" % z, dict(crash_mom5_mode="zs", crash_mom5_z=z)))
    return g


def main():
    t0 = time.time()
    print("== N-38 抄底 5 日跌幅接入 | 主窗 %s→%s ==" % (MAIN_START, END), flush=True)
    res = {}
    for name, kw in build_grid():
        per = {}
        for s in STARTS:
            per[s] = run(kw, s)
        res[name] = {"kw": kw, "starts": per}
        b = res["基线(无5日门)"]["starts"]
        cells = []
        for s in STARTS:
            r, bs = per[s], b[s]
            cells.append("%s %5.2f(%+5.2f)%6.2f%5.2f/%d" % (
                "主" if s == MAIN_START else s[2:4],
                r["ann"], r["ann"] - bs["ann"], r["dd"], r["cal"], len(r["crash_days"])))
        print("%-20s | %s (%.0fs)" % (name, " | ".join(cells), time.time() - t0), flush=True)
    print("  列含义: 起点 年化(相对基线差) 回撤 Calmar /抄底次数", flush=True)

    # ── 协议判据 ──
    print("\n== 判据 ==", flush=True)
    base = res["基线(无5日门)"]["starts"]
    summary = []
    for name, d in res.items():
        if name == "基线(无5日门)":
            continue
        per = d["starts"]
        m = per[MAIN_START]; bm = base[MAIN_START]
        d_ann = m["ann"] - bm["ann"]
        d_dd = m["dd"] - bm["dd"]           # >0 表示回撤更浅(更好)
        d_cal = m["cal"] - bm["cal"]
        n_c, n_b = len(m["crash_days"]), len(bm["crash_days"])
        # 判据 1: 三主指标不劣 (年化不低, 回撤不深, Calmar 不低)
        g1 = (d_ann >= 0) and (d_dd >= 0) and (d_cal >= 0)
        # 判据 2: 抄底次数不塌缩
        g2 = n_c >= 0.5 * n_b
        # 判据 3: 两个压力窗也不劣于各自基线
        strs = []
        for s in STRESS_STARTS:
            rr, bb = per[s], base[s]
            strs.append((rr["ann"] - bb["ann"] >= 0) and (rr["dd"] - bb["dd"] >= 0))
        g3 = all(strs)
        summary.append(dict(name=name, kw=d["kw"], d_ann=round(d_ann, 2), d_dd=round(d_dd, 2),
                            d_cal=round(d_cal, 2), crash=n_c, crash_base=n_b,
                            g1=g1, g2=g2, g3=g3, stress=strs,
                            ann=m["ann"], dd=m["dd"], cal=m["cal"],
                            deleted=[x for x in bm["crash_days"] if x not in m["crash_days"]],
                            added=[x for x in m["crash_days"] if x not in bm["crash_days"]]))
        print("%-20s Δ年化%+6.2f Δ回撤%+6.2f ΔCalmar%+6.2f 抄底 %2d/%2d | ①%s ②%s ③%s%s"
              % (name, d_ann, d_dd, d_cal, n_c, n_b,
                 "✓" if g1 else "✗", "✓" if g2 else "✗", "✓" if g3 else "✗",
                 "  → 候选" if (g1 and g2 and g3) else ""), flush=True)

    # 整条带是否单调同向（防单点噪声）
    print("\n== 带内方向检查（协议判据0: 整条带同向）==", flush=True)
    for grp, key in (("A 增量", "add"), ("B 替代", "repl"), ("C z化", "zs")):
        rows = [x for x in summary if x["kw"].get("crash_mom5_mode") == key]
        if not rows:
            continue
        signs = [x["d_ann"] >= 0 for x in rows]
        same = all(signs) or (not any(signs))
        print("  %s: Δ年化序列 %s → %s" % (
            grp, [x["d_ann"] for x in rows], "方向一致" if same else "**方向不一致=噪声**"), flush=True)

    json.dump({"starts": STARTS, "main_start": MAIN_START, "end": END, "result": res,
               "summary": summary}, open(OUT, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s  (总耗时 %.0fs)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
