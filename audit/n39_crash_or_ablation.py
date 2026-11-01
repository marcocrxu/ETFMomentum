# -*- coding: utf-8 -*-
"""N-39 恐慌抄底 × 5 日跌幅 OR 组合 消融（协议见 audit/n39_crash_or_protocol.md，跑前已注册）

与 N-38（AND 收紧）相反方向的假设：5 日跌幅作为**并列触发通道**，任一成立即抄底。
  O1 or_deep : 原条件 OR (MOM5 <= X%  AND  收盘<MA200),  X ∈ {-6,-8,-10,-12}
  O2 or_ma   : 原条件 OR (MOM5 <= X%),                X ∈ {-6,-8,-10,-12}   (最宽松)
  O3 or_z    : 原条件 OR (MOM5的z <= Z AND 收盘<MA200), Z ∈ {-1.5,-2.0,-2.5}

除三主指标外，必须报「抄底次数比」与「新增笔平均盈亏」，防止放宽后退化成追高器。

用法: cd /home/ecs-user/etf_strategy && .venv/bin/python audit/n39_crash_or_ablation.py
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
OUT = os.path.join(HERE, "audit", "out", "n39_crash_or.json")


def crash_ledger(decs):
    """抄底逐笔: 入场日 → 下一动作日(出场)其 pnl（与 N-30 抄底账本同口径）"""
    out = []
    for i, d in enumerate(decs):
        if d.get("reason") == "恐慌抄底":
            nxt = decs[i + 1] if i + 1 < len(decs) else None
            out.append({"entry": d["date"], "exit": nxt["date"] if nxt else None,
                        "pnl": nxt.get("pnl") if nxt else None})
    return out


def run(kw, start):
    r = br.run_backtest(dict(universe=UNIVERSE, start_date=start, end_date=END, **kw))
    if "error" in r:
        raise RuntimeError("%s %s" % (start, r["error"]))
    led = crash_ledger(r.get("decisions", []))
    pnls = [x["pnl"] for x in led if x["pnl"] is not None]
    return {"ann": float(r["ann_ret"]), "dd": float(r["max_dd"]), "cal": float(r["calmar"]),
            "total": float(r["total_ret"]), "sharpe": float(r["sharpe"]),
            "trades": int(r["trade_count"]),
            "ledger": led, "crash_pnls": pnls,
            "win": (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else None,
            "avg": (sum(pnls) / len(pnls)) if pnls else None}


def build_grid():
    g = [("基线(只OR前原规则)", {})]
    for x in (-6.0, -8.0, -10.0, -12.0):
        g.append(("O1深度 MOM5<=%g%%" % x, dict(crash_mom5_mode="or_deep", crash_mom5_pct=x)))
    for x in (-6.0, -8.0, -10.0, -12.0):
        g.append(("O2只看跌幅%g%%" % x, dict(crash_mom5_mode="or_ma", crash_mom5_pct=x)))
    for z in (-1.5, -2.0, -2.5):
        g.append(("O3 z化 z<=%g" % z, dict(crash_mom5_mode="or_z", crash_mom5_z=z)))
    return g


def main():
    t0 = time.time()
    print("== N-39 抄底 OR 组合 | 主窗 %s→%s ==" % (MAIN_START, END), flush=True)
    res = {}
    for name, kw in build_grid():
        per = {}
        for s in STARTS:
            per[s] = run(kw, s)
        res[name] = {"kw": kw, "starts": per}
        b = res["基线(只OR前原规则)"]["starts"]
        cells = []
        for s in STARTS:
            r, bs = per[s], b[s]
            cells.append("%s %5.2f(%+5.2f)%6.2f%5.2f/%d" % (
                "主" if s == MAIN_START else s[2:4],
                r["ann"], r["ann"] - bs["ann"], r["dd"], r["cal"], len(r["ledger"])))
        print("%-20s | %s (%.0fs)" % (name, " | ".join(cells), time.time() - t0), flush=True)
    print("  列: 起点 年化(Δ) 回撤 Calmar /抄底次数", flush=True)

    print("\n== 判据 ==", flush=True)
    base = res["基线(只OR前原规则)"]["starts"]
    bm = base[MAIN_START]
    b_ent = {x["entry"] for x in bm["ledger"]}
    summary = []
    for name, d in res.items():
        if name.startswith("基线"):
            continue
        per = d["starts"]
        m = per[MAIN_START]
        d_ann = m["ann"] - bm["ann"]
        d_dd = m["dd"] - bm["dd"]
        d_cal = m["cal"] - bm["cal"]
        n_c, n_b = len(m["ledger"]), len(bm["ledger"])
        # 判据2: 次数不超基线 ×2
        g2 = n_c <= 2 * n_b
        # 判据3: 三主指标全部不劣
        g3 = (d_ann >= 0) and (d_dd >= 0) and (d_cal >= 0)
        # 判据4: 压力窗不劣于各自基线
        g4 = all((per[s]["ann"] - base[s]["ann"] >= 0) and (per[s]["dd"] - base[s]["dd"] >= 0)
                 for s in STRESS_STARTS)
        new = [x for x in m["ledger"] if x["entry"] not in b_ent]
        new_p = [x["pnl"] for x in new if x["pnl"] is not None]
        summary.append(dict(
            name=name, kw=d["kw"], d_ann=round(d_ann, 2), d_dd=round(d_dd, 2), d_cal=round(d_cal, 2),
            crash=n_c, crash_base=n_b, ratio=round(n_c / n_b, 2) if n_b else None,
            g2=g2, g3=g3, g4=g4, ann=m["ann"], dd=m["dd"], cal=m["cal"],
            win=round(m["win"], 3) if m["win"] is not None else None,
            avg=round(m["avg"], 2) if m["avg"] is not None else None,
            n_new=len(new),
            new_avg=round(sum(new_p) / len(new_p), 2) if new_p else None,
            new_win=round(sum(1 for p in new_p if p > 0) / len(new_p), 3) if new_p else None,
            added=[x["entry"] for x in new]))
        print("%-20s Δ年化%+6.2f Δ回撤%+6.2f ΔCalmar%+6.2f | 抄底 %2d/%2d (×%4.1f) "
              "新增%2d笔 均%6s%% 胜%5s | ②%s ③%s ④%s%s"
              % (name, d_ann, d_dd, d_cal, n_c, n_b, (n_c / n_b if n_b else 0),
                 len(new), summary[-1]["new_avg"], summary[-1]["new_win"],
                 "✓" if g2 else "✗", "✓" if g3 else "✗", "✓" if g4 else "✗",
                 "  → 候选" if (g2 and g3 and g4) else ""), flush=True)

    print("\n== 带内方向 ==", flush=True)
    for grp, key in (("O1深度", "or_deep"), ("O2只看跌幅", "or_ma"), ("O3 z化", "or_z")):
        rows = [x for x in summary if x["kw"].get("crash_mom5_mode") == key]
        if rows:
            print("  %s: Δ年化 %s → %s" % (
                grp, [x["d_ann"] for x in rows],
                "同向" if (all(x["d_ann"] >= 0 for x in rows) or all(x["d_ann"] < 0 for x in rows)) else "**异向=噪声**"),
                flush=True)

    json.dump({"starts": STARTS, "main_start": MAIN_START, "end": END, "result": res,
               "summary": summary}, open(OUT, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s  (总耗时 %.0fs)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
