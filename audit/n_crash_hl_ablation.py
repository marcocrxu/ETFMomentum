# -*- coding: utf-8 -*-
"""N-30 恐慌抄底 × 近 N 日高低点结构门 消融（协议见 audit/n_crash_hl_protocol.md，跑前已注册）

面板: 六起点 → 2026-08-20（逐起点自比基线）
网格: 22 格 —— 深度门 / 位置门 / 离底门 × 窗口 60(3个月)/120(6个月)
基线: 生产默认（crash_hl_win=0，结构门关闭）

用法: cd /home/ecs-user/etf_strategy && .venv/bin/python audit/n_crash_hl_ablation.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import web.bt_runner as br  # noqa: E402

UNIVERSE = "纳指,黄金,创业"
STARTS = ["2014-01-02", "2016-01-04", "2018-01-02", "2020-01-02", "2022-01-04", "2024-01-02"]
END = "2026-08-20"
OUT = os.path.join(HERE, "audit", "out", "n_crash_hl_ablation.json")


def crash_ledger(decs):
    """抄底逐笔: 入场日 → 下一动作日(出场)与其 pnl（口径同 e_group 的抄底账本）"""
    out = []
    for i, d in enumerate(decs):
        if d.get("reason") == "恐慌抄底":
            nxt = decs[i + 1] if i + 1 < len(decs) else None
            out.append({"entry": d["date"], "action": d.get("action"),
                        "exit": nxt["date"] if nxt else None,
                        "pnl": nxt.get("pnl") if nxt else None,
                        "exit_reason": nxt.get("reason") if nxt else None})
    return out


def run(kw, start):
    r = br.run_backtest(dict(universe=UNIVERSE, start_date=start, end_date=END, **kw))
    if "error" in r:
        raise RuntimeError("%s %s" % (start, r["error"]))
    led = crash_ledger(r["decisions"])
    return {"ann": float(r["ann_ret"]), "dd": float(r["max_dd"]), "cal": float(r["calmar"]),
            "total": float(r["total_ret"]), "trades": int(r["trade_count"]),
            "crash_n": len(led), "crash": led}


def build_grid():
    g = []
    for win in (60, 120):
        for v in (5, 10, 15, 20):
            g.append(("深%d日回撤≥%g%%" % (win, v), dict(crash_hl_win=win, crash_dd_min=float(v))))
        for v in (0.2, 0.3, 0.4, 0.5):
            g.append(("位%d日分位≤%.1f" % (win, v), dict(crash_hl_win=win, crash_pos_max=float(v))))
        for v in (3, 5, 8):
            g.append(("离%d日低点≤%g%%" % (win, v), dict(crash_hl_win=win, crash_off_low_max=float(v))))
    return g


def main():
    t_all = time.time()
    print("== 基线（六起点, 结构门关闭）==", flush=True)
    base = {}
    for s in STARTS:
        base[s] = run({}, s)
        print("  %s ann %6.2f dd %7.2f cal %5.2f trades %3d crash %2d  [%s]"
              % (s, base[s]["ann"], base[s]["dd"], base[s]["cal"], base[s]["trades"],
                 base[s]["crash_n"], ",".join(c["entry"] for c in base[s]["crash"])), flush=True)

    rows = []
    for name, kw in build_grid():
        res, diffs = {}, {}
        for s in STARTS:
            res[s] = run(kw, s)
            diffs[s] = round(res[s]["ann"] - base[s]["ann"], 2)
        wins = sum(1 for s in STARTS if diffs[s] >= 0)
        med = float(sorted(diffs.values())[len(diffs) // 2]) if len(diffs) % 2 else \
            float(sum(sorted(diffs.values())[len(diffs) // 2 - 1:len(diffs) // 2 + 1]) / 2)
        worst = min(diffs.values())
        crash_del = {s: [c["entry"] for c in base[s]["crash"]
                         if c["entry"] not in [x["entry"] for x in res[s]["crash"]]] for s in STARTS}
        n_del = sum(len(v) for v in crash_del.values())
        rows.append({"name": name, "kw": kw, "starts": res, "diffs": diffs,
                     "wins": wins, "median": med, "worst": worst,
                     "crash_deleted": crash_del, "n_deleted": n_del})
        print("%-18s | " % name
              + " ".join("%s %+6.2f" % (s[2:7], diffs[s]) for s in STARTS)
              + " | 胜 %d/6 med %+5.2f worst %+6.2f | 删抄底 %d 笔 (%.0fs)"
              % (wins, med, worst, n_del, time.time() - t_all), flush=True)

    print("\n== 资格判定（协议第 5 节四条）==", flush=True)
    verdict = []
    for r in rows:
        g1 = r["wins"] >= 5
        g2 = r["worst"] >= -3.0
        g3 = r["median"] >= 0.3
        pos = [s for s in STARTS if r["diffs"][s] > 0]
        g4 = not (len(pos) == 1)          # 不是孤立尖峰
        r["gate1"], r["gate2"], r["gate3"], r["gate4"] = g1, g2, g3, g4
        ok = g1 and g2 and g3 and g4
        verdict.append({"name": r["name"], "kw": r["kw"], "pass": ok,
                        "wins": r["wins"], "median": r["median"], "worst": r["worst"],
                        "n_deleted": r["n_deleted"]})
        print("%-18s ①%s ②%s ③%s ④%s  %s"
              % (r["name"], "✓" if g1 else "✗", "✓" if g2 else "✗",
                 "✓" if g3 else "✗", "✓" if g4 else "✗", "→ 候选" if ok else ""), flush=True)

    # 被删掉的抄底笔在基线里赚了多少（判断闸门删好笔还是坏笔）
    print("\n== 各变体删掉的抄底笔, 在基线里的盈亏 ==", flush=True)
    base_pnl = {}
    for s in STARTS:
        base_pnl[s] = {c["entry"]: c["pnl"] for c in base[s]["crash"]}
    for r in rows:
        items = []
        for s in STARTS:
            for d in r["crash_deleted"][s]:
                items.append("%s@%s %s%%" % (s[2:7], d, base_pnl[s].get(d)))
        if items:
            print("%-18s %s" % (r["name"], " | ".join(items)), flush=True)

    json.dump({"starts": STARTS, "end": END, "baseline": base, "rows": rows,
               "verdict": verdict},
              open(OUT, "w"), ensure_ascii=False, indent=1, default=str)
    print("\nsaved -> %s  (总耗时 %.0fs)" % (OUT, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
