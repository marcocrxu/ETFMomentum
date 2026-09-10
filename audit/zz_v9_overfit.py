"""L-11: v9 过拟合审查 (修复版: 父进程加载数据, fork 继承, 子进程零 fetch)
1) 作者窗口 (2014~数据末) 活跃默认参数敏感性扫描
2) 双窗拟合: A组(恐慌/急跌/熊市门槛)@2015-2020, B组(排名/轮动/体制)@2020-2025
3) 整合后外推 2014-2016 / 全窗 / 我方主样本窗, 对比 v9 默认
"""
import sys, json, multiprocessing as mp

V9 = "/home/ecs-user/etf_strategy/backtest/v9_official"
sys.path.insert(0, V9)

# 父进程单次加载 → fork 继承
from market_data import UNIVERSE, fetch_history
H = {c: fetch_history(c) for c in UNIVERSE}
CAL = [x[0] for x in H["510300"]]

def _run(task):
    from backtest import backtest, yearly
    label, ws, we, kw = task
    r = backtest(H, CAL, start=ws, end=we, **kw)
    if not r["daily"]:
        return {"label": label, "error": "empty"}
    yrs = {y: round(x * 100, 1) for y, x in yearly(r["daily"])}
    return {"label": label, "ws": ws, "we": we,
            "ret": round((r["nav"] - 1) * 100, 2), "ann": round(r["ann"] * 100, 2),
            "dd": round(r["max_dd"] * 100, 2), "calmar": round(r["calmar"], 2),
            "sharpe": round(r["sharpe"], 2), "sw": r["switches"],
            "crash": len(r["crash_buys"]), "yearly": yrs}

def run_all(tasks, procs=4):
    pool = mp.Pool(procs)          # fork: 子进程继承 H/CAL
    try:
        return pool.map(_run, tasks)
    finally:
        pool.close(); pool.join()

def _cast(name, v):
    if name in ("mom_main", "crash_lock", "ma_bull"):
        return int(float(v))
    if name in ("crash_mom5", "crash_below_ma", "bear_enter_mom", "buffer", "overheat", "panic_drop"):
        return float(v)
    return v == "True"

if __name__ == "__main__":
    WS, WE = "2014-01-01", "9999"
    WA, WB = "2015-01-01", "2020-12-31"
    WC, WD = "2020-01-01", "2025-12-31"
    out = {}

    print("== 0) 默认复验 (应与 14925/-27.0/1.81 一致) ==")
    for r in run_all([("default", WS, WE, {})], procs=1):
        print("  default: ret=%9.2f dd=%7.2f calmar=%5.2f sw=%d" % (r["ret"], r["dd"], r["calmar"], r["sw"]))

    print("\n== 1) 敏感性扫描 (作者窗口) ==")
    sweep = []
    def S(name, vals):
        for v in vals:
            sweep.append(("%s=%s" % (name, v), WS, WE, {name: v}))
    S("mom_main", [10, 15, 20, 25, 30, 40])
    S("buffer", [0.0, 0.01, 0.02, 0.03, 0.04])
    S("overheat", [0.30, 0.40, 0.50, 9.9])
    S("panic_drop", [0.0, 0.03, 0.04, 0.05, 0.06])
    S("bear_enter_mom", [0.0, 0.05, 0.07, 0.10])
    S("crash_mom5", [0.0, -0.06, -0.08, -0.10, -0.12])
    S("crash_below_ma", [0.10, 0.15, 0.20, 0.25])
    S("crash_lock", [3, 5, 7, 10])
    S("score_wls", [True, False])
    S("never_empty", [True, False])
    S("ma_bull", [200, 250])
    for r in run_all(sweep):
        out[r["label"]] = r
        print("  %-18s ret=%9.2f ann=%6.2f dd=%7.2f calmar=%5.2f sharpe=%4.2f sw=%3d crash=%d" %
              (r["label"], r["ret"], r["ann"], r["dd"], r["calmar"], r["sharpe"], r["sw"], r["crash"]))

    print("\n== 2) 双窗拟合 (贪心, 按 Calmar) ==")
    GROUP_A = [("crash_mom5", [-0.06, -0.08, -0.10, -0.12, 0.0]),
               ("crash_below_ma", [0.10, 0.15, 0.20, 0.25]),
               ("crash_lock", [3, 5, 7, 10]),
               ("panic_drop", [0.0, 0.03, 0.04, 0.05, 0.06]),
               ("bear_enter_mom", [0.0, 0.05, 0.07, 0.10])]
    GROUP_B = [("mom_main", [10, 15, 20, 25, 30, 40]),
               ("buffer", [0.0, 0.01, 0.02, 0.03, 0.04]),
               ("overheat", [0.30, 0.40, 0.50, 9.9]),
               ("score_wls", [True, False]),
               ("never_empty", [True, False])]

    def fit(group, ws, we, fixed, tag):
        chosen = {}
        print("  --- %s 拟合 %s ~ %s ---" % (tag, ws, we))
        for name, vals in group:
            tasks = []
            for v in vals:
                p = dict(fixed); p.update(chosen); p[name] = v
                tasks.append(("%s=%s" % (name, v), ws, we, p))
            rs = run_all(tasks)
            bestc = max(r["calmar"] for r in rs)
            for r in rs:
                v = r["label"].split("=", 1)[1]
                mark = ""
                if r["calmar"] == bestc:
                    chosen[name] = _cast(name, v); mark = " <-- best"
                print("    %-16s=%-6s ret=%9.2f dd=%7.2f calmar=%5.2f%s" %
                      (name, v, r["ret"], r["dd"], r["calmar"], mark))
        return chosen

    fa = fit(GROUP_A, WA, WB, {}, "A组(恐慌/急跌/熊市门槛)")
    fb = fit(GROUP_B, WC, WD, fa, "B组(排名/轮动/体制)")
    merged = dict(fa); merged.update(fb)
    print("\n整合参数:", json.dumps(merged, ensure_ascii=False))

    print("\n== 3) 外推检验 ==")
    ev = []
    for ws, we, tag in (("2014-01-01", "2016-12-31", "2014-2016外推"),
                        ("2014-01-01", "9999", "全窗"),
                        ("2016-07-07", "2026-08-20", "我方主样本窗")):
        ev.append(("v9默认:" + tag, ws, we, {}))
        ev.append(("v9整合:" + tag, ws, we, merged))
    for r in run_all(ev):
        out["eval|" + r["label"]] = r
        print("  %-20s %s~%s: ret=%9.2f ann=%6.2f dd=%7.2f calmar=%5.2f sharpe=%4.2f sw=%3d" %
              (r["label"], r["ws"], "end" if r["we"] == "9999" else r["we"],
               r["ret"], r["ann"], r["dd"], r["calmar"], r["sharpe"], r["sw"]))

    out["merged"] = merged
    with open("/home/ecs-user/etf_strategy/audit/out/v9_overfit.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved /home/ecs-user/etf_strategy/audit/out/v9_overfit.json")
