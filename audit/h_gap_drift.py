"""H 组 · exception_gap 最优档的时间漂移审计 — 2026-08-17

问题: 最近若干年最佳 exception_gap 是否变化 (默认 3.0 是否还站得住)?
方法: 多个时间窗口 × gap 梯度 {1.5..6.0} 全组合跑, 按 total_ret 与 Calmar 找各窗口最优档,
      看最优档是否随时间漂移。输出: audit/out/h_gap_drift.json
"""
import sys, os, json
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np

from harness import load_mats, run_cfg, base_cfg, metrics, save

WINDOWS = [
    ("2016-2021", "2016-07-07", "2021-06-30"),
    ("2021-2026", "2021-07-01", "2026-08-12"),
    ("2016-2019", "2016-07-07", "2019-12-31"),
    ("2019-2023", "2019-01-01", "2023-12-31"),
    ("2021-2025", "2021-01-01", "2025-12-31"),
    ("2022-2026", "2022-01-01", "2026-08-12"),
    ("2023-2026", "2023-01-01", "2026-08-12"),
    ("2024-2026", "2024-01-01", "2026-08-12"),
    ("full 2016-2026", "2016-07-07", "2026-08-12"),
]
GAPS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def main():
    out = {"windows": {}}
    for wname, s, e in WINDOWS:
        mats = load_mats("纳指,黄金,创业".split(","), start=s, end=e)
        rows = []
        for g in GAPS:
            cfg = base_cfg(exception_gap=g)
            actions, _, user_nav = run_cfg(cfg, mats)
            m = metrics(user_nav, len(user_nav))
            rows.append({"gap": g, **m, "actions": len(actions)})
        best_ret = max(rows, key=lambda r: r["total_ret"])
        best_cal = max(rows, key=lambda r: r["calmar"])
        out["windows"][wname] = {
            "rows": rows,
            "best_by_total": {"gap": best_ret["gap"], "total_ret": best_ret["total_ret"],
                              "calmar": best_ret["calmar"], "mdd": best_ret["mdd"]},
            "best_by_calmar": {"gap": best_cal["gap"], "total_ret": best_cal["total_ret"],
                               "calmar": best_cal["calmar"], "mdd": best_cal["mdd"]},
            "at_gap3": next(r for r in rows if r["gap"] == 3.0),
        }
        print(wname, "| best total @gap", best_ret["gap"],
              "| best calmar @gap", best_cal["gap"])
    save(out, "h_gap_drift")
    # 汇总: 各窗口 gap3 的排名
    print("\ngap=3.0 在各窗口的排名 (按 total_ret, 1=最优):")
    for wname, w in out["windows"].items():
        ranked = sorted(w["rows"], key=lambda r: -r["total_ret"])
        rank = [i for i, r in enumerate(ranked) if r["gap"] == 3.0][0] + 1
        print(" ", wname, "rank", rank, "/", len(ranked))


if __name__ == "__main__":
    main()
