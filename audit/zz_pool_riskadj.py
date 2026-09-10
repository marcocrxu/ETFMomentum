"""扩池 + mom20/vol20 风险调整排名 消融 (2026-08-28, 用户提议)
对照: 基线窄池3 / 基线+risk_adj / 无条件扩池+risk_adj / 条件扩池(sh_mom20)+risk_adj / 条件扩池(sh_ma250)+risk_adj
当前默认口径 (QQQ 0.7 开), 三窗口。"""
import sys, json, time
sys.path.insert(0, '/home/ecs-user/etf_strategy')
from web.bt_runner import run_backtest

WINDOWS = [
    ("2016-07-07", "2026-08-20", "默认窗"),
    ("2015-01-05", "2026-08-20", "2015长窗"),
    ("2020-01-01", "2026-08-20", "2020短窗"),
]
BASE = {"universe": "纳指,黄金,创业"}
EXTRA_UNIV = "纳指,黄金,创业,有色,红利低波,科创50,中证2000"
EXTRA = "有色,红利低波,科创50,中证2000"

out = []


def run(tag, **kw):
    p = dict(BASE)
    p.update(kw)
    t0 = time.time()
    r = run_backtest(p)
    if "error" in r:
        print("%-52s ERROR %s" % (tag, r["error"]))
        return
    print("%-52s ann %6.2f%% dd %6.2f%% cal %5.2f sharpe %4.2f total %7.1f%% trades %3d (%.0fs)"
          % (tag, r["ann_ret"], r["max_dd"], r["calmar"], r.get("sharpe", 0),
             r["total_ret"], r["trade_count"], time.time() - t0))
    out.append({"tag": tag, "window": kw.get("_wtag"), "ann": r["ann_ret"], "dd": r["max_dd"],
                "cal": r["calmar"], "sharpe": r.get("sharpe"), "total": r["total_ret"],
                "trades": r["trade_count"], "yearly": r.get("yearly")})


for ws, we, wtag in WINDOWS:
    print("\n===== %s (%s ~ %s) =====" % (wtag, ws, we))
    run("基线 窄池3", start_date=ws, end_date=we, _wtag=wtag)
    run("基线 + risk_adj (窄池上复证否决)", start_date=ws, end_date=we, risk_adj_on=True, _wtag=wtag)
    run("无条件扩池 + risk_adj", start_date=ws, end_date=we, universe=EXTRA_UNIV,
        risk_adj_on=True, _wtag=wtag)
    run("条件扩池(sh_mom20) + risk_adj", start_date=ws, end_date=we, universe=EXTRA_UNIV,
        bull_pool_extra=EXTRA, bull_pool_rule="sh_mom20", risk_adj_on=True, _wtag=wtag)
    run("条件扩池(sh_ma250) + risk_adj", start_date=ws, end_date=we, universe=EXTRA_UNIV,
        bull_pool_extra=EXTRA, bull_pool_rule="sh_ma250", risk_adj_on=True, _wtag=wtag)

json.dump(out, open('/home/ecs-user/etf_strategy/audit/out/pool_riskadj_ablation.json', 'w'),
          ensure_ascii=False, indent=1)
print("\nsaved -> audit/out/pool_riskadj_ablation.json")
