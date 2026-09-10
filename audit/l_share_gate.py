"""L 组 · 宽基ETF份额风险门消融 — 2026-09-07

提议: 把 Resonance 的"国家队份额流"引入动量轮动做 filter (不做信号、不做正向豁免)。
  share_gate: 篮子(510300/510500/510050) 20日净申购率 < -X% 时禁止新买/切换入 A 股标的,
              纳指/黄金天然豁免 (QDII/商品份额不含国家队信息), NaN(数据未覆盖)不拦。
数据: 上交所 ETF 规模接口 (data/sharefeed.py), 2016-01-04 起逐日, T+1 shift 无前视。
基线: 当前默认口径 (share_gate=0 关闭)。

先验分布 (20日净申购率, 2016-2026): P10=-7.0% P25=-3.2% P50=+0.1%;
  <-3% 占 26% 天数 -> 阈值候选取 5/7/10, 更松的会过度屏蔽动量参与度。
注意: 2026-01 起三宽基份额同步骤降 ~50% (510300: 895亿->382亿),
  真实赎回 or 披露口径变化无法区分 — 2023-2026 窗口的收益含此不确定性。

输出: audit/out/l_share_gate.json
"""
import sys, json
sys.path.insert(0, "/home/ecs-user/etf_strategy"); sys.path.insert(0, "/home/ecs-user/etf_strategy/audit")
from harness import bt

GATES = [0.0, 5.0, 7.0, 10.0]
WINDOWS = [
    ("全区间", "2016-07-07", "2026-09-04"),
    ("2016-2019", "2016-07-07", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2026-09-04"),
]

out = {}
for wname, s, e in WINDOWS:
    for g in GATES:
        r = bt(start_date=s, end_date=e, share_gate=g)
        key = f"{wname}|gate={g}"
        out[key] = {k: r[k] for k in ("total_ret", "ann_ret", "max_dd", "sharpe", "trade_count")}
        print(key, out[key], flush=True)

with open("/home/ecs-user/etf_strategy/audit/out/l_share_gate.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved -> audit/out/l_share_gate.json")
