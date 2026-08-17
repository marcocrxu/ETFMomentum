"""E8 修正: 恐慌日与引擎同口径(全历史扩张阈值), bootstrap 对比默认窗 8 笔实际入场"""
import sys
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd, json
from harness import FEED, bt

dfs = FEED.get_dataframe(["399006.SZ", "000001.SH"], "20130101", "20260812")
cyb_df = dfs["399006.SZ"].set_index("trade_date")["close"]
sh_df = dfs["000001.SH"].set_index("trade_date")["close"]
all_d = sorted(set(cyb_df.index) | set(sh_df.index))
cyb_p = np.array([cyb_df.get(d, np.nan) for d in all_d])
sh_p = np.array([sh_df.get(d, np.nan) for d in all_d])

def panic_days(px):
    m10 = np.full(len(px), np.nan); m10[10:] = (px[10:] / px[:-10] - 1) * 100
    s = pd.Series(m10)
    th = (s.expanding(min_periods=60).mean() - 2.5 * s.expanding(min_periods=60).std()).shift(1).values
    ma = pd.Series(px).rolling(200).mean().values
    return set(np.where((m10 < th) & (px < ma) & ~np.isnan(ma))[0].tolist())

panic = sorted(panic_days(cyb_p) | panic_days(sh_p))
# 与引擎窗口(2016-07起, 含2014-02预热扩张史)对齐: 用2014-02起扩张重算更贴近
def panic_days_engine(px, start_i):
    m10 = np.full(len(px), np.nan); m10[10:] = (px[10:] / px[:-10] - 1) * 100
    s = pd.Series(m10)
    th = (s.expanding(min_periods=60).mean() - 2.5 * s.expanding(min_periods=60).std()).shift(1).values
    ma = pd.Series(px).rolling(200).mean().values
    hit = np.where((m10 < th) & (px < ma) & ~np.isnan(ma))[0]
    return set(h for h in hit if h >= start_i)

start2016 = next(i for i, d in enumerate(all_d) if str(d)[:10] >= "2016-07-07")
# 引擎口径: 扩张窗起点 = bt_start - 320自然日 = 2015-08-22 (预热起点), 而非2013全历史
eng0 = next(i for i, d in enumerate(all_d) if str(d)[:10] >= "2015-08-22")
def panic_days_engine_v2(px, calib0, dec0):
    m10 = np.full(len(px), np.nan); m10[10:] = (px[10:] / px[:-10] - 1) * 100
    th = np.full(len(px), np.nan)
    for t in range(calib0, len(px)):
        valid = m10[calib0:t][~np.isnan(m10[calib0:t])]
        if len(valid) >= 60:
            th[t] = valid.mean() - 2.5 * valid.std()
    ma = pd.Series(px).rolling(200).mean().values
    hit = np.where((m10 < th) & (px < ma) & ~np.isnan(ma))[0]
    return set(h for h in hit if h >= dec0)

panic_eng = sorted(panic_days_engine_v2(cyb_p, eng0, start2016) | panic_days_engine_v2(sh_p, eng0, start2016))

def fwd20(i): return (cyb_p[i + 20] / cyb_p[i] - 1) * 100 if i + 20 < len(cyb_p) else None

# 恐慌日去重成"事件簇"(相邻<=5天算同一事件), 引擎实际也只会入首次
def cluster(days):
    evs = []; cur = [days[0]]
    for d in days[1:]:
        if d - cur[-1] <= 5: cur.append(d)
        else: evs.append(cur); cur = [d]
    evs.append(cur)
    return [c[0] for c in evs]  # 簇首日(引擎入场日)

panic_events_all = cluster(panic) if panic else []
panic_events_eng = cluster(panic_eng) if panic_eng else []
# 实际 8 笔入场日
base = bt()
entries = [d["date"] for d in base["decisions"] if d["reason"] == "恐慌抄底"]
entry_idx = [next(i for i, d in enumerate(all_d) if str(d)[:10] >= e) for e in entries]

rng = np.random.default_rng(7)
pop = [fwd20(i) for i in panic_events_eng]
pop = [x for x in pop if x is not None]
actual = [fwd20(i) for i in entry_idx]
actual = [x for x in actual if x is not None]
boots = [np.mean(rng.choice(pop, size=len(actual), replace=False)) for _ in range(10000)] if len(pop) >= len(actual) else []
pct = round(100 * np.mean(np.array(boots) <= np.mean(actual)), 1) if boots else None

out = {
    "panic_days_full_hist": [str(all_d[i])[:10] for i in panic],
    "panic_event_clusters_engine_window": [str(all_d[i])[:10] for i in panic_events_eng],
    "population_event_fwd20": [round(x, 2) for x in pop],
    "population_mean": round(np.mean(pop), 2),
    "actual_8_entries": entries,
    "actual_fwd20": [round(x, 2) for x in actual],
    "actual_mean": round(np.mean(actual), 2),
    "bootstrap_percentile": pct,
    "verdict": "百分位>50 → 选择有增量; ≈50 → 纯恐慌反弹beta",
}
json_path = "/root/etf_strategy/audit/out/e_group.json"
r = json.load(open(json_path)); r["E8"] = out
json.dump(r, open(json_path, "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1))
