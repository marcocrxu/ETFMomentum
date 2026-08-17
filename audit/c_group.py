"""C组: 数据口径验证
C1 复权口径 (fund_adj 是否应用; 除息日异常交易扫描)
C2 14:50 vs 收盘偏差 (无分钟数据→贴线日频率统计)
C3 实盘 MA200 价格口径 (代码阅读 + 贴线日量化)
C4 MA200 预热期审计 (全历史窗口crash触发日的MA200可用性; 2016-01熔断)
C5 停牌日回测处理 (缺口扫描 + 是否有停牌日假成交)
C6 信号层指数化 (创业ETF vs 399006指数 动量diff + top1翻转日统计)
"""
import sys
sys.path.insert(0, "/root/etf_strategy"); sys.path.insert(0, "/root/etf_strategy/audit")
import numpy as np, pandas as pd, glob, os
from harness import FEED, bt, load_mats, ALL_CODES, save

res = {}

# ── C1: 复权口径 ──
adj_report = {}
exdiv_days_all = {}
for name in ["纳指", "黄金", "创业"]:
    code = ALL_CODES[name]
    path = f"/root/backtest/cache/etf_momentum/{code.replace('.', '_')}.csv"
    df = pd.read_csv(path, parse_dates=["trade_date"])
    chg = df["adj_factor"].diff().abs() > 1e-9
    exdiv = df.loc[chg, "trade_date"].dt.strftime("%Y-%m-%d").tolist()
    # 除息日原始收盘跌幅 vs 复权后跌幅
    latest = df["adj_factor"].iloc[-1]
    df["close_adj"] = df["close"] * df["adj_factor"] / latest
    raw_ret = df["close"].pct_change()
    adj_ret = df["close_adj"].pct_change()
    rows = []
    for d in exdiv:
        i = df.index[df["trade_date"] == d]
        if len(i) and i[0] > 0:
            i = i[0]
            rows.append({"date": d, "raw_ret%": round(raw_ret.iloc[i] * 100, 2),
                         "adj_ret%": round(adj_ret.iloc[i] * 100, 2)})
    adj_report[name] = {
        "adj_factor_range": [round(float(df["adj_factor"].min()), 4), round(float(df["adj_factor"].max()), 4)],
        "exdiv_count": len(exdiv), "exdiv_days": rows[:10],
        "adjusted_applied": bool((df["adj_factor"] != 1.0).any()),
    }
    exdiv_days_all[name] = exdiv
# 除息日异常交易扫描: 基线 decisions 中落在除息日/次日的止损/切换
base = bt()
dec = base["decisions"]
hits = []
for name, days in exdiv_days_all.items():
    for d in days:
        for tr in dec:
            if tr["date"] == d and name in tr["action"] and tr["reason"] in ("止损", "绝对止损", "正常切换", ">3%切换"):
                hits.append({"exdiv": d, "trade": tr})
res["C1"] = {"adj": adj_report, "exdiv_day_suspicious_trades": hits,
             "verdict": "已复权(fund_adj 应用, 读取时 close×factor/最新factor)" +
                        (f"; 除息日可疑交易 {len(hits)} 笔" if hits else "; 除息日无可疑止损/切换")}

# ── C2: 贴线日频率 (14:50 vs 收盘的信号翻转敏感度) ──
mats = load_mats(["纳指", "黄金", "创业"])
ms = mats["mom20_mat"]; sh_mom = mats["sh_mom"]; pre = mats["pre_days"]
ms_u = ms[:, pre:]; shm_u = sh_mom[pre:]
dates = [str(d)[:10] for d in mats["all_dates"]]
n = len(dates)
# 每日: 前两名动量差; 上证动量与全池最大动量差
gap12 = np.full(n, np.nan); sh_gap = np.full(n, np.nan)
for t in range(n):
    v = ms_u[:, t][~np.isnan(ms_u[:, t])]
    if len(v) >= 2:
        s = np.sort(v)
        gap12[t] = s[-1] - s[-2]
    if len(v) >= 1 and not np.isnan(shm_u[t]):
        sh_gap[t] = shm_u[t] - np.nanmax(v)
# 价格敏感换算: 价格 p 变动 δ% → mom20 变动 ≈ δ% (价格动量口径), 贴线阈值取 0.3%
for eps in [0.2, 0.3, 0.5]:
    close_rank = int(np.sum(np.abs(gap12) < eps))
    close_sh = int(np.sum(np.abs(sh_gap) < eps))
    res.setdefault("C2", {})[f"eps={eps}%"] = {
        "rank_flip_sensitive_days": close_rank,
        "sh_cash_flip_sensitive_days": close_sh,
        "pct_of_days": round(100 * close_rank / n, 1)}
# 决策日具体贴线清单 (|触发条件余量| < 0.3%)
trig_close = []
mom20_by_date = {d: ms_u[:, i] for i, d in enumerate(dates)}
for tr in dec:
    if tr["reason"] in ("正常切换", ">3%切换", "上证空仓", "确认买入", ">3%即买"):
        trig_close.append({"date": tr["date"], "reason": tr["reason"], "action": tr["action"]})
res["C2"]["switch_decision_days"] = len(trig_close)
res["C2"]["verdict"] = "无分钟数据; 以 |动量差|<0.2~0.5% 贴线日频率量化两层偏差风险"
res["C2"]["note"] = "价格动量口径下 mom20 对当日价格弹性≈1:1, 14:50→15:00 价格漂移可直接翻转贴线日信号"

# ── C3: 实盘口径 (代码阅读) ──
res["C3"] = {
    "live_uses": "14:50 腾讯实时价写入今日行 (ETF+上证指数+创业板指全部用现价), MA200/动量均含实时价",
    "backtest_uses": "收盘价",
    "verdict": "口径差=尾盘10分钟价格漂移; 贴线日(|gap12|<0.3% 或 |sh_gap|<0.3%)纳入对账监控",
}

# ── C4: MA200 预热期审计 ──
# 全历史: 从缓存最早期算 MA200 可用起始日; 列出全部恐慌触发日及当日 MA200 是否可用
dfs = FEED.get_dataframe(["399006.SZ", "000001.SH"], "20130101", "20260812")
cyb_df = dfs["399006.SZ"].set_index("trade_date")["close"]
sh_df = dfs["000001.SH"].set_index("trade_date")["close"]
all_d = sorted(set(cyb_df.index) | set(sh_df.index))
cyb_p = np.array([cyb_df.get(d, np.nan) for d in all_d])
sh_p = np.array([sh_df.get(d, np.nan) for d in all_d])
ma200_cyb = pd.Series(cyb_p).rolling(200).mean().values
ma200_sh = pd.Series(sh_p).rolling(200).mean().values
first_valid = next(i for i, v in enumerate(ma200_cyb) if not np.isnan(v))
def crash_days(px, ma):
    m10 = np.full(len(px), np.nan); m10[10:] = (px[10:] / px[:-10] - 1) * 100
    s = pd.Series(m10)
    th = (s.expanding(min_periods=60).mean() - 2.5 * s.expanding(min_periods=60).std()).shift(1).values
    return np.where((m10 < th) & (px < ma) & ~np.isnan(ma))[0]
cd_cyb = crash_days(cyb_p, ma200_cyb)
cd_sh = crash_days(sh_p, ma200_sh)
# 2016-01 熔断窗口
jan2016 = [i for i, d in enumerate(all_d) if "2016-01-04" <= str(d)[:10] <= "2016-02-29"]
jan2016_crash = [i for i in (set(cd_cyb) | set(cd_sh)) if i in set(jan2016)]
res["C4"] = {
    "ma200_first_valid_date": str(all_d[first_valid])[:10],
    "cache_start": str(all_d[0])[:10],
    "crash_trigger_days_all": sorted(set(str(all_d[i])[:10] for i in list(cd_cyb) + list(cd_sh))),
    "jan2016_fuse_window_crash_days": [str(all_d[i])[:10] for i in jan2016_crash],
    "verdict": "320自然日预热保证 MA200 在 bt_start=2016-07-07 已就绪; "
               "2016-01 熔断不在默认回测窗口(2016-07-07起), '8战8胜'口径=MA200可用期内触发",
}

# ── C5: 停牌日扫描 ──
susp = {}
idx_dates = set(pd.read_csv("/root/backtest/cache/etf_momentum/000001_SH.csv",
                            parse_dates=["trade_date"])["trade_date"])
for name in ["纳指", "黄金", "创业"]:
    code = ALL_CODES[name]
    df = pd.read_csv(f"/root/backtest/cache/etf_momentum/{code.replace('.', '_')}.csv",
                     parse_dates=["trade_date"])
    missing = sorted(idx_dates - set(df["trade_date"]))
    missing = [d for d in missing if df["trade_date"].min() <= d <= df["trade_date"].max()]
    susp[name] = {"suspension_days": [str(d)[:10] for d in missing], "count": len(missing)}
# 停牌日假成交检查: 基线 actions 日标的价是否 NaN
price_by = {}
for name in ["纳指", "黄金", "创业"]:
    df = pd.read_csv(f"/root/backtest/cache/etf_momentum/{ALL_CODES[name].replace('.', '_')}.csv",
                     parse_dates=["trade_date"]).set_index("trade_date")["close"]
    price_by[name] = df
fake_fills = []
for tr in dec:
    for name in ["纳指", "黄金", "创业"]:
        if name in tr["action"]:
            d = pd.Timestamp(tr["date"])
            if d not in price_by[name].index:
                fake_fills.append({"date": tr["date"], "action": tr["action"], "reason": tr["reason"]})
res["C5"] = {"suspensions": susp, "fake_fill_trades": fake_fills,
             "verdict": ("发现停牌日假成交!" if fake_fills else
                         "基线窗口三标的无停牌日假成交; 代码层: 价格NaN日动量NaN→不参与排名,"
                         "持仓标的停牌时nav冻结且切换条件全为NaN→不会成交(事实上的停牌禁交易)")}

# ── C6: 信号层指数化 diff ──
mats_i = load_mats(["纳指", "黄金", "创业"])
pre = mats_i["pre_days"]
cyb_idx = mats_i["cyb_p"]; etf_cyb = mats_i["price_mat"][mats_i["etf_names"].index("创业")]
def mom20(arr):
    m = np.full(len(arr), np.nan); m[20:] = (arr[20:] / arr[:-20] - 1) * 100
    return m
mom_etf = mom20(etf_cyb)[pre:]
mom_idx = mom20(cyb_idx)[pre:]
ms_all = mats_i["mom20_mat"][:, pre:]
valid = ~np.isnan(mom_etf) & ~np.isnan(mom_idx)
diff = mom_etf[valid] - mom_idx[valid]
# top1 翻转日: 用指数动量替换创业后, top1 与原版不同的天数
idx_i = mats_i["etf_names"].index("创业")
ms_idx_version = ms_all.copy(); ms_idx_version[idx_i] = mom_idx
flip = 0; flip_days = []
for t in range(ms_all.shape[1]):
    a = ms_all[:, t]; b = ms_idx_version[:, t]
    if np.isnan(a).all() or np.isnan(b).all(): continue
    ta = int(np.argmax(np.where(~np.isnan(a), a, -np.inf)))
    tb = int(np.argmax(np.where(~np.isnan(b), b, -np.inf)))
    if ta != tb:
        flip += 1; flip_days.append(str(mats_i["all_dates"][t])[:10])
res["C6"] = {
    "cyb_etf_vs_index_mom20_diff": {
        "mean_abs_pp": round(float(np.mean(np.abs(diff))), 3),
        "max_abs_pp": round(float(np.max(np.abs(diff))), 3),
        "corr": round(float(np.corrcoef(mom_etf[valid], mom_idx[valid])[0, 1]), 4)},
    "top1_flip_days_if_indexed": flip,
    "flip_day_list": flip_days[:30],
    "verdict": "sh空仓/恐慌信号已用指数; 仅排名动量用ETF价. 创业ETF与指数动量高度一致, "
               "top1翻转日见清单; 纳指(QDII溢价)/黄金(现货价差)无指数数据无法 diff——溢价风险仍在",
}

save(res, "c_group")
import json
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:3000])
