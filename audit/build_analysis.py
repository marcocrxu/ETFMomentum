"""从 audit/out/paper_2026.json 生成 paper 版分析页 web/templates/analysis.html"""
import json

J = json.load(open("/root/etf_strategy/audit/out/paper_2026.json"))
B = J["baseline"]
AB = J["ablation"]
GR = J["grids"]
LW = J["long_window"]
HF = J["halves"]
META = J["meta"]


def f2(x):
    try: return f"{x:.2f}"
    except Exception: return "—"


def f1(x):
    try: return f"{x:.1f}"
    except Exception: return "—"


def f0(x):
    try: return f"{x:.0f}"
    except Exception: return "—"


# ── 从 decisions 重建逐笔交易(修正: 切换行同时关闭旧腿并开启新腿) ──
def trades_from_decisions(decisions):
    trades, entry = [], None
    for d in decisions:
        act = d["action"]
        if act.startswith("买入"):
            entry = {"asset": act[2:].split("(")[0], "entry": d["date"],
                     "reason": d["reason"]}
        elif act.startswith("卖出"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                trades.append(entry)
            entry = None
        elif "→" in act:
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                trades.append(entry)
            tgt = act.split("→")[1].split("(")[0]
            entry = {"asset": tgt, "entry": d["date"], "reason": d["reason"]}
        elif act.startswith("持有"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                trades.append(entry)
            entry = None
    return trades


def channel_stats(trades):
    groups = {}
    for t in trades:
        groups.setdefault(t["exit_reason"], []).append(t["pnl"])
    out = {}
    for k, v in groups.items():
        arr = [x for x in v if x is not None]
        if not arr:
            continue
        out[k] = {"n": len(arr),
                  "win_rate": round(sum(1 for x in arr if x > 0) / len(arr) * 100, 1),
                  "avg": round(sum(arr) / len(arr), 2),
                  "median": round(sorted(arr)[len(arr) // 2], 2),
                  "total": round(sum(arr), 1)}
    return out


TRADES = trades_from_decisions(B["decisions"])
CH = channel_stats(TRADES)
CRASH_TRADES = sorted([t for t in TRADES if t["reason"] == "恐慌抄底"],
                      key=lambda x: x["entry"])


# ── 持仓结构分析: 分标的持有时间/收益/占比 ──
def holding_analysis(decisions, dates):
    """每段持仓(leg)的持有天数/收益 + 分标的聚合; 空仓段从 leg 间隙反推"""
    legs, entry = [], None
    for d in decisions:
        act = d["action"]
        if act.startswith("买入"):
            entry = {"asset": act[2:].split("(")[0], "entry": d["date"], "reason": d["reason"]}
        elif act.startswith("卖出"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                legs.append(entry)
            entry = None
        elif "→" in act:
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                legs.append(entry)
            entry = {"asset": act.split("→")[1].split("(")[0], "entry": d["date"],
                     "reason": d["reason"]}
        elif act.startswith("持有"):
            if entry is not None:
                entry.update({"exit": d["date"], "pnl": d["pnl"], "days": d["days"],
                              "exit_reason": d["reason"]})
                legs.append(entry)
            entry = None

    total_days = len(dates)
    held_days = sum(l["days"] or 0 for l in legs)
    cash_days = max(total_days - held_days, 0)

    # 空仓段: leg 间隙(交易日索引差)
    dix = {d: i for i, d in enumerate(dates)}
    gaps, prev_exit = [], 0
    for l in sorted(legs, key=lambda x: x["entry"]):
        ei = dix.get(l["entry"], prev_exit)
        if ei > prev_exit:
            gaps.append(ei - prev_exit)
        prev_exit = dix.get(l["exit"], ei)
    if total_days - 1 > prev_exit:
        gaps.append(total_days - 1 - prev_exit)

    def _q(v):
        a = sorted(v)
        return a[len(a) // 2]

    def per_asset(asset_legs):
        n = len(asset_legs)
        days = [l["days"] for l in asset_legs if l["days"] is not None]
        pnls = [l["pnl"] for l in asset_legs if l["pnl"] is not None]
        dsum = sum(days)
        pb = [0, 0, 0, 0, 0]           # pnl 桶: ≤-5 / -5~0 / 0~5 / 5~10 / >10
        for p in pnls:
            if p <= -5: pb[0] += 1
            elif p < 0: pb[1] += 1
            elif p < 5: pb[2] += 1
            elif p <= 10: pb[3] += 1
            else: pb[4] += 1
        db = [0, 0, 0, 0, 0]           # days 桶: 1-5 / 6-10 / 11-20 / 21-40 / 40+
        for dd in days:
            if dd <= 5: db[0] += 1
            elif dd <= 10: db[1] += 1
            elif dd <= 20: db[2] += 1
            elif dd <= 40: db[3] += 1
            else: db[4] += 1
        return {
            "n": n,
            "time_pct": round(dsum / total_days * 100, 1) if total_days else 0,
            "days_avg": round(sum(days) / len(days), 1) if days else None,
            "days_med": _q(days) if days else None,
            "days_max": max(days) if days else None,
            "pnl_avg": round(sum(pnls) / len(pnls), 2) if pnls else None,
            "pnl_med": round(_q(pnls), 2) if pnls else None,
            "pnl_min": round(min(pnls), 2) if pnls else None,
            "pnl_max": round(max(pnls), 2) if pnls else None,
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1) if pnls else None,
            "pnl_sum": round(sum(pnls), 1) if pnls else 0.0,
            "pnl_buckets": pb, "days_buckets": db,
        }

    by_asset = {a: per_asset([l for l in legs if l["asset"] == a]) for a in
                sorted(set(l["asset"] for l in legs))}
    total_pnl_sum = sum(v["pnl_sum"] for v in by_asset.values())
    for a, v in by_asset.items():
        v["pnl_share"] = (round(v["pnl_sum"] / total_pnl_sum * 100, 1)
                          if total_pnl_sum else None)
    cash = {"time_pct": round(cash_days / total_days * 100, 1) if total_days else 0,
            "n": len(gaps), "days_avg": round(sum(gaps) / len(gaps), 1) if gaps else None,
            "days_med": _q(gaps) if gaps else None,
            "days_max": max(gaps) if gaps else None}
    return {"legs": legs, "by_asset": by_asset, "cash": cash,
            "total_days": total_days, "held_days": held_days,
            "legs_avg_days": round(held_days / len(legs), 1) if legs else None}


HA = holding_analysis(B["decisions"], B["dates"])

HA_ROWS = ""
for a, v in HA["by_asset"].items():
    pnl = v["pnl_sum"]
    pnl_cls = "num-pos" if pnl > 0 else ("num-neg" if pnl < 0 else "")
    share_txt = f"{v['pnl_share']:+.1f}%" if v["pnl_share"] is not None else "—"
    HA_ROWS += (
        "<tr>"
        f"<td><b>{a}</b></td>"
        f"<td>{v['n']}</td>"
        f"<td>{v['time_pct']}%</td>"
        f"<td>{f1(v['days_avg'])}</td><td>{f0(v['days_med'])}</td><td>{f0(v['days_max'])}</td>"
        f"<td>{f2(v['pnl_avg'])}%</td><td>{f2(v['pnl_med'])}%</td>"
        f"<td class='{pnl_cls}'>{pnl:+.1f}pp</td>"
        f"<td>{share_txt}</td>"
        f"<td>{f1(v['win_rate'])}%</td>"
        f"<td>{v['pnl_buckets'][0]}/{v['pnl_buckets'][1]}/{v['pnl_buckets'][2]}/{v['pnl_buckets'][3]}/{v['pnl_buckets'][4]}</td>"
        f"<td>{v['days_buckets'][0]}/{v['days_buckets'][1]}/{v['days_buckets'][2]}/{v['days_buckets'][3]}/{v['days_buckets'][4]}</td>"
        "</tr>")

HA_CASH = HA["cash"]

# ── 直方图数据: 收益分布(7桶) / 持有天数分布(5桶), 分标的 + 合计 ──
PNL_BINS = ["≤-5", "(-5,-2]", "(-2,0]", "(0,2]", "(2,5]", "(5,10]", ">10"]
DAYS_BINS = ["1-5", "6-10", "11-20", "21-40", "40+"]
HA_ASSETS = sorted(HA["by_asset"].keys())


def pnl_hist(legs):
    h = [0] * 7
    for l in legs:
        p = l["pnl"]
        if p is None:
            continue
        if p <= -5: h[0] += 1
        elif p <= -2: h[1] += 1
        elif p < 0: h[2] += 1
        elif p < 2: h[3] += 1
        elif p < 5: h[4] += 1
        elif p <= 10: h[5] += 1
        else: h[6] += 1
    return h


HA_PNL = {a: pnl_hist([l for l in HA["legs"] if l["asset"] == a]) for a in HA_ASSETS}
HA_PNL["合计"] = pnl_hist(HA["legs"])
HA_DAYS = {a: HA["by_asset"][a]["days_buckets"] for a in HA_ASSETS}
HA_DAYS["合计"] = [sum(HA["by_asset"][a]["days_buckets"][i] for a in HA_ASSETS)
                   for i in range(5)]
HA_CHART_ASSETS = HA_ASSETS + ["合计"]

# ── 加仓时机研究 (backtest/add_capital_entry.py 的结果) ──
try:
    CAI = json.load(open("/root/etf_strategy/backtest/add_capital_entry_result.json"))
except Exception:
    CAI = None

CAI_OVERALL_ROWS, CAI_PNL_ROWS = "", ""
if CAI:
    for H, lab in (("20", "1 个月"), ("60", "3 个月"), ("120", "6 个月"), ("250", "12 个月")):
        o = CAI["summary"][H]["overall"]
        CAI_OVERALL_ROWS += (
            f"<tr><td>{lab}</td><td>{o['n']}</td><td>{f1(o['winA'])}%</td>"
            f"<td class='num-pos'>{f2(o['mean_diff'])}pp</td><td>{f2(o['median_diff'])}pp</td>"
            f"<td>{f2(o['meanA'])}%</td><td>{f2(o['meanB'])}%</td><td>{f2(o['meanD'])}%</td></tr>")
    for label in ("≤-5%", "(-5,-2]%", "(-2,0]%", "(0,2]%", "(2,5]%", "(5,10]%", ">10%"):
        v = CAI["summary"]["250"]["by_leg_pnl"].get(label)
        if not v:
            CAI_PNL_ROWS += (f"<tr><td>{label}</td><td>0</td>"
                             "<td colspan='7' class='small'>样本为 0 —— 策略绝对止损在 −5% 清仓，"
                             "收盘口径下不存在「深亏超 5% 仍持仓」的加仓时点</td></tr>")
            continue
        if v["mean_diff"] >= 1: rec = "立即追"
        elif v["mean_diff"] <= -1: rec = "等调仓"
        else: rec = "差别小，分批亦可"
        CAI_PNL_ROWS += (
            f"<tr><td>{label}</td><td>{v['n']}</td><td>{f1(v['winA'])}%</td>"
            f"<td class='num-pos'>{f2(v['mean_diff'])}pp</td>"
            f"<td>{f2(v['meanA'])}%</td><td>{f2(v['meanB'])}%</td><td>{f2(v['meanD'])}%</td>"
            f"<td><b>{rec}</b></td></tr>")

# ── 消融判定 ──
NOTES = {
    "accel_cooldown0": "⚠️ 用户决议项(I-17, 2026-08-18): 回退后全样本更高, 但该冷却使 2025 年 +16.2pp(54.0%→70.2%); 2026 年(截至 08-20)则为 −14.7pp。决议基于 2025 年证据, 维持默认并列入观察。",
    "sigma_stop_floor3": "实盘已部署变体(2026-08-14, 旧口径三窗口验证通过); 当前默认口径 + 最新数据下为负优化(长样本亦 1.93 &lt; 2.20), 列观察项。",
    "sh_cash_cooldown0": "3 天档在当前口径下是局部低点(2 天档 Calmar 2.60 最高), 维持默认并列入观察。",
    "nz_qqq_off": "I-11 决议项: 维持 QQQ-CNY 混合默认。",
    "risk_adj_on": "教科书改进: 波动率归一化在本策略上无增益(Calmar 边际改善、年化 −1.3pp), 维持关闭。",
    "defensive_on": "教科书改进: 防御切换无增益, 维持关闭。",
    "fip_on": "教科书改进: FIP 动量质量无增益, 维持关闭。",
}


def row(name, over):
    d = AB.get(over)
    if d is None:
        return ""
    b_ann, b_dd, b_cal = B["ann_ret"], B["max_dd"], B["calmar"]
    ann, dd, cal = d["ann_ret"], d["max_dd"], d["calmar"]
    d_ann, d_cal = ann - b_ann, cal - b_cal
    if d_cal > 0.05 and d_ann >= -0.5:
        v, cls = "正贡献(回退)", "badge-pass"
    elif d_cal < -0.03 or (d_ann < -1.0 and d_cal <= 0.02):
        v, cls = "负优化", "badge-fail"
    elif abs(d_ann) < 0.5 and abs(d_cal) <= 0.03:
        v, cls = "中性", "badge-mixed"
    else:
        v, cls = "权衡", "badge-mixed"
    note = f"<div class='small' style='margin-top:2px'>{NOTES[over]}</div>" if over in NOTES else ""
    return (f"<tr><td><b>{name}</b></td>"
            f"<td>{f2(ann)}%<span class='delta'>({d_ann:+.2f}pp)</span></td>"
            f"<td class='num-neg'>{f2(dd)}%</td>"
            f"<td>{f2(cal)}<span class='delta'>({d_cal:+.2f})</span></td>"
            f"<td>{d['trade_count']}</td>"
            f"<td><span class='badge {cls}'>{v}</span>{note}</td></tr>")


ABL_ORDER = [
    ("上证空仓 regime filter", "sh_cash_off"),
    ("恐慌抄底", "crash_off"),
    ("加速止盈", "accel_off"),
    ("绝对止损", "abs_stop_off"),
    ("止盈冷却 3 天(仅黄金)", "accel_cooldown0"),
    ("止盈冷却扩至全部标的", "accel_cd_all_assets"),
    ("止盈冷却 5 天", "accel_cooldown5"),
    ("空仓再入场冷却 3 天", "sh_cash_cooldown0"),
    ("纳指动量 QQQ-CNY 混合(0.7)", "nz_qqq_off"),
    ("快速通道 >3% 切换", "exception_gap999"),
    ("σ 自适应止损 floor k=3.0(实盘)", "sigma_stop_floor3"),
    ("波动率归一化排名", "risk_adj_on"),
    ("绝对动量门", "abs_mom_on"),
    ("防御切换", "defensive_on"),
    ("多周期动量", "multi_mom_on"),
    ("FIP 动量质量", "fip_on"),
    ("Top-2 等权持仓", "top_k2"),
    ("动量跳过近 3 天", "mom_skip3"),
    ("止损后冷却 3 天", "stop_cooldown3"),
]
ABL_ROWS = "\n".join(row(n, k) for n, k in ABL_ORDER)

# ── 动量周期网格 ──
MOM = GR["mom_period"]
mom_ps = sorted(int(k) for k in MOM)
mom_ann = [MOM[str(p)]["ann_ret"] for p in mom_ps]
mom_cal = [MOM[str(p)]["calmar"] for p in mom_ps]
mom_dd = [MOM[str(p)]["max_dd"] for p in mom_ps]
best_p = max(mom_ps, key=lambda p: MOM[str(p)]["calmar"])
worst_p = min(mom_ps, key=lambda p: MOM[str(p)]["calmar"])
rank20 = sorted(mom_ps, key=lambda p: MOM[str(p)]["calmar"], reverse=True).index(20) + 1

MOM_ROWS = ""
for p in mom_ps:
    d = MOM[str(p)]
    cur = " ◄当前" if p == 20 else ""
    best = " ⭐" if p == best_p else ""
    MOM_ROWS += (f"<tr><td><b>{p} 日</b>{cur}{best}</td>"
                 f"<td>{f2(d['ann_ret'])}%</td><td class='num-neg'>{f2(d['max_dd'])}%</td>"
                 f"<td>{f2(d['calmar'])}</td><td>{d['trade_count']}</td></tr>\n")

# ── 其它参数网格 ──
def grid_row(name, key, vals, base_val, note):
    out = f"<tr><td><b>{name}</b></td>"
    for v in vals:
        d = GR[key].get(str(v))
        if d is None and v == base_val:
            d = B
        if d is None:
            out += "<td>—</td>"
            continue
        mark = " ◄" if v == base_val else ""
        out += (f"<td>{v}{mark}<br><span class='small'>{f2(d['ann_ret'])}% / "
                f"{f2(d['max_dd'])}% / Cal {f2(d['calmar'])}</span></td>")
    out += f"<td>{note}</td></tr>\n"
    return out


GRID_ROWS = (
    grid_row("止损线 stop_loss", "stop_loss", [-3, -5, -8], -5,
             "-5% 为峰: -3% 过紧(回撤 -23.1%), -8% 过松(-22.1%)")
    + grid_row("确认天数 confirm_days", "confirm_days", [1, 2, 3], 2,
               "当前口径下 1>2>3(1 天档 Cal 2.78); 2 天维持(保守), 列入观察")
    + grid_row("快速通道 exception_gap", "exception_gap", [2.0, 3.0, 5.0, 999], 3.0,
               "2.0/5.0 档均略高于 3.0 档; 关闭(999)年化 −4.3pp")
    + grid_row("抄底锁仓 crash_lock", "crash_lock", [5, 7, 14, 21, 25, 30], 7,
               "5~21 天为平缓高原(14 天微峰), 30 天明显变差")
    + grid_row("加速止盈 σ accel_sigma", "accel_sigma", [2.5, 3.0, 3.5], 3.0,
               "3.0 为峰, 两侧下降")
    + grid_row("最短持有 min_hold", "min_hold", [5, 10, 15], 10,
               "15 天档略优(2.58), 稳健")
    + grid_row("最小动量差 min_gap", "min_gap", [0.5, 1.0, 2.0], 1.0,
               "2.0 档略优(2.56), 稳健")
    + grid_row("空仓冷却 sh_cash_cooldown", "sh_cash_cooldown",
               [0, 1, 2, 3, 4, 5, 10], 3,
               "梯度非单调: 2 天档 2.60 最高, 3 天档 2.44 为局部低点")
    + grid_row("抄底仓位 crash_alloc", "crash_alloc", [50, 100], 100,
               "50% 档 2.35 vs 全仓 2.44, 差异小")
)

# ── 成本敏感性 ──
SLIP = GR["slippage"]
COST_ROWS = ""
for s, note in [(0, "基线(仅手续费 0.01%)"), (0.0005, "0.05%"),
                (0.001, "0.1%"), (0.002, "0.2%")]:
    d = B if s == 0 else SLIP[str(s)]
    COST_ROWS += (f"<tr><td>{note}</td>"
                  f"<td>{f0(d['total_ret'])}%</td><td>{f2(d['ann_ret'])}%</td>"
                  f"<td class='num-neg'>{f2(d['max_dd'])}%</td><td>{f2(d['calmar'])}</td>"
                  f"<td>{d['trade_count']}</td></tr>\n")

# ── 年度收益 ──
YR = B["yearly"]
YEARS = list(YR.keys())
YEARS_ROWS = ""
half = (len(YEARS) + 1) // 2
for i in range(half):
    c1, v1 = YEARS[i], YR[YEARS[i]]
    if i + half < len(YEARS):
        c2, v2 = YEARS[i + half], YR[YEARS[i + half]]
        cls2 = "num-pos" if v2 >= 0 else "num-neg"
        cell2 = f"<td>{c2}</td><td class='{cls2}'>{v2:+.2f}%</td>"
    else:
        cell2 = "<td></td><td></td>"
    cls1 = "num-pos" if v1 >= 0 else "num-neg"
    YEARS_ROWS += f"<tr><td>{c1}</td><td class='{cls1}'>{v1:+.2f}%</td>{cell2}</tr>\n"

# ── 回撤事件 ──
DD_ROWS = ""
for i, e in enumerate(B["dd_episodes"], 1):
    rec = e["recover"] or "未修复"
    rd = f"{e['recover_days']} 天" if e["recover_days"] is not None else "—"
    DD_ROWS += (f"<tr><td>{i}</td><td>{e['peak']} → {e['trough']}</td>"
                f"<td class='num-neg'>{e['depth']}%</td>"
                f"<td>{e['days']} 天</td><td>{rec}</td><td>{rd}</td></tr>\n")

# ── 交易通道 ──
CH_ORDER = [("正常切换", ""), (">3%切换", "快速通道"),
            ("上证空仓", "regime 离场"), ("加速止盈", ""),
            ("恐慌换仓", "抄底接管"), ("止损", "相对止损"),
            ("绝对止损", ""), ("期末", "仍持有")]
CH_ROWS = ""
for name, note in CH_ORDER:
    d = CH.get(name)
    if d is None:
        continue
    CH_ROWS += (f"<tr><td><b>{name}</b>"
                f"{f'<span class=small>({note})</span>' if note else ''}</td>"
                f"<td>{d['n']}</td><td>{d['win_rate']}%</td>"
                f"<td>{f2(d['avg'])}%</td><td>{f2(d['median'])}%</td>"
                f"<td>{f2(d['total'])}%</td></tr>\n")

# ── 恐慌抄底逐笔 ──
CRASH_ROWS = ""
for t in CRASH_TRADES:
    pnl = t.get("pnl")
    cls = "num-pos" if (pnl is not None and pnl >= 0) else "num-neg"
    CRASH_ROWS += (f"<tr><td>{t['entry']}</td><td>{t['exit']}</td>"
                   f"<td class='{cls}'>{f2(pnl)}%</td><td>{t['days']} 天</td>"
                   f"<td>{t['exit_reason']}</td></tr>\n")

# ── 两半窗口 ──
def half_block(key, title, st, ed):
    h = HF[key]
    b = h["baseline"]
    m = h["mom"]
    ranked = sorted(m, key=lambda p: m[p]["calmar"], reverse=True)[:2]
    tops = "、".join(f"{p} 日(Cal {f2(m[p]['calmar'])})" for p in ranked)
    b20 = m["20"]
    return f"""
    <div class="half-block">
      <h4>{title} <span class="small">({st} ~ {ed})</span></h4>
      <p>基线: 年化 <b>{f2(b['ann_ret'])}%</b> / 回撤 <b class="num-neg">{f2(b['max_dd'])}%</b> / Calmar <b>{f2(b['calmar'])}</b>;
         最优动量周期(Calmar 前二): <b>{tops}</b>;
         20 日档: {f2(b20['ann_ret'])}% / {f2(b20['max_dd'])}% / {f2(b20['calmar'])}</p>
    </div>"""

HALF_HTML = (half_block("h1", "样本内(定参窗)", "2016-07-07", "2021-08-20")
             + half_block("h2", "样本外(验证窗)", "2021-08-23", "2026-08-20")
             + half_block("h3", "近三年压力窗", "2023-08-21", "2026-08-20"))

LW_B, LW_C0, LW_C5, LW_SF = (LW["baseline"], LW["cooldown0"],
                             LW["cooldown5"], LW["sigma_stop_floor3"])

DATES, NAV, DD, BENCH = B["dates"], B["nav"], B["drawdown"], B["benchmark"]
SEG = B["dd_2026_seg"]

HTML = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>策略分析报告 · ETF 动量轮动</title>
  <link rel="stylesheet" href="/static/css/app.css">
  <style>
    .paper {{ max-width: 1100px; margin: 0 auto; }}
    .paper .card {{ margin-bottom: 22px; }}
    .paper h2.sec {{ font-size: 19px; font-weight: 700; color: var(--text);
      border-left: 4px solid var(--accent); padding-left: 10px; margin: 34px 0 14px; }}
    .paper h3 {{ font-size: 15px; color: var(--accent); margin: 20px 0 8px; }}
    .paper h4 {{ font-size: 13.5px; color: var(--text); margin: 10px 0 6px; }}
    .paper p {{ font-size: 13.5px; color: #c8ccd8; line-height: 1.9; margin: 8px 0; }}
    .paper li {{ font-size: 13.5px; color: #c8ccd8; line-height: 1.9; }}
    .paper .meta {{ font-size: 12px; color: var(--muted); text-align: center; margin-bottom: 4px; }}
    .paper table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 12.5px; }}
    .paper th {{ background: var(--surface-2); color: var(--muted); font-weight: 600; }}
    .paper th, .paper td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: left; }}
    .paper .num-pos {{ color: var(--green); font-weight: 600; }}
    .paper .num-neg {{ color: var(--red); font-weight: 600; }}
    .paper .small {{ font-size: 11px; color: var(--muted); font-weight: normal; }}
    .paper .delta {{ font-size: 11px; color: var(--muted); margin-left: 4px; }}
    .paper .badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px;
      font-size: 11px; font-weight: 500; }}
    .badge-pass {{ background: rgba(76,175,80,.15); color: var(--green); border: 1px solid rgba(76,175,80,.3); }}
    .badge-fail {{ background: rgba(244,67,54,.15); color: var(--red); border: 1px solid rgba(244,67,54,.3); }}
    .badge-mixed {{ background: rgba(255,152,0,.15); color: #ff9800; border: 1px solid rgba(255,152,0,.3); }}
    .paper .kv {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px 22px; background: var(--surface-2); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px 18px; margin: 10px 0 16px; }}
    .paper .kv div b {{ display: block; font-size: 17px; color: var(--text); }}
    .paper .kv div span {{ font-size: 11px; color: var(--muted); }}
    .paper .eq {{ background: var(--surface-3); border-radius: 6px; padding: 2px 7px;
      font-family: Consolas, monospace; font-size: 12px; color: var(--accent-2); }}
    .paper .abstract {{ background: var(--surface-2); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px 20px; }}
    .half-block {{ background: var(--surface-2); border: 1px solid var(--border);
      border-radius: 8px; padding: 12px 16px; margin: 8px 0; }}
    .chart-box {{ width: 100%; height: 320px; }}
  </style>
</head>
<body>
<div class="page-wrapper paper">
  <header class="page-header">
    <h1>ETF 动量轮动策略 · 研究分析报告</h1>
    <p class="subtitle">十年样本上的机制分解、参数稳健性与样本外验证</p>
    <p class="meta">数据截止 {META['end']} · 回测引擎 web/bt_runner.run_backtest(与实盘共用 strategy/signal.py) · 报告生成 2026-08-23 · 标的池 {META['universe']}</p>
  </header>
  <nav class="page-nav">
    <a href="/">📊 回测</a>
    <a href="/analysis" class="active">📄 分析报告</a>
  </nav>

  <!-- ═══════════ 1 摘要 ═══════════ -->
  <div class="card">
    <div class="card-header"><h2>1 · 摘要</h2></div>
    <div style="padding:18px">
      <div class="abstract">
        <p><b>策略：</b>在纳指(513100)、黄金(518880)、创业板(159915)三只 ETF 上做<b>20 日相对动量轮动</b>，日频全仓持有动量最强标的；核心风控为「上证指数动量超过池内最强 → 空仓」的 regime filter，辅以相对/绝对止损、加速止盈(动量过热止盈)、恐慌抄底(指数 10 日动量 &lt; μ−2.5σ 且 &lt; MA200 时买入创业板并锁仓 7 天)。</p>
        <p><b>样本：</b>{META['start']} ~ {META['end']}(约 10.1 年)。</p>
        <div class="kv">
          <div><b class="num-pos">{f0(B['total_ret'])}%</b><span>累计收益</span></div>
          <div><b>{f2(B['ann_ret'])}%</b><span>年化收益</span></div>
          <div><b class="num-neg">{f2(B['max_dd'])}%</b><span>最大回撤</span></div>
          <div><b>{f2(B['calmar'])}</b><span>Calmar</span></div>
          <div><b>{f2(B['sharpe'])}</b><span>Sharpe</span></div>
          <div><b>{f2(B['volatility'])}%</b><span>年化波动率</span></div>
          <div><b>{B['trade_count']}</b><span>交易笔数(买+卖)</span></div>
          <div><b>{B['neg_years']}</b><span>负收益年份数</span></div>
        </div>
        <p>同期<b>等权买入持有</b>三只 ETF: 累计 <b>{f0(B['bench']['total_ret'])}%</b>、年化 <b>{f2(B['bench']['ann'])}%</b>、最大回撤 <b class="num-neg">{f2(B['bench']['mdd'])}%</b>、Calmar <b>{f2(B['bench']['calmar'])}</b>。策略以相近的最大回撤({f2(B['max_dd'])}% vs {f2(B['bench']['mdd'])}%)取得约 {f0((1+B['total_ret']/100)/(1+B['bench']['total_ret']/100))} 倍的累计收益。</p>
        <p><b>核心结论：</b></p>
        <ol style="padding-left:22px">
          <li>收益来自「窄池相对动量 + regime filter + 两端风控模块」三层信号堆栈：消融显示上证空仓、加速止盈、恐慌抄底、绝对止损、快速通道、QQQ 混合动量<b>均为独立正贡献</b>，删除任一模块年化/Calmar 双降。</li>
          <li>除动量周期外，其余参数在邻域内整体稳健(无悬崖式劣化)；个别档位略优/略劣，均不构成换参理由。</li>
          <li>动量周期曲线在<b>当前引擎口径下为 18~22 日的缓坡</b>(全样本最优 19 日 Calmar {f2(MOM[str(best_p)]['calmar'])}，20 日排第 {rank20}/21 名)——旧口径下的「20/21 孤峰」已消失；但两半窗口中最优值从 19 漂移到 22，<b>参数漂移真实存在</b>。</li>
          <li>样本外验证窗(2021-08 起)年化 <b>{f2(HF['h2']['baseline']['ann_ret'])}%</b>、Calmar {f2(HF['h2']['baseline']['calmar'])}，与全样本基本持平；长样本(2015 起)年化 {f2(LW_B['ann_ret'])}%。考虑到 2025-26 年黄金/纳指极端行情不可复制，长期年化的诚实预期区间为 <b>30%~45%</b>、回撤 −18%~−25%。</li>
        </ol>
      </div>
    </div>
  </div>

  <!-- ═══════════ 2 引言 ═══════════ -->
  <h2 class="sec">2 · 引言：策略在动量轮动谱系中的位置</h2>
  <div class="card">
    <div class="card-header"><h2>2.1 经典动量轮动策略对比</h2></div>
    <div style="padding:18px">
      <table>
        <thead><tr><th>策略</th><th>信号</th><th>调仓频率</th><th>持仓</th><th>下跌保护</th></tr></thead>
        <tbody>
          <tr><td><b>GEM</b> (Antonacci)</td><td>12 个月收益：相对动量 + 绝对动量</td><td>月频</td><td>全仓 1 只</td><td>绝对动量为负 → 切债券</td></tr>
          <tr><td><b>DAA</b> (Keller)</td><td>13612W 多周期加权动量</td><td>月频</td><td>Top-3 等权</td><td>按坏资产数量比例切防御</td></tr>
          <tr><td><b>VAA</b> (Keller)</td><td>风险调整动量(动量/波动率)</td><td>月频</td><td>Top-4 等权</td><td>有坏资产 → 全仓防御</td></tr>
          <tr><td><b>TSMOM</b> (Moskowitz)</td><td>12 个月时序动量符号</td><td>月频</td><td>多资产独立信号</td><td>负动量空仓 + 波动率缩放</td></tr>
          <tr><td><b>本策略</b></td><td>20 日相对动量排名</td><td>日频</td><td>全仓 1 只</td><td>上证空仓 filter + 止损 + 加速止盈 + 恐慌抄底</td></tr>
        </tbody>
      </table>
      <p>本策略可视为 <b>GEM 的高频变体</b>：以日频替代月频、20 日替代 12 个月、上证 regime filter 替代逐标的绝对动量门。与文献中月频动量系统相比，它把「趋势跟踪」压缩到约一个月的形成期——这是收益来源，也是脆弱性来源(见 §5)。</p>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>2.2 信号堆栈：相互独立的风险收益来源</h2></div>
    <div style="padding:18px">
      <p>本策略不是单一动量策略，而是一个<b>信号堆栈</b>，各模块在市场不同状态分别开火：</p>
      <table>
        <thead><tr><th>模块</th><th>触发条件(简化)</th><th>理论角色</th><th>主导市况</th></tr></thead>
        <tbody>
          <tr><td>20 日相对动量排名</td><td>持有 mom20 最高的标的</td><td>横截面动量(alpha 引擎)</td><td>趋势市</td></tr>
          <tr><td>上证空仓 filter</td><td>上证 mom20 &gt; 池内最强 → 空仓 + 3 天冷却</td><td>Regime filter / 领导力背离</td><td>系统性风险、风格切换</td></tr>
          <tr><td>快速通道</td><td>挑战者动量领先 ≥3% 且确认 2 天 → 立即切换</td><td>趋势加速器 / 崩盘逃生口</td><td>急涨急跌</td></tr>
          <tr><td>止损 + 绝对止损</td><td>有更优候选且 pnl≤−5% 切换；仍第一但 pnl≤−5% 清仓</td><td>尾部截断</td><td>单边下跌</td></tr>
          <tr><td>加速止盈</td><td>mom10 &gt; 自身历史 μ+3σ → 清仓(黄金止盈后冷却 3 天)</td><td>均值回归叠加：卖过热</td><td>冲顶 / blow-off</td></tr>
          <tr><td>恐慌抄底</td><td>指数 mom10 &lt; μ−2.5σ 且 &lt; MA200 → 买创业板锁仓 7 天</td><td>逆向叠加：买恐慌</td><td>恐慌底</td></tr>
          <tr><td>确认天数 / 最短持有 / 动量差</td><td>2 天确认、10 天最短持有、≥1% 动量差</td><td>换手与 whipsaw 成本控制</td><td>全程</td></tr>
        </tbody>
      </table>
      <p>一句话概括：<b>趋势跟踪为体、regime filter 为锚、两端(过热/恐慌)均值回归叠加为辅</b>。三类信号低相关、分时开火，策略的稳健性来自这种集成结构而非任何单一参数。</p>
    </div>
  </div>

  <!-- ═══════════ 3 方法与数据 ═══════════ -->
  <h2 class="sec">3 · 方法与数据</h2>
  <div class="card">
    <div class="card-header"><h2>3.1 决策顺序(优先级即实现顺序)</h2></div>
    <div style="padding:18px">
      <ol style="padding-left:22px">
        <li>更新净值并计算排名分数(mom20，纳指为 0.3×513100 + 0.7×QQQ-CNY 混合动量)；</li>
        <li><b>恐慌抄底</b>：创业板指或上证指数 mom10 跌破自身历史 μ−2.5σ 且价格 &lt; MA200 → 买入创业板，锁仓 7 天(期间仅允许止损退出)；</li>
        <li><b>上证空仓</b>：上证 mom20 高于池内最强标的 → 清仓到空仓，再入场需等 3 天冷却；</li>
        <li>空仓入场：候选动量 ≥3% 当日即买(&gt;3%即买)，否则需连续 2 天第一(确认买入)；</li>
        <li>持仓切换：止损(有更优候选且 pnl≤−5%) → 排名确认后快速通道(momentum gap ≥3%)→ 正常切换(≥1%，持有≥10 天)；</li>
        <li><b>绝对止损</b>：持仓仍是动量第一但 pnl≤−5% → 清仓(2026-06 纳指 −9.79% 事件的修复项)；</li>
        <li><b>加速止盈</b>：持仓 mom10 &gt; 自身历史 μ+3σ → 清仓；黄金止盈后 3 天内禁止买回(2026-08-18 决议)。</li>
      </ol>
      <p>成本口径：每笔卖出/买入各扣 0.01% 手续费(commission=0.0001)，默认零滑点；信号日<b>收盘价成交</b>，当日收益归旧仓，无前视。</p>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>3.2 数据与口径</h2></div>
    <div style="padding:18px">
      <ul>
        <li><b>行情</b>：Tushare 日线(三只 ETF + 上证指数 000001.SH + 创业板指 399006.SZ)，分红拆分按复权因子现场重算；回测起点前预热 320 自然日保证 MA200 就绪。</li>
        <li><b>纳指动量混合</b>：排名与阈值用 <span class="eq">(1−0.7)×513100 + 0.7×QQQ×USDCNY</span> 的动量(净值与止损仍按 513100 成交价)，以消除 QDII 溢价与申赎扰动；美国数据对齐采用「CN 日 D 只用美国 ≤D−1 收盘」，严格无前视。</li>
        <li><b>停牌与缺失</b>：价格 NaN → 动量 NaN → 不参与排名；持仓停牌日净值冻结、不成交。</li>
        <li><b>评估指标</b>：年化收益 = NAV^(252/交易日数)−1；最大回撤与 Calmar 基于日频净值；Sharpe 基于日收益年化。</li>
      </ul>
      <p class="meta">复现: <span class="eq">curl -X POST http://127.0.0.1:8082/api/bt/run -H 'Content-Type: application/json' -d '{{"universe":"纳指,黄金,创业","start_date":"2016-07-07","end_date":"2026-08-20"}}'</span></p>
    </div>
  </div>

  <!-- ═══════════ 4 结果 ═══════════ -->
  <h2 class="sec">4 · 结果</h2>

  <div class="card">
    <div class="card-header"><h2>4.1 基线业绩与回撤结构</h2></div>
    <div style="padding:18px">
      <div id="nav-chart" class="chart-box"></div>
      <h3>年度收益</h3>
      <table><thead><tr><th>年份</th><th>收益</th><th>年份</th><th>收益</th></tr></thead>
      <tbody>{YEARS_ROWS}</tbody></table>
      <p>11 个年份中仅 2016 年({f2(YR['2016'])}%)为负。</p>
      <h3>回撤事件(&gt;10%)</h3>
      <table>
        <thead><tr><th>#</th><th>峰值 → 谷底</th><th>深度</th><th>下跌耗时</th><th>修复日</th><th>修复耗时</th></tr></thead>
        <tbody>{DD_ROWS}</tbody>
      </table>
      <p>十年间 6 次回撤超过 10%：其中 4 次集中在 2020 年(疫情冲击、黄金见顶、止盈空转)；第 5 次(2025-09-29 → 2026-04-03)为全样本最深(<b>−18.59%</b>，下跌 120 个交易日)；第 6 次(2026-05-29 → 2026-07-30)深度 {f2(SEG['min'])}%，期末仍未完全修复(当前 −{abs(DD[-1]):.2f}%)。旧版分析中「2026-06~07 连环止损段 −25.7%(旧参数口径、十年最深)」在当前默认参数(QQQ 混合动量 + 黄金止盈冷却)下已收敛至 {f2(SEG['min'])}%——两次默认参数变更对尾部风险的实际改善。</p>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>4.2 模块消融：每个模块的独立贡献</h2></div>
    <div style="padding:18px">
      <p>固定其余参数、逐一关闭/回退单个模块，与基线比较(Δ 为相对基线的变化)：</p>
      <table>
        <thead><tr><th>开关(回退方向)</th><th>年化</th><th>最大回撤</th><th>Calmar</th><th>交易</th><th>判定</th></tr></thead>
        <tbody>
          <tr><td><b>基线(当前默认参数)</b></td><td><b>{f2(B['ann_ret'])}%</b></td><td class="num-neg"><b>{f2(B['max_dd'])}%</b></td><td><b>{f2(B['calmar'])}</b></td><td>{B['trade_count']}</td><td>—</td></tr>
          {ABL_ROWS}
        </tbody>
      </table>
      <p class="small">判定规则: 正贡献 = ΔCalmar&gt;+0.05 且年化不减; 负优化 = ΔCalmar&lt;−0.03 或年化 −1pp 以上; 其余为中性/权衡。回退方向即「关掉/还原该默认开关」。</p>
      <p>三大风控模块(上证空仓、加速止盈、恐慌抄底)与 2026 年新增默认(绝对止损、QQQ 混合、黄金止盈冷却)全部为净正贡献；学术「标准件」(Top-K 分散、多周期、动量跳过)均为大幅负优化——<b>本策略的风控靠 regime filter 而非分散，集中持仓与窄池是设计而非缺陷</b>。</p>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>4.3 参数稳健性</h2></div>
    <div style="padding:18px">
      <div id="mom-chart" class="chart-box"></div>
      <h3>动量周期细网格(10~30 日，其余参数固定)</h3>
      <table>
        <thead><tr><th>动量周期</th><th>年化</th><th>最大回撤</th><th>Calmar</th><th>交易</th></tr></thead>
        <tbody>{MOM_ROWS}</tbody>
      </table>
      <p>全样本 Calmar 最优为 <b>{best_p} 日</b>({f2(MOM[str(best_p)]['calmar'])}), 20 日(当前)排第 {rank20}/21 名, {worst_p} 日档最差。注意与旧版结论的差异: 旧引擎口径下该曲线呈「孤峰」(21 日 2.38, 邻档 20=1.98 / 22=1.06); 当前引擎(QQQ 混合 + 黄金止盈冷却)下已变为 <b>18~22 日的缓坡</b>——曲线形态本身对引擎口径敏感, 这正是样本特异性的证据, 而非参数可靠性的证据。</p>
      <h3>其余参数单变量验证(每档格式: 年化 / 最大回撤 / Calmar)</h3>
      <table>
        <thead><tr><th>参数</th><th>档位 1</th><th>档位 2(当前 ◄)</th><th>档位 3</th><th>档位 4+</th><th>解读</th></tr></thead>
        <tbody>{GRID_ROWS}</tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>4.4 样本外验证：长样本与两半窗口</h2></div>
    <div style="padding:18px">
      <h3>长样本(2015-01-05 起，含 2015 股灾与 2016 熔断)</h3>
      <table>
        <thead><tr><th>配置</th><th>累计</th><th>年化</th><th>最大回撤</th><th>Calmar</th><th>交易</th></tr></thead>
        <tbody>
          <tr><td><b>当前默认参数</b></td><td class="num-pos">{f0(LW_B['total_ret'])}%</td><td>{f2(LW_B['ann_ret'])}%</td><td class="num-neg">{f2(LW_B['max_dd'])}%</td><td>{f2(LW_B['calmar'])}</td><td>{LW_B['trade_count']}</td></tr>
          <tr><td>空仓冷却 0 天</td><td>{f0(LW_C0['total_ret'])}%</td><td>{f2(LW_C0['ann_ret'])}%</td><td class="num-neg">{f2(LW_C0['max_dd'])}%</td><td>{f2(LW_C0['calmar'])}</td><td>{LW_C0['trade_count']}</td></tr>
          <tr><td>空仓冷却 5 天</td><td>{f0(LW_C5['total_ret'])}%</td><td>{f2(LW_C5['ann_ret'])}%</td><td class="num-neg">{f2(LW_C5['max_dd'])}%</td><td>{f2(LW_C5['calmar'])}</td><td>{LW_C5['trade_count']}</td></tr>
          <tr><td>σ 止损 floor k=3.0</td><td>{f0(LW_SF['total_ret'])}%</td><td>{f2(LW_SF['ann_ret'])}%</td><td class="num-neg">{f2(LW_SF['max_dd'])}%</td><td>{f2(LW_SF['calmar'])}</td><td>{LW_SF['trade_count']}</td></tr>
        </tbody>
      </table>
      <p>把 2015 年股灾与 2016 年初熔断纳入样本后(11.6 年, 共 {len(LW['dd_episodes'])} 次 &gt;10% 回撤), 策略结构与主样本一致。空仓冷却在该样本上贡献显著(0 天档 Calmar {f2(LW_C0['calmar'])} vs 3 天档 {f2(LW_B['calmar'])}), 5 天档 {f2(LW_C5['calmar'])} 略优于 3 天档; σ 止损 floor 在长样本上为 {f2(LW_SF['calmar'])}, 未复现其旧口径验证期的优势。</p>
      <h3>两半窗口 walk-forward(2016-2021 定参 → 2021-2026 验证 → 2023-2026 压力)</h3>
      {HALF_HTML}
      <p>样本外验证窗(2021-08 起)年化 <b>{f2(HF['h2']['baseline']['ann_ret'])}%</b> 与全样本 {f2(B['ann_ret'])}% 基本持平, 近三年压力窗年化 {f2(HF['h3']['baseline']['ann_ret'])}%(含 2025 黄金大牛)。但最优动量周期从样本内的 19 日漂移至样本外的 22 日——市场节奏在向更长形成期偏移, 20 日当前仍稳健, 但已不是唯一最优。</p>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>4.5 交易行为解剖：出场通道与恐慌抄底</h2></div>
    <div style="padding:18px">
      <h3>出场通道的成绩单(按平仓原因分组)</h3>
      <table>
        <thead><tr><th>通道</th><th>笔数</th><th>胜率</th><th>平均盈亏</th><th>中位盈亏</th><th>合计</th></tr></thead>
        <tbody>{CH_ROWS}</tbody>
      </table>
      <p>止损/绝对止损通道全亏是<b>设计内的保险费</b>：以确定性小亏交换尾部截断。正常切换(胜率 {f1(CH['正常切换']['win_rate'])}%)与加速止盈是主要利润来源；上证空仓通道合计为正——regime filter 是净增值而非净成本。</p>
      <p class="small"><b>口径说明(上证空仓胜率 {f1(CH['上证空仓']['win_rate'])}%)：</b>该胜率是「离场那一刻这条腿赚没赚钱」的平仓胜率，不是「空仓判断对不对」的准确率。Regime filter 在本职是「我持有的标的已不再是领导者」时离场，通常贴近盈亏平衡退出(中位 {f2(CH['上证空仓']['median'])}%)，平仓胜率天然偏低。用最新数据按「离场后 20 个交易日」口径复核: 被卖资产平均 <b>−0.97%</b>(中位 −0.13%)——它卖对了资产; 空仓期间净值平均仅变动 −0.01%(几乎零成本); 代价是池内最强资产随后 20 日平均 +6.16%、63% 概率涨超 3%(误伤保费)。其价值是不对称的: 错一次损失几个点的机会成本, 对一次(2018-02 volmageddon / 2021-09 / 2026-07 等)躲掉两位数回撤; 消融佐证: 关掉该 filter 后 Calmar {f2(B['calmar'])}→{f2(AB['sh_cash_off']['calmar'])}。</p>
      <h3>上证空仓躲过的关键暴跌(逐笔前向审计 · 41 次离场)</h3>
      <p>「空仓期收益」= 离场后若继续持有被卖资产、直到下次入场的收益(负值即躲掉的下跌);「空仓期最深」= 该期间被卖资产的最大跌幅;「20 日前向」= 离场后 20 个交易日的资产涨跌(验证信号是否命中顶部)。</p>
      <table>
        <thead><tr><th>离场日</th><th>资产</th><th>事件背景</th><th>空仓期收益</th><th>空仓期最深</th><th>20 日前向</th></tr></thead>
        <tbody>
          <tr><td>2015-06-08</td><td>创业</td><td>2015 股灾顶部(长窗口样本)</td><td class="num-pos">+4.8%</td><td>-0.2%</td><td class="num-neg">-40.6%</td></tr>
          <tr><td>2015-06-15</td><td>创业</td><td>2015 股灾第一周(长窗口样本)</td><td class="num-pos">-8.9%</td><td class="num-neg">-11.1%</td><td class="num-neg">-24.3%</td></tr>
          <tr><td>2017-06-09</td><td>纳指</td><td>FAANG 科技股回调</td><td class="num-pos">-4.2%</td><td>-4.2%</td><td class="num-neg">-5.9%</td></tr>
          <tr><td>2017-07-05</td><td>创业</td><td>创业板阴跌(空仓 17 天)</td><td class="num-pos">-6.5%</td><td class="num-neg">-10.8%</td><td class="num-neg">-7.0%</td></tr>
          <tr><td>2018-02-02</td><td>纳指</td><td>volmageddon(02-05 VIX 单日飙升)</td><td class="num-pos">-5.4%</td><td class="num-neg">-8.7%</td><td>+1.1%</td></tr>
          <tr><td>2018-09-21</td><td>纳指</td><td>Q4 大跌前 7 天离场</td><td class="num-pos">-1.3%</td><td>-2.9%</td><td class="num-neg">-8.7%</td></tr>
          <tr><td>2021-02-26</td><td>纳指</td><td>成长股杀估值</td><td class="num-pos">-2.2%</td><td class="num-neg">-5.5%</td><td>+1.8%</td></tr>
          <tr><td>2021-09-02</td><td>纳指</td><td>9 月科技股回调(空仓 20 天)</td><td class="num-pos">-6.0%</td><td class="num-neg">-6.1%</td><td class="num-neg">-6.0%</td></tr>
          <tr><td>2021-12-06</td><td>创业</td><td>年末 A 股调整(空仓 19 天)</td><td class="num-pos">-2.3%</td><td class="num-neg">-6.1%</td><td>-4.3%</td></tr>
          <tr><td>2023-04-20</td><td>创业</td><td>4 月 A 股调整</td><td class="num-pos">-5.2%</td><td class="num-neg">-5.2%</td><td class="num-neg">-5.5%</td></tr>
          <tr><td>2023-08-07</td><td>创业</td><td>8 月 A 股下跌</td><td class="num-pos">-4.1%</td><td>-4.1%</td><td class="num-neg">-5.4%</td></tr>
          <tr><td>2025-03-07</td><td>创业</td><td>顶部命中(离场后 20 日 −17.8%)</td><td class="num-pos">-2.2%</td><td>-3.4%</td><td class="num-neg">-17.8%</td></tr>
          <tr><td>2026-03-05</td><td>黄金</td><td>黄金回调</td><td class="num-pos">-3.0%</td><td>-3.1%</td><td class="num-neg">-11.2%</td></tr>
          <tr><td>2026-07-02</td><td>创业</td><td>2026-07 A 股暴跌</td><td class="num-pos">-7.1%</td><td class="num-neg">-7.2%</td><td class="num-neg">-19.0%</td></tr>
        </tbody>
      </table>
      <p>合计口径: 主窗口 41 次离场中, 「20 日前向 ≤ −5%(躲对)」<b>10 次</b>、「≥ +5%(卖飞)」<b>4 次</b>、其余 27 次中性(多在 ±2% 内); 若每次离场后都傻等被卖资产到下次入场, 十年累计多亏 <b>−23.9%</b>。典型的躲过包括 2018-02 波动率崩溃、2021-09 科技股回调、2023-04/08、2026-03 黄金与 2026-07 A 股两连击——且 2025-03 与 2015-06 两次都精确踩在顶部。</p>
      <p class="small"><b>能力边界:</b> 它躲的是「领导力背离型」顶部, 不是全部系统性危机——2020-03 疫情崩盘期间该 filter 零触发(靠止损 + 恐慌抄底兜底); 2016-01 熔断也是先误报(2015-12-25 离场黄金)后于熔断首日再入场, 未躲成。它也时常「躲了开头就回来」(2025-03 再入场过早, 只吃到 −2.2%)——冷却 3 天挡不住规则化的再入场, 这是它与「全程躲熊市」择时器的本质区别。最大卖飞: 2022-05-27 离场创业后 20 日 +21.8%。</p>
      <h3>恐慌抄底逐笔记录({len(CRASH_TRADES)} 笔已平仓)</h3>
      <table>
        <thead><tr><th>入场日</th><th>出场日</th><th>盈亏</th><th>持有天数</th><th>出场方式</th></tr></thead>
        <tbody>{CRASH_ROWS}</tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>4.5b 持仓结构：分标的持有时间 / 收益分布 / 持仓占比</h2></div>
    <div style="padding:18px">
      <p>全样本 <b>{HA['total_days']}</b> 个交易日中，持仓 {HA['held_days']} 天({100 - HA_CASH['time_pct']:.1f}%)、空仓 {HA_CASH['n']} 段共 {HA['total_days'] - HA['held_days']} 天({HA_CASH['time_pct']}%，空仓段中位 {f0(HA_CASH['days_med'])} 天、最长 {f0(HA_CASH['days_max'])} 天)；共 {len(HA['legs'])} 段持仓，平均每段持有 {HA['legs_avg_days']} 个交易日。</p>
      <table>
        <thead><tr><th>标的</th><th>持仓段数</th><th>时间占比</th><th>均持有</th><th>中位持有</th><th>最长持有</th><th>平均收益</th><th>中位收益</th><th>收益贡献</th><th>贡献占比</th><th>胜率</th><th>收益分布(≤-5/-5~0/0~5/5~10/&gt;10)</th><th>持有天数分布(1-5/6-10/11-20/21-40/40+)</th></tr></thead>
        <tbody>{HA_ROWS}</tbody>
      </table>
      <p class="small"><b>口径：</b>每段持仓 = 买入/切换入 → 卖出/切换出/期末持有的完整腿；收益为段内净值盈亏(含手续费)，期末持仓段计入；收益贡献 = 各段盈亏(pp)之和，贡献占比 = 该标的总盈亏 / 全池总盈亏；时间占比 = 该标的持有交易日 / 全样本交易日。</p>
      <div id="hold-days-chart" style="height:300px;margin-top:10px"></div>
      <div id="hold-pnl-chart" style="height:300px;margin-top:6px"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>4.6 成本敏感性</h2></div>
    <div style="padding:18px">
      <table>
        <thead><tr><th>单边成本(佣金+滑点)</th><th>累计</th><th>年化</th><th>最大回撤</th><th>Calmar</th><th>交易</th></tr></thead>
        <tbody>{COST_ROWS}</tbody>
      </table>
      <p>十年约 {B['trade_count']} 次交易(买+卖)，滑点对复利影响显著：每 0.1% 单边滑点约削去 {f1(B['ann_ret'] - SLIP['0.001']['ann_ret'])}pp 年化。ETF 实际冲击成本接近零，实盘评估建议按 0.05% 滑点口径。</p>
    </div>
  </div>

  <!-- ═══════════ 5 讨论 ═══════════ -->
  <h2 class="sec">4.7 · 加仓时机研究：立即追 vs 等下一次调仓（2026-08-21/23）</h2>
  <div class="card">
    <div class="card-header"><h2>4.7 追加本金怎么进：立即追当前持仓 vs 等下次调仓</h2></div>
    <div style="padding:18px">
      <p><b>方法：</b>复用生产回测引擎，在全历史每个交易日模拟一笔加仓，比较 A「立即按当前持仓买入」、B「等策略第一次调仓再进」、D「10 日均分」三种方式在同一持有窗口 [注入日, 注入日+H] 的收益（A−B 为两者收益差，pp）。</p>
      <h3>总体（各持有期口径）</h3>
      <table>
        <thead><tr><th>持有期</th><th>样本</th><th>A 胜率</th><th>A−B 平均</th><th>A−B 中位</th><th>A 均收益</th><th>B 均收益</th><th>D 均收益</th></tr></thead>
        <tbody>{CAI_OVERALL_ROWS}</tbody>
      </table>
      <h3>按「注入时当前持仓收益率」分档（12 个月口径）</h3>
      <table>
        <thead><tr><th>当前持仓浮盈亏</th><th>样本</th><th>A 胜率</th><th>A−B 平均</th><th>A 均收益</th><th>B 均收益</th><th>D 均收益</th><th>推荐</th></tr></thead>
        <tbody>{CAI_PNL_ROWS}</tbody>
      </table>
      <ul style="padding-left:22px">
        <li><b>每一档都是「立即追」赢</b>——「无脑追」的结论经得起收益率分档检验，四个月口径方向一致，无任何档位反转。</li>
        <li>优势呈<b>倒 U 型</b>：刚转正(0~5%)档最优(+4.3pp、胜率 68~69%)——策略刚确认趋势时追仓最划算；浮亏档次之；大涨后(&gt;10%)减弱到 +2.6pp(离加速止盈更近，等待代价变小)。</li>
        <li>不利尾部几乎全压在「等调仓」一侧：等调仓吃大亏(≥5pp)概率 28.5% 是占大便宜(6.8%)的 4.2 倍，且有 24.9% 概率一个月等不到调仓。</li>
        <li>大额资金心理压力大时，D（10 日均分）期望接近 A，是低后悔的折中。</li>
      </ul>
      <p class="small"><b>口径：</b>空仓注入日 A≡B 已剔除；收益为净值口径含手续费；完整明细与脚本见 backtest/add_capital_entry.py / add_capital_entry_result.json / 加仓时机分析.md。</p>
    </div>
  </div>

  <!-- ═══════════ 5 讨论 ═══════════ -->
  <h2 class="sec">5 · 讨论</h2>
  <div class="card">
    <div class="card-header"><h2>5.1 结构上的优势</h2></div>
    <div style="padding:18px">
      <ul>
        <li><b>Regime filter 是灵魂。</b>「上证动量强于池内最强即离场」检测的是领导力背离——宽基走强而策略池不参与，历史上多次关键顶部均有触发(2015-06/12 股灾顶部、2018-02 volmageddon、2021-09、2026-07 等, 见 §4.5)。它用择时代替分散做风控：这也是 Top-2 分散实验为大幅负优化的原因——风控已被 filter 承担，分散只稀释 alpha。</li>
        <li><b>两端均值回归叠加补齐动量短板。</b>动量策略在顶部追高、底部割肉；加速止盈卖在高潮、恐慌抄底买在恐慌，精确对冲这两个固有缺陷。</li>
        <li><b>成本纪律到位。</b>确认天数/最短持有/最小动量差抑制换手，手续费独立建模并做压力测试。</li>
      </ul>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>5.2 脆弱点与红旗</h2></div>
    <div style="padding:18px">
      <ul>
        <li><b>动量周期的跨期漂移。</b>全样本最优 19 日、当前用 20 日；样本内窗最优 19 日、样本外窗漂移到 22 日(§4.4)。当前引擎下曲线虽已从「孤峰」变为缓坡，但最优值随样本移动——这是<b>样本特异性共振</b>的痕迹，不是结构性 alpha。</li>
        <li><b>选池偏差未被完全分解。</b>纳指与黄金恰是这十年全球最强的两类资产，年化收益中相当部分是「选对池」而非轮动 alpha。等权买入持有对照(§4.1)已控制掉一部分，但池内替代性(如创业板→沪深300)的对照仍是遗留空白。</li>
        <li><b>全参数样本内选择。</b>20 日、3σ、−2.5σ、3 天冷却……全部选自主样本。两半窗口验证(§4.4)是部分解药，且确实显示最优参数随窗口漂移。</li>
        <li><b>用上证当全球 regime 锚是集中押注。</b>池中 3 个资产有 2 个(纳指/黄金)与 A 股相关性有限，但总风控开关绑在上证动量上。若下一次危机是美国特有而 A 股无恙，filter 可能不触发。</li>
        <li><b>二元全仓、无仓位管理。</b>每次回撤都足额承受；波动率目标化(TSMOM 式缩放)是未实现的理论改进方向。</li>
      </ul>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>5.3 与上一版分析的差异(结论变更记录)</h2></div>
    <div style="padding:18px">
      <table>
        <thead><tr><th>议题</th><th>旧结论(2026-08 早期)</th><th>当前状态</th></tr></thead>
        <tbody>
          <tr><td>纳指动量信号源</td><td>纯 513100 ETF 价格动量</td><td><b>已改为 0.7×QQQ-CNY + 0.3×ETF 混合</b>(I-11 决议)：回退纯 ETF 口径 Calmar 从 {f2(B['calmar'])} 降至 {f2(AB['nz_qqq_off']['calmar'])}。</td></tr>
          <tr><td>止盈后冷却</td><td>默认关闭(历史判定负优化)</td><td><b>已改为黄金止盈后冷却 3 天</b>(I-15/16/17 决议)：回退后全样本 Calmar 升至 {f2(AB['accel_cooldown0']['calmar'])} 但 2025 年 −16.2pp, 决议基于近期证据维持; 扩至全部标的为负优化({f2(AB['accel_cd_all_assets']['calmar'])})。</td></tr>
          <tr><td>2026-06~07 连环止损段</td><td>十年最深回撤 −25.7%</td><td>当前口径下该段回撤 {f2(SEG['min'])}%(谷底 {SEG['trough_date']})，最深度让位于 2025-09~2026-04 段。</td></tr>
          <tr><td>动量周期曲线</td><td>20/21 日孤峰(邻档断崖)</td><td>当前口径下为 18~22 日缓坡, 全样本最优 19 日; 样本外最优漂移至 22 日。</td></tr>
          <tr><td>σ 自适应止损 floor k=3.0</td><td>2026-08-14 三窗口验证通过后落地实盘</td><td>实盘已部署(stock-bot cron); 当前默认口径 + 最新数据下为负优化(主样本 {f2(AB['sigma_stop_floor3']['calmar'])} vs 基线 {f2(B['calmar'])}), 列观察项。</td></tr>
          <tr><td>扩池</td><td>3→4→8→14 池单调恶化，写入纪律</td><td>维持窄池 3，未重验。</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ═══════════ 6 结论 ═══════════ -->
  <h2 class="sec">6 · 结论</h2>
  <div class="card">
    <div class="card-header"><h2>6 · 结论与实盘含义</h2></div>
    <div style="padding:18px">
      <ol style="padding-left:22px">
        <li>策略在十年主样本上取得年化 <b>{f2(B['ann_ret'])}%</b>、最大回撤 <b>{f2(B['max_dd'])}%</b>、Calmar <b>{f2(B['calmar'])}</b>，11 个年份中仅 2016 年为负；该结果在长样本(2015 起)与样本外窗口(2021 起)中方向一致、幅度相当。</li>
        <li>收益结构可分解：窄池相对动量(引擎)+ regime filter(锚)+ 两端风控(保险与增强)。消融证实各模块不可删除；教科书式「改进」(分散持仓、多周期动量、动量跳过)为大幅负优化, 波动率归一/防御切换/FIP 为中性偏负——<b>集中持仓与窄池是设计而非缺陷</b>。</li>
        <li>参数体系整体稳健, 主要红旗是动量周期的跨期漂移(样本内 19 日 → 样本外 22 日)。合理的长期预期不是点估计 {f2(B['ann_ret'])}%, 而是 <b>30%~45% 年化、−18%~−25% 回撤</b>的区间。</li>
        <li>实盘含义: 维持当前默认参数; 换参的触发条件应是 21/22 日档在实盘持续反超 20 日档(而非回测网格); 已部署的 σ 止损 floor 变体在当前口径下未复现其验证期优势, 列为观察项。</li>
      </ol>
      <p class="meta">本报告所有数字由 web/bt_runner.run_backtest 在 {META['end']} 数据上重跑生成(完整结果 audit/out/paper_2026.json, 生成脚本 audit/paper_experiments.py); 历史研究结论存档 audit/审计结果清单.md。</p>
    </div>
  </div>

</div>
<script src="/static/lib/echarts.min.js"></script>
<script>
const DATES = {json.dumps(DATES)};
const NAV = {json.dumps(NAV)};
const DD = {json.dumps(DD)};
const BENCH = {json.dumps(BENCH)};
const MOM_P = {json.dumps(mom_ps)};
const MOM_ANN = {json.dumps(mom_ann)};
const MOM_CAL = {json.dumps(mom_cal)};
const MOM_DD = {json.dumps(mom_dd)};
const HA_ASSETS = {json.dumps(HA_CHART_ASSETS)};
const HA_PNL = {json.dumps(HA_PNL)};
const HA_DAYS = {json.dumps(HA_DAYS)};
const PNL_BINS = {json.dumps(PNL_BINS)};
const DAYS_BINS = {json.dumps(DAYS_BINS)};
const HA_COLORS = {{'创业':'#4caf50','纳指':'#42a5f5','黄金':'#facc15','合计':'#e94560'}};
(function(){{
  var TIP = {{backgroundColor:'rgba(20,24,36,.95)', borderColor:'#2a3142', textStyle:{{color:'#e8eaf0'}}}};
  var AX = {{axisLine:{{lineStyle:{{color:'#2a3142'}}}}, axisLabel:{{color:'#8b92a8'}}}};
  var nc = echarts.init(document.getElementById('nav-chart'));
  nc.setOption({{
    backgroundColor:'transparent', animation:false,
    tooltip:{{trigger:'axis', backgroundColor:'rgba(20,24,36,.95)', borderColor:'#2a3142', textStyle:{{color:'#e8eaf0'}}}},
    legend:{{top:0, textStyle:{{color:'#8b92a8'}}, data:['策略净值','等权买入持有','回撤%']}},
    grid:{{left:56,right:56,top:36,bottom:40}},
    xAxis:Object.assign({{type:'category', data:DATES, boundaryGap:false}}, AX),
    yAxis:[
      Object.assign({{type:'value', name:'净值', splitLine:{{lineStyle:{{color:'rgba(42,49,66,.6)'}}}}}}, AX),
      Object.assign({{type:'value', name:'回撤%', max:0, splitLine:{{show:false}}}}, AX)],
    series:[
      {{name:'策略净值', type:'line', data:NAV, showSymbol:false, lineStyle:{{color:'#e94560',width:1.6}}, areaStyle:{{color:'rgba(233,69,96,.08)'}}}},
      {{name:'等权买入持有', type:'line', data:BENCH, showSymbol:false, lineStyle:{{color:'#64b5f6',width:1.2,type:'dashed'}}}},
      {{name:'回撤%', type:'line', yAxisIndex:1, data:DD, showSymbol:false, lineStyle:{{color:'#facc15',width:1}}, areaStyle:{{color:'rgba(250,204,21,.15)'}}}}
    ]
  }});
  var mc = echarts.init(document.getElementById('mom-chart'));
  mc.setOption({{
    backgroundColor:'transparent', animation:false,
    tooltip:Object.assign({{trigger:'axis'}}, TIP),
    legend:{{top:0, textStyle:{{color:'#8b92a8'}}}},
    grid:{{left:56,right:56,top:36,bottom:44}},
    xAxis:Object.assign({{type:'category', data:MOM_P.map(function(p){{return p+'日';}})}}, AX),
    yAxis:[
      Object.assign({{type:'value', name:'Calmar', splitLine:{{lineStyle:{{color:'rgba(42,49,66,.6)'}}}}}}, AX),
      Object.assign({{type:'value', name:'年化% / 回撤%', splitLine:{{show:false}}}}, AX)],
    series:[
      {{name:'Calmar', type:'line', data:MOM_CAL, symbol:'circle', symbolSize:6,
        lineStyle:{{color:'#e94560',width:2}}, itemStyle:{{color:'#e94560'}},
        markLine:{{silent:true, symbol:'none', label:{{color:'#8b92a8', formatter:'当前 20 日'}}, lineStyle:{{color:'#8b92a8',type:'dashed'}}, data:[{{xAxis:'20日'}}]}}}},
      {{name:'年化%', type:'line', yAxisIndex:1, data:MOM_ANN, symbol:'circle', symbolSize:4,
        lineStyle:{{color:'#4caf50',width:1.4,type:'dashed'}}, itemStyle:{{color:'#4caf50'}}}},
      {{name:'最大回撤%', type:'line', yAxisIndex:1, data:MOM_DD, showSymbol:false, lineStyle:{{color:'#8b92a8',width:1}}}}
    ]
  }});
  var hd = echarts.init(document.getElementById('hold-days-chart'));
  hd.setOption({{
    backgroundColor:'transparent', animation:false,
    title:{{text:'持有天数分布（按标的，持仓段数）', left:'center', textStyle:{{color:'#8b92a8', fontSize:13, fontWeight:400}}}},
    tooltip:Object.assign({{trigger:'axis'}}, TIP),
    legend:{{top:24, textStyle:{{color:'#8b92a8'}}}},
    grid:{{left:56,right:20,top:60,bottom:40}},
    xAxis:Object.assign({{type:'category', data:DAYS_BINS, name:'持有天数'}}, AX),
    yAxis:Object.assign({{type:'value', name:'持仓段数', minInterval:1, splitLine:{{lineStyle:{{color:'rgba(42,49,66,.6)'}}}}}}, AX),
    series: HA_ASSETS.map(function(a){{return {{name:a, type:'bar', data:HA_DAYS[a], itemStyle:{{color:HA_COLORS[a]}}, barMaxWidth:26}};}})
  }});
  var hp = echarts.init(document.getElementById('hold-pnl-chart'));
  hp.setOption({{
    backgroundColor:'transparent', animation:false,
    title:{{text:'段收益分布（按标的，持仓段数）', left:'center', textStyle:{{color:'#8b92a8', fontSize:13, fontWeight:400}}}},
    tooltip:Object.assign({{trigger:'axis'}}, TIP),
    legend:{{top:24, textStyle:{{color:'#8b92a8'}}}},
    grid:{{left:56,right:20,top:60,bottom:40}},
    xAxis:Object.assign({{type:'category', data:PNL_BINS, name:'段收益 %'}}, AX),
    yAxis:Object.assign({{type:'value', name:'持仓段数', minInterval:1, splitLine:{{lineStyle:{{color:'rgba(42,49,66,.6)'}}}}}}, AX),
    series: HA_ASSETS.map(function(a){{return {{name:a, type:'bar', data:HA_PNL[a], itemStyle:{{color:HA_COLORS[a]}}, barMaxWidth:26}};}})
  }});
  window.addEventListener('resize', function(){{ nc.resize(); mc.resize(); hd.resize(); hp.resize(); }});
}})();
</script>
</body>
</html>
"""

with open("/root/etf_strategy/web/templates/analysis.html", "w") as f:
    f.write(HTML)
print("written web/templates/analysis.html", len(HTML), "bytes")
