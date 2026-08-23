"""私密实盘对账: 手工维护的实盘买卖记录 → 实盘净值曲线 + 与策略同窗口对比

记账模型: 多标的分批仓位模型 ——
  每条记录: date / action(buy|sell) / asset / price(实际成交价) / pct(仓位%, 默认100) / note
  buy:  pct = 动用当前可用现金的百分比 (100=满仓买入)
  sell: pct = 卖出该标的当前持仓的百分比 (100=清仓该标的)
  允许同时持有多个标的; 同日多条记录按 先卖后买 执行。
  买入价→当日收盘、昨收→卖出价的损益如实计入净值。
数据文件: web/mytrack_data.json (已 gitignore)
"""
import os, json, datetime as _dt
import numpy as np
import pandas as pd

from data.feed import TushareFeed
from web.config_local import TUSHARE_TOKEN, MYTRACK_KEY  # noqa: F401  (key 在 app.py 用)
from web.bt_runner import ALL_CODES, run_backtest

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mytrack_data.json")
COMBOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mytrack_combos.json")
FEED = TushareFeed(TUSHARE_TOKEN, cache_dir="/root/backtest/cache/etf_momentum")

# ── 策略组合对比（可调标的池/参数，随对账页一起画图）────────────────────
DEFAULT_COMBOS = [
    {"name": "从策略(纯QQQ)", "universe": ["纳指", "黄金", "创业"],
     "params": {"nz_mom_qqq_on": True, "nz_mom_qqq_w": 1.0}},
    {"name": "+科创50", "universe": ["纳指", "黄金", "创业", "科创50"], "params": {}},
    {"name": "黄金+纳指", "universe": ["黄金", "纳指"], "params": {}},
]

# 组合可调参数的校验规则（白名单）
COMBO_PARAM_RULES = {
    "sh_cash": bool, "mom_period": int, "nz_mom_qqq_on": bool,
    "nz_mom_qqq_w": float, "accel_cooldown": int, "accel_cooldown_gold_only": bool,
    "stop_loss": float, "stop_sigma": float, "stop_sigma_floor": bool,
    "min_hold": int, "min_gap": float, "exception_gap": float,
    "confirm_days": int, "sh_cash_cooldown": int,
}


def load_combos():
    """组合定义列表（无文件时给默认）"""
    if not os.path.exists(COMBOS_FILE):
        return {"combos": [dict(c) for c in DEFAULT_COMBOS]}
    try:
        with open(COMBOS_FILE) as f:
            return json.load(f)
    except Exception:
        return {"combos": [dict(c) for c in DEFAULT_COMBOS]}


def validate_combos(combos):
    """返回 (err, cleaned)。err 为 None 表示通过"""
    all_codes = set(ALL_CODES.keys())
    if not isinstance(combos, list) or not (1 <= len(combos) <= 10):
        return "组合数量需为 1~10 个", None
    clean = []
    for c in combos:
        if not isinstance(c, dict):
            return "组合格式错误", None
        name = str(c.get("name", "")).strip()[:20]
        uni = c.get("universe") or []
        if not name:
            return "组合名不能为空", None
        if not isinstance(uni, list) or not (2 <= len(uni) <= 8):
            return "组合「%s」标的需 2~8 个" % name, None
        bad = [u for u in uni if u not in all_codes]
        if bad:
            return "组合「%s」含未知标的: %s" % (name, bad), None
        params = {}
        for k, v in (c.get("params") or {}).items():
            if k not in COMBO_PARAM_RULES:
                continue
            try:
                typ = COMBO_PARAM_RULES[k]
                if typ is bool:
                    params[k] = v if isinstance(v, bool) else bool(v)
                elif typ is int:
                    params[k] = int(v)
                else:
                    params[k] = float(v)
            except Exception:
                continue
        # QQQ 权重限制在 [0,1]（0=纯ETF, 0.7=主策略口径, 1=纯QQQ）
        if "nz_mom_qqq_w" in params:
            params["nz_mom_qqq_w"] = round(min(max(params["nz_mom_qqq_w"], 0.0), 1.0), 2)
        # 冷却天数限制 0~10
        if "accel_cooldown" in params:
            params["accel_cooldown"] = min(max(params["accel_cooldown"], 0), 10)
        # 动量周期限制 10~60
        if "mom_period" in params:
            params["mom_period"] = min(max(params["mom_period"], 10), 60)
        clean.append({"name": name, "universe": list(uni), "params": params})
    return None, clean


def save_combos(combos):
    tmp = COMBOS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"combos": combos}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, COMBOS_FILE)


def load_records():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE) as f:
        return json.load(f).get("records", [])


def load_book():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE) as f:
        return json.load(f)


def save_records(records, initial_cash=None):
    book = load_book()
    if initial_cash is None:
        initial_cash = book.get("initial_cash")
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"initial_cash": initial_cash, "records": records},
                  f, ensure_ascii=False, indent=1)
    os.replace(tmp, DATA_FILE)


def _close_map(asset):
    """某标的 交易日->收盘价 (含缓存)"""
    code = ALL_CODES[asset]
    df = FEED.get_dataframe([code], "20150101", _dt.date.today().strftime("%Y%m%d"))[code]
    return dict(zip(df["trade_date"], df["close"]))


def _norm(records):
    return sorted(records, key=lambda r: (r["date"], 0 if r["action"] == "sell" else 1))


def _simulate(records, cmaps=None, need_cmaps=False, initial_cash=None):
    """逐日模拟。返回 (err, state) —— err 为 None 时 state 含每日净值等。
    cmaps: {asset: {Timestamp: close}}; need_cmaps=False 时只做资金流校验(不取价格)。
    initial_cash: 初始资金(元); 缺省=首笔买入金额(假设首笔满仓)。"""
    if cmaps is None:
        cmaps = {}
    if need_cmaps:
        for r in records:
            a = r.get("asset")
            if a in ALL_CODES and a not in cmaps:
                cmaps[a] = _close_map(a)
    if initial_cash is None:
        initial_cash = next((float(r["amount"]) for r in records
                             if r["action"] == "buy" and r.get("amount")), 1.0)
    cash, holds = float(initial_cash), {}  # holds: asset -> {"units": float, "lots": [(date,price,units)]}
    daily = []                # (date_str, nav)
    closed_lots, marks = [], []
    all_dates = set()
    for cm in cmaps.values():
        all_dates |= set(cm.keys())
    cal = sorted(all_dates)
    if records:
        cal = [d for d in cal if str(d)[:10] >= records[0]["date"]]
    rec_by_date = {}
    for r in records:
        rec_by_date.setdefault(r["date"], []).append(r)

    for dt in cal:
        d = str(dt)[:10]
        for r in rec_by_date.get(d, []):
            a, p = r["asset"], float(r["price"])
            pct = float(r.get("pct", 100))
            amt = r.get("amount")          # 成交额(元), 优先级高于 pct
            if r["action"] == "sell":
                h = holds.get(a)
                if not h or h["units"] <= 1e-9:
                    return f"{d} 卖出 {a}: 当前无该标的持仓", None
                u = (float(amt) / p) if amt else h["units"] * pct / 100.0
                if u > h["units"] * 1.005:
                    return f"{d} 卖出 {a}: 卖出量超过持仓", None
                u = min(u, h["units"])
                cash += u * p
                # FIFO 闭合 lot
                left = u
                while left > 1e-9 and h["lots"]:
                    ld, lp, lu = h["lots"][0]
                    take = min(lu, left)
                    closed_lots.append({"asset": a, "entry_date": ld, "entry_price": lp,
                                        "exit_date": d, "exit_price": p, "units": take,
                                        "note": r.get("note", "")})
                    left -= take
                    if take >= lu - 1e-9: h["lots"].pop(0)
                    else: h["lots"][0] = (ld, lp, lu - take)
                h["units"] -= u
                if h["units"] <= 1e-9: holds.pop(a)
                marks.append({"date": d, "kind": "sell", "label": f"卖{a}"})
            else:
                spend = float(amt) if amt else cash * pct / 100.0
                if spend > cash * 1.005 or spend <= 0:
                    return f"{d} 买入 {a}: 可用现金不足(剩余 {cash:.4f})", None
                spend = min(spend, cash)
                cash -= spend
                h = holds.setdefault(a, {"units": 0.0, "lots": []})
                h["units"] += spend / p
                h["lots"].append((d, p, spend / p))
                marks.append({"date": d, "kind": "buy", "label": f"买{a}"})
        nav = cash
        if need_cmaps:
            for a, h in holds.items():
                cm = cmaps[a]
                if dt in cm: nav += h["units"] * cm[dt]
                else:  # 停牌: 用最近可用收盘价
                    prev = [k for k in cm.keys() if k <= dt]
                    if prev: nav += h["units"] * cm[prev[-1]]
        daily.append((d, nav, dict(holds), cash))
    return None, {"daily": daily, "closed_lots": closed_lots, "marks": marks,
                  "holds": holds, "cash": cash, "cmaps": cmaps}


def validate_records(records, initial_cash=None):
    """保存前校验: 字段合法 / 交易日 / 价格偏离收盘 ≤8% / 现金流可行"""
    if not isinstance(records, list):
        return "记录格式错误", None
    records = _norm(records)
    cmaps = {}
    for i, r in enumerate(records):
        try:
            _dt.datetime.strptime(r["date"], "%Y-%m-%d")
        except Exception:
            return f"第{i+1}条: 日期格式须为 YYYY-MM-DD", None
        if r["action"] not in ("buy", "sell"):
            return f"第{i+1}条: action 须为 buy/sell", None
        if r.get("asset") not in ALL_CODES:
            return f"第{i+1}条: 未知标的 {r.get('asset')}", None
        try:
            price = float(r["price"]); assert price > 0
        except Exception:
            return f"第{i+1}条: 价格须为正数", None
        pct = float(r.get("pct", 100))
        if not (0 < pct <= 100):
            return f"第{i+1}条: 仓位% 须在 (0, 100]", None
        if r.get("amount") is not None:
            try:
                assert float(r["amount"]) > 0
            except Exception:
                return f"第{i+1}条: 成交额须为正数", None
        a = r["asset"]
        if a not in cmaps:
            cmaps[a] = _close_map(a)
        d = pd.Timestamp(r["date"])
        if d not in cmaps[a]:
            near = min(cmaps[a].keys(), key=lambda k: abs((k - d).days))
            return f"第{i+1}条: {r['date']} 不是交易日(最近交易日 {str(near)[:10]})", None
        dev = (price / cmaps[a][d] - 1) * 100
        if abs(dev) > 8:
            return (f"第{i+1}条: {r['date']} {a} 记录价 {price} 与当日收盘 "
                    f"{cmaps[a][d]:.3f} 偏离 {dev:+.1f}%, 疑似笔误(>8% 拒存)"), None
    if initial_cash is None:
        initial_cash = load_book().get("initial_cash")
    err, _ = _simulate(records, cmaps, need_cmaps=True, initial_cash=initial_cash)
    if err:
        return err, None
    return None, records


def _metrics(nav):
    nav = np.array(nav, dtype=float)
    if len(nav) < 2:
        return {"total_ret": 0, "ann": None, "mdd": 0}
    cm = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - cm) / cm * 100))
    years = max(len(nav) / 244, 0.01)
    total = (nav[-1] - 1) * 100
    ann = ((nav[-1] ** (1 / years)) - 1) * 100 if (nav[-1] > 0 and len(nav) >= 60) else None
    return {"total_ret": round(total, 2), "ann": (round(ann, 2) if ann is not None else None),
            "mdd": round(mdd, 2)}


# ── 策略对账基准的时间线 ────────────────────────────────────────────────
# ── 策略对账基准 ────────────────────────────────────────────────────────
# 用户决议 (2026-08-20): 对账基准 = 当前默认口径的策略（策略调整过就按调整过的来），
# 不做"当时规则"的点时间回放——避免自欺欺人。默认值变化直接反映到整条曲线。


def strat_run(start_date, end_date, universe="纳指,黄金,创业", extra=None):
    """策略回放（当前默认口径; 组合对比可加 extra 参数覆盖）。
    返回 (dates, nav) 或 (None, None)；strat_run.last_pos 记录期末持仓（实时尾部用）。"""
    bt = run_backtest({"universe": universe,
                       "start_date": start_date, "end_date": end_date,
                       **(extra or {})})
    if "error" in bt or not bt.get("nav"):
        return None, None
    pos = None
    try:
        pl = bt.get("position_labels", [])
        ps = bt.get("position_series", [])
        if ps and len(ps) == len(bt["dates"]):
            idx = int(ps[-1])
            # position_series: -1=空仓, 0..n-1 对应 position_labels[1:]
            if 0 <= idx < len(pl) - 1:
                pos = pl[idx + 1]
    except Exception:
        pos = None
    strat_run.last_pos = pos
    return list(bt["dates"]), [float(v) for v in bt["nav"]]


strat_run.last_pos = None


# 基准指数 → 腾讯行情代码（对账页指数曲线尾部实时用; 科创50 基准用同名 ETF 近似）
IDX_TENCENT = {"上证指数": "sh000001", "创业板指数": "sz399006",
               "深证成指": "sz399001"}

# 外盘/外汇（对账页实时展示用）
SPECIAL_TENCENT = {"QQQ": "usQQQ", "USDCNY": "whUSDCNY"}

# 指数 Tushare 代码 (σ20 计算用)
IDX_CODES = {"上证指数": "000001.SH", "创业板指数": "399006.SZ",
             "深证成指": "399001.SZ", "科创50": "000688.SH"}


def _vol20_of_code(code):
    """从缓存直接读某代码的 20 日波动率(日%, 复权价口径); 不触发 Tushare 拉取"""
    try:
        path = os.path.join(FEED.cache_dir, code.replace(".", "_") + ".csv")
        df = pd.read_csv(path, usecols=["close", "adj_factor"])
        adj = df["close"] * df["adj_factor"] / df["adj_factor"].iloc[-1]
        c = adj.tail(21).values
        if len(c) < 6:
            return None
        r = np.diff(c) / c[:-1]
        return float(np.std(r) * 100)
    except Exception:
        return None


def vol20s(assets):
    """{asset: {vol20: 日%, ann: 年化%, stop: min(5%, 3σ) 止损线}} — 被选中的所有标的"""
    out = {}
    qq_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "qqq_usdcny.csv")
    for a in assets:
        try:
            if a in ALL_CODES:
                v = _vol20_of_code(ALL_CODES[a])
            elif a in IDX_CODES:
                v = _vol20_of_code(IDX_CODES[a])
            elif a in ("QQQ", "USDCNY") and os.path.exists(qq_csv):
                df = pd.read_csv(qq_csv, usecols=["QQQ", "USDCNY=X"]).tail(21)
                col = "QQQ" if a == "QQQ" else "USDCNY=X"
                c = df[col].values
                r = np.diff(c) / c[:-1]
                v = float(np.std(r) * 100) if len(c) > 1 else None
            else:
                continue
            if v is None:
                continue
            out[a] = {"vol20": round(v, 3),
                      "ann": round(v * np.sqrt(252), 1),
                      "stop": round(min(5.0, 3 * v), 2)}
        except Exception:
            continue
    return out


def live_quotes(assets):
    """腾讯实时行情: {asset: {price, prev_close, pct, time}, trade_day: bool}。
    分钟级刷新用。asset 可以是标的池名称(ALL_CODES)、基准指数名称、或 QQQ/USDCNY。
    A股用长格式(无 s_ 前缀)以拿到行情自带时间戳: 据此判定 trade_day ——
    非交易日/开盘前, 行情时间仍停留在上一交易日 → 前端不画新一天的点。"""
    tags = []
    for a in assets:
        code = ALL_CODES.get(a)
        if code:
            code = code.split(".")[0]             # 去掉 .SH/.SZ 后缀
            ex = "sh" if code[0] in "569" else "sz"
            tags.append(("%s%s" % (ex, code), a))
        elif a in IDX_TENCENT:
            tags.append((IDX_TENCENT[a], a))
        elif a in SPECIAL_TENCENT:
            tags.append((SPECIAL_TENCENT[a], a))
    if not tags:
        return {}
    try:
        import requests
        url = "https://qt.gtimg.cn/q=" + ",".join(t for t, _ in tags)
        resp = requests.get(url, timeout=5)
        resp.encoding = "gbk"
        raw = resp.text
    except Exception:
        return {}
    tag_to_asset = dict(tags)
    now_str = _dt.datetime.now().strftime("%H:%M:%S")
    today_s = _dt.datetime.now().strftime("%Y%m%d")
    a_share_dates = set()
    out = {}
    for line in raw.strip().split("\n"):
        if "=" not in line or "~" not in line:
            continue
        tag = line.split("=", 1)[0].strip().replace("v_", "")
        asset = tag_to_asset.get(tag)
        if not asset:
            continue
        parts = line.split("=", 1)[1].strip().strip('";').split("~")
        if len(parts) < 6:
            continue
        try:
            if tag == "usQQQ":
                # 美股: 3现价 4昨收 30时间 32涨跌幅%
                price = float(parts[3])
                prev = float(parts[4])
                pct = float(parts[32]) if len(parts) > 32 else 0.0
                t = parts[30] if len(parts) > 30 else now_str
            elif tag == "whUSDCNY":
                # 外汇: 3现价 5时间(YYYYMMDDHHMMSS) 6昨收 12涨跌额 13涨跌幅%
                price = float(parts[3])
                prev = float(parts[6])
                pct = float(parts[13]) if len(parts) > 13 else 0.0
                t = parts[5] if len(parts) > 5 else now_str
                if len(t) == 14:
                    t = "%s-%s-%s %s:%s:%s" % (t[:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14])
            else:
                # A股/ETF 长格式: 3现价 4昨收 30时间(YYYYMMDDHHMMSS) 32涨跌幅%
                price = float(parts[3])
                prev = float(parts[4])
                pct = float(parts[32]) if len(parts) > 32 else 0.0
                t_raw = parts[30] if len(parts) > 30 else ""
                if len(t_raw) >= 14:
                    t = "%s-%s-%s %s:%s:%s" % (t_raw[:4], t_raw[4:6], t_raw[6:8],
                                               t_raw[8:10], t_raw[10:12], t_raw[12:14])
                    a_share_dates.add(t_raw[:8])
                else:
                    t = now_str
        except Exception:
            continue
        if price <= 0:
            continue
        out[asset] = {"price": price, "prev_close": prev,
                      "pct": round(pct, 2), "time": t}
    # 任一 A 股标的的行情时间=今天 → 今天是已开盘的交易日
    out["trade_day"] = today_s in a_share_dates
    # 所有被选中标的的 σ20 (供页面显示; 本地缓存计算, 无外部请求)
    out["vol20"] = vol20s(list(out.keys()))
    return out


def compute_track():
    """页面全量数据: 我的净值 / 策略净值 / 逐笔(FIFO lot)对账 / 当前持仓 / 汇总"""
    records = _norm(load_records())
    if not records:
        return {"empty": True, "records": [], "assets": sorted(ALL_CODES.keys())}

    err, st = _simulate(records, need_cmaps=True,
                        initial_cash=load_book().get("initial_cash"))
    if err:  # 数据文件理论上已过校验, 兜底
        return {"empty": True, "records": records, "assets": sorted(ALL_CODES.keys()),
                "error": err}

    dates = [x[0] for x in st["daily"]]
    cash0 = (load_book().get("initial_cash")
             or next((float(r["amount"]) for r in records
                      if r["action"] == "buy" and r.get("amount")), 1.0))
    nav_series = [round(x[1] / cash0, 4) for x in st["daily"]]   # 归一化: 起点=1
    cmaps = st["cmaps"]

    # 买卖点落到当日净值上
    nav_by_date = dict(zip(dates, nav_series))
    marks = [{**m, "nav": nav_by_date[m["date"]]} for m in st["marks"]]

    # 逐笔 lot 的对账字段(偏差/同期策略)
    trades = []
    for lot in st["closed_lots"]:
        a = lot["asset"]
        cm = cmaps[a]
        d0, d1 = pd.Timestamp(lot["entry_date"]), pd.Timestamp(lot["exit_date"])
        trades.append({
            "asset": a, "entry_date": lot["entry_date"], "exit_date": lot["exit_date"],
            "entry_price": lot["entry_price"], "exit_price": lot["exit_price"],
            "ret": round((lot["exit_price"] / lot["entry_price"] - 1) * 100, 2),
            "entry_dev": round((lot["entry_price"] / cm[d0] - 1) * 100, 2),
            "exit_dev": round((lot["exit_price"] / cm[d1] - 1) * 100, 2),
            "note": lot.get("note", "")})

    # 当前持仓(多标的)
    open_pos = []
    last_dt = pd.Timestamp(dates[-1])
    for a, h in st["holds"].items():
        cost = sum(lp * lu for _, lp, lu in h["lots"])
        last_close = cmaps[a].get(last_dt)
        if last_close:
            open_pos.append({"asset": a, "avg_cost": round(cost / h["units"], 3),
                             "last": last_close, "units": h["units"],
                             "cost": round(cost, 2),
                             "float_ret": round((h["units"] * last_close / cost - 1) * 100, 2),
                             "weight": round(h["units"] * last_close / st["daily"][-1][1] * 100, 1)})

    # ── 策略同窗口对比（当前默认口径, 用户决议 2026-08-20）──────────────
    s_dates, s_norm = strat_run(dates[0], dates[-1])
    strat = None
    if s_dates is not None:
        strat = {"dates": s_dates, "nav": [round(v, 4) for v in s_norm],
                 "position": strat_run.last_pos,
                 **_metrics(s_norm)}
        s_idx = {d: i for i, d in enumerate(s_dates)}
        for t in trades:
            i0, i1 = s_idx.get(t["entry_date"]), s_idx.get(t["exit_date"])
            t["strat_ret"] = (round((s_norm[i1] / s_norm[i0] - 1) * 100, 2)
                              if i0 is not None and i1 is not None else None)

    # ── 四大指数基准(与我的净值同窗口 rebase) ──
    benchmarks = []
    IDX = {"上证指数": "000001.SH", "创业板指数": "399006.SZ", "深证成指": "399001.SZ",
           "科创50": "000688.SH"}
    try:
        idx_dfs = FEED.get_dataframe(list(IDX.values()), str(dates[0]).replace("-", ""),
                                     str(dates[-1]).replace("-", ""))
        ts_dates = [pd.Timestamp(d) for d in dates]
        for nm, code in IDX.items():
            df = idx_dfs.get(code)
            if df is None or len(df) == 0:
                continue
            cm = dict(zip(pd.to_datetime(df["trade_date"]), df["close"].astype(float)))
            base = next((cm[d] for d in ts_dates if d in cm), None)
            if not base or base <= 0:
                continue
            nav = [round(cm[d] / base, 4) if d in cm else None for d in ts_dates]
            benchmarks.append({"name": nm, "nav": nav,
                               **_metrics([v for v in nav if v is not None])})
    except Exception:
        pass  # 指数数据不可用时宁可不画基准线, 不让整页挂掉

    # ── 策略组合对比（可调标的池/参数）──────────────────────────────────
    combo_defs = load_combos().get("combos", [])
    combos = []
    for c in combo_defs:
        try:
            cd, cn = strat_run(dates[0], dates[-1],
                                         ",".join(c.get("universe") or []),
                                         c.get("params") or {})
        except Exception:
            cd = None
        if cd:
            combos.append({"name": c.get("name", ""),
                           "universe": ",".join(c.get("universe") or []),
                           "dates": cd, "nav": [round(v, 4) for v in cn],
                           "position": strat_run.last_pos,
                           **_metrics(cn)})

    wins = [t for t in trades if t["ret"] > 0]
    return {
        "empty": False,
        "records": records,
        "dates": dates, "nav": nav_series, "marks": marks,
        "trades": trades, "open_pos": open_pos,
        "mine": {**_metrics(nav_series), "n_trades": len(trades),
                 "win_rate": round(100 * len(wins) / len(trades), 1) if trades else None,
                 "cash_left": round(st["cash"], 2),
                 "nav_yuan": round(st["daily"][-1][1], 2),
                 "initial_cash": load_book().get("initial_cash")},
        "strat": strat,
        "benchmarks": benchmarks,
        "combos": combos,
        "combo_defs": combo_defs,
        "assets": sorted(ALL_CODES.keys()),
    }
