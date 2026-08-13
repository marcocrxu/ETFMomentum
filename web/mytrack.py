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
FEED = TushareFeed(TUSHARE_TOKEN, cache_dir="/root/backtest/cache/etf_momentum")


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
                             "last": last_close,
                             "float_ret": round((h["units"] * last_close / cost - 1) * 100, 2),
                             "weight": round(h["units"] * last_close / st["daily"][-1][1] * 100, 1)})

    # ── 策略同窗口对比 ──
    bt = run_backtest({"universe": "纳指,黄金,创业", "start_date": dates[0], "end_date": dates[-1]})
    strat = None
    if "error" not in bt:
        s_dates, s_nav = bt["dates"], bt["nav"]
        base = s_nav[0]
        s_norm = [v / base for v in s_nav]
        strat = {"dates": s_dates, "nav": [round(v, 4) for v in s_norm],
                 **_metrics(s_norm)}
        s_idx = {d: i for i, d in enumerate(s_dates)}
        for t in trades:
            i0, i1 = s_idx.get(t["entry_date"]), s_idx.get(t["exit_date"])
            t["strat_ret"] = (round((s_norm[i1] / s_norm[i0] - 1) * 100, 2)
                              if i0 is not None and i1 is not None else None)

    # ── 三大指数基准(与我的净值同窗口 rebase) ──
    benchmarks = []
    IDX = {"上证指数": "000001.SH", "创业板指数": "399006.SZ", "深证成指": "399001.SZ"}
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
        "assets": sorted(ALL_CODES.keys()),
    }
