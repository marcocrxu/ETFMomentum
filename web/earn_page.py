# -*- coding: utf-8 -*-
"""赚钱效应页数据装配 — /earn (纯展示, 不含任何策略判断)

数据: web/static/data/earn_effect.csv (由 web/earn_build10y.py 生成)

价格假设矩阵: 买入价 b ∈ {o 开盘, l 最低, a 均价, c 收盘, h 最高} 取自信号日 T,
             卖出价 s ∈ 同上 取自次日 T+1, 收益 = 卖出/买入 - 1, 组内等权。
另含可执行口径 (T+1 开盘买 → T+2 开盘/收盘卖): 字段 {u}_{g}_o / {u}_{g}_c。
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "web", "static", "data", "earn_effect.csv")

GROUPS = [("lim", "昨日涨停", "🏷️"), ("top", "昨日成交额前10", "🔥"),
          ("up", "昨日上涨", "🟢"), ("dn", "昨日下跌", "🔴")]
BUYS = [("o", "开盘"), ("l", "最低"), ("a", "均价"), ("c", "收盘"), ("h", "最高")]
SELLS = [("h", "最高"), ("c", "收盘"), ("a", "均价"), ("l", "最低"), ("o", "开盘")]
CURVES = [("lh", "最优(最低买→最高卖)"), ("aa", "中位(均价买→均价卖)"),
          ("hl", "最差(最高买→最低卖)"), ("cc", "基准(收盘买→收盘卖)")]


def _num(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)


def _df():
    if not os.path.exists(CSV):
        return None
    return pd.read_csv(CSV, parse_dates=["date"]).sort_values("date")


def get_data():
    df = _df()
    if df is None or len(df) == 0:
        return {"error": "数据未生成, 请先运行 python web/earn_build10y.py"}

    last = df.iloc[-1]
    out = {"date": last["date"].strftime("%Y-%m-%d"),
           "start": df["date"].iloc[0].strftime("%Y-%m-%d"),
           "days": int(len(df)),
           "groups": [{"key": k, "label": l, "icon": i} for k, l, i in GROUPS],
           "buys": [{"key": k, "label": v} for k, v in BUYS],
           "sells": [{"key": k, "label": v} for k, v in SELLS],
           "curve_defs": [{"key": k, "label": v} for k, v in CURVES]}

    latest, matrix = {}, {}
    for u in ("all", "cyb"):
        latest[u] = {"breadth": _num(last[f"{u}_breadth"], 1),
                     "median": _num(last[f"{u}_median"], 3),
                     "lim_n": int(last[f"{u}_lim_n"]), "groups": []}
        matrix[u] = {}
        for k, lab, icon in GROUPS:
            latest[u]["groups"].append({
                "key": k, "label": lab, "icon": icon,
                "cc": _num(last[f"{u}_{k}_cc"]),
                "lh": _num(last[f"{u}_{k}_lh"]),
                "hl": _num(last[f"{u}_{k}_hl"]),
                "exec_open": _num(last[f"{u}_{k}_o"]),
                "exec_close": _num(last[f"{u}_{k}_c"]),
            })
            matrix[u][k] = {b: {s: _num(last[f"{u}_{k}_{b}{s}"]) for s, _ in SELLS}
                            for b, _ in BUYS}

    # 收益曲线 (累计净值, 对数轴)
    series = {"dates": [d.strftime("%Y-%m-%d") for d in df["date"]]}
    curves = {}
    for k, _lab, _icon in GROUPS:
        curves[k] = {}
        for u in ("all", "cyb"):
            for ck, _ in CURVES:
                col = f"{u}_{k}_{ck}"
                if col not in df.columns:
                    continue
                s = df[col].fillna(0) / 100.0
                nav = (1 + s).cumprod()
                curves[k][f"{u}_{ck}"] = [round(float(x), 6) for x in nav]
    out["latest"] = latest
    out["matrix"] = matrix
    out["series"] = series
    out["curves"] = curves

    d = df.copy()
    d["year"] = d["date"].dt.year
    yearly = {}
    for u in ("all", "cyb"):
        rows = []
        for y, g in d.groupby("year"):
            rows.append({"year": int(y), "days": int(len(g)),
                         "lim": _num(g[f"{u}_lim_cc"].mean()),
                         "lim_lh": _num(g[f"{u}_lim_lh"].mean()),
                         "lim_exec": _num(g[f"{u}_lim_o"].mean()),
                         "lim_n": _num(g[f"{u}_lim_n"].mean(), 1),
                         "breadth": _num(g[f"{u}_breadth"].mean(), 1)})
        yearly[u] = rows
    out["yearly"] = yearly
    return out
