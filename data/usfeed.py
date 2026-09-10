"""美股信号源 — QQQ(美元价)×USDCNY 的人民币口径日线序列, 用于纳指动量混合 (I 组审计)

数据: data/qqq_usdcny.csv (2013-06 起, Yahoo Finance chart API, 已随仓库提交)
刷新: refresh_qqq_cny() 拉取最新数据原子写回 (可选, 需外网; 失败不影响读取旧缓存)
对齐: CN 交易日 D 的收盘决策, 只用美国日历日期 ≤ D-1 的已收盘交易段 (严格无前视)
"""
import os
import time

import numpy as np
import pandas as pd

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qqq_usdcny.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _fetch(sym, period1="2013-06-01"):
    import requests
    p1 = int(pd.Timestamp(period1).timestamp())
    p2 = int(pd.Timestamp.now().timestamp())
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%s" % sym,
                     params={"period1": p1, "period2": p2, "interval": "1d"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    d = r.json()["chart"]["result"][0]
    ts = d["timestamp"]
    close = d["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s", utc=True).date, sym: close})
    return df.dropna()


def _fetch_qqq_sina():
    """新浪美股 QQQ 日线 (Yahoo 不可达时的首选兜底; 2026-09 验证与 Yahoo 同值)"""
    import requests, json, re
    r = requests.get("https://stock.finance.sina.com.cn/usstock/api/jsonp.php/x/US_MinKService.getDailyK",
                     params={"symbol": "qqq"},
                     headers={**UA, "Referer": "https://finance.sina.com.cn"},
                     timeout=30)
    r.raise_for_status()
    m = re.search(r"\((\[.*\])\)", r.text, re.S)
    if not m:
        raise RuntimeError("新浪 QQQ 返回格式异常")
    rows = json.loads(m.group(1))
    if not rows:
        raise RuntimeError("新浪 QQQ 返回空")
    df = pd.DataFrame({"date": [x["d"] for x in rows],
                       "QQQ": [float(x["c"]) for x in rows]})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").dropna()


def _fetch_qqq_em(beg="20130601"):
    """东财 QQQ 日线 (备用兜底; 该接口对本机 WAF 拦截不稳定, 排在新浪后面)"""
    import requests
    r = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get",
                     params={"secid": "105.QQQ", "fields1": "f1,f2,f3",
                             "fields2": "f51,f53", "klt": "101", "fqt": "1",
                             "beg": beg, "end": "20500101", "lmt": "1000000"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    kl = (r.json().get("data") or {}).get("klines") or []
    if not kl:
        raise RuntimeError("东财 QQQ 返回空")
    df = pd.DataFrame([k.split(",")[:2] for k in kl], columns=["date", "QQQ"])
    df["date"] = pd.to_datetime(df["date"])
    df["QQQ"] = df["QQQ"].astype(float)
    return df.dropna()


def _fetch_usdcny_cm(beg):
    """中国货币网 USD/CNY 每日中间价 (官方源)。

    注意: 这是中间价口径, 与 CSV 历史的 Yahoo 现货存在稳定价差 (约 -0.9%),
    直接拼接会在接缝处制造假跳变 (0.7 权重下 ≈ +0.6% 假动量, 足以翻转贴线决策),
    因此调用方必须按重叠期价差重定基 —— 见 _refresh_qqq_cny_fallback。"""
    import requests
    r = requests.post("https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew",
                      data={"startDate": beg,
                            "endDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
                            "currency": "USD/CNY", "pageNum": "1", "pageSize": "500"},
                      headers=UA, timeout=30)
    r.raise_for_status()
    recs = r.json().get("records") or []
    if not recs:
        raise RuntimeError("中国货币网 USDCNY 返回空")
    df = pd.DataFrame({"date": [x["date"] for x in recs],
                       "USDCNY=X": [float(x["values"][0]) for x in recs]})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").dropna()


def _refresh_qqq_cny_fallback():
    """Yahoo 不可达时的兜底刷新: QQQ=新浪(东财备用), USDCNY=中国货币网(重定基)。

    只追加新日期行, 历史行不动。汇率重定基: 用重叠期 (旧CSV Yahoo现货 / 新中间价)
    的比值中位数缩放新汇率, 消除口径价差造成的接缝跳变。"""
    old = pd.read_csv(CSV, parse_dates=["date"])
    last = old["date"].max()
    beg = (last - pd.Timedelta(days=20))
    try:
        q = _fetch_qqq_sina()
    except Exception as e:
        print("  新浪 QQQ 失败(%s), 再试东财" % type(e).__name__)
        q = _fetch_qqq_em(beg.strftime("%Y%m%d"))
    q = q[q["date"] >= pd.Timestamp(beg)]
    fx = _fetch_usdcny_cm(beg.strftime("%Y-%m-%d"))

    # 重定基: 重叠期内 旧(Yahoo现货)/新(中间价) 比值中位数
    m = old.merge(fx, on="date", suffixes=("_old", "_new")).dropna()
    if len(m) >= 2:
        ratio = float((m["USDCNY=X_old"] / m["USDCNY=X_new"]).median())
        fx["USDCNY=X"] = fx["USDCNY=X"] * ratio

    merged = q.merge(fx, on="date", how="inner").sort_values("date")
    new_rows = merged[merged["date"] > last]
    if new_rows.empty:
        raise RuntimeError("兜底源无新数据 (最新仍为 %s)" % last.date())
    out = pd.concat([old, new_rows], ignore_index=True).sort_values("date")
    out["date"] = out["date"].dt.date
    tmp = CSV + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, CSV)
    out["qqq_cny"] = out["QQQ"] * out["USDCNY=X"]
    print("  QQQ-CNY 已用兜底源(新浪/东财+中国货币网)追加 %d 行, 最新 %s"
          % (len(new_rows), out['date'].iloc[-1]))
    return out


def refresh_qqq_cny():
    """拉取 QQQ + USDCNY 最新日线, 原子写回 data/qqq_usdcny.csv

    用 merge_asof(backward) 而非 inner join: 汇率 bar 的时间戳(北京 07:00 =
    UTC 前一日 23:00)在 UTC 日期轴上与 QQQ 错位, inner join 会把 QQQ 最新一根
    丢弃(没有同 UTC 日期的汇率行), 导致 CSV 永远落后一天 → 实盘推送
    "QQQ-CNY 数据未就绪" 持续回退纯 ETF (2026-08-21 实锤)。
    backward 语义: 每根 QQQ 配 ≤ 当日的最新汇率, 历史行不变, 仅补齐最新行。

    Yahoo 不可达时自动切兜底源 (东财+中国货币网), 见 _refresh_qqq_cny_fallback。
    """
    try:
        q = _fetch("QQQ")
        time.sleep(1)
        fx = _fetch("USDCNY=X")
    except Exception as e:
        print("  Yahoo 刷新失败(%s), 切兜底源" % type(e).__name__)
        return _refresh_qqq_cny_fallback()
    q = q.sort_values("date")
    fx = fx.sort_values("date")
    # 老 pandas(1.x) 的 merge_asof 要求 datetime64 键, 合并后再转回 date 对象
    q["date"] = pd.to_datetime(q["date"])
    fx["date"] = pd.to_datetime(fx["date"])
    df = pd.merge_asof(q, fx, on="date", direction="backward").dropna()
    df["date"] = df["date"].dt.date
    tmp = CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, CSV)
    # 返回前补齐 qqq_cny 列 (CSV 保持 date/QQQ/USDCNY=X 三列格式不变,
    # load_qqq_cny 读取时再算; 但调用方可能直接把返回值喂给 align_to_cn,
    # 缺失该列会 KeyError: 'qqq_cny' → 实盘"数据未就绪"回退纯 ETF, 2026-08-26 实锤)
    df["qqq_cny"] = df["QQQ"] * df["USDCNY=X"]
    return df


def load_qqq_cny():
    """读取本地 CSV, 返回含 qqq_cny 列 (QQQ×USDCNY) 的 DataFrame"""
    df = pd.read_csv(CSV, parse_dates=["date"])
    df["qqq_cny"] = df["QQQ"] * df["USDCNY=X"]
    return df


def align_to_cn(cn_dates, df=None):
    """CN 交易日序列 → QQQ-CNY 收盘数组: 每个 CN 日期取美国日期 ≤ D-1 的最后收盘 (无前视)"""
    if df is None:
        df = load_qqq_cny()
    s = df.set_index("date")["qqq_cny"]
    idx = pd.DatetimeIndex(s.index)
    out = np.full(len(cn_dates), np.nan)
    for i, d in enumerate(cn_dates):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        pos = idx.searchsorted(dt, side="right") - 1
        if pos >= 0:
            out[i] = s.iloc[pos]
    return out


def align_fear_to_cn(cn_dates, kind):
    """恐慌指标对齐到 CN 交易日 (无前视: CN日D 用美国日期≤D-1 的收盘)
    kind: 'vix' (data/vix.csv) 或 'skew' (data/fear_indices.csv 的 ^SKEW 列)"""
    if kind == "vix":
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vix.csv"),
                         parse_dates=["date"])
        col = "vix"
    else:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fear_indices.csv"),
                         parse_dates=["date"])
        col = "^SKEW"
    df = df[df[col].notna()].sort_values("date")
    s = df.set_index("date")[col]
    idx = pd.DatetimeIndex(s.index)
    out = np.full(len(cn_dates), np.nan)
    for i, d in enumerate(cn_dates):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        p = idx.searchsorted(dt, side="right") - 1
        if p >= 0:
            out[i] = s.iloc[p]
    return out


TREASURY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treasury.csv")


def align_ty_shock_to_cn(cn_dates, window=20):
    """美债利率冲击门特征: |10Y 较 window 个美股交易日前的变化| (pp) 对齐到 CN 交易日
    (无前视: CN日D 用美国日期≤D-1 收盘; 数据 data/treasury.csv = FRED DGS1/DGS10, 2014-01 起)
    rate_shock_gate 使用: 特征>门槛(如0.4) 时禁止新买/切换纳指 (K-3 候选, 2026-08-27 研究)"""
    df = pd.read_csv(TREASURY_CSV, parse_dates=["date"])
    df = df[df["DGS10"].notna()].sort_values("date")
    df["shock"] = df["DGS10"].diff(window).abs()
    s = df.set_index("date")["shock"]
    idx = pd.DatetimeIndex(s.index)
    out = np.full(len(cn_dates), np.nan)
    for i, d in enumerate(cn_dates):
        dt = pd.Timestamp(d) - pd.Timedelta(days=1)
        p = idx.searchsorted(dt, side="right") - 1
        if p >= 0:
            out[i] = s.iloc[p]
    return out


# ── 新鲜度检查 (实盘告警用) ─────────────────────────────
_US_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def last_us_trade_day(day=None):
    """CN 日 day 收盘决策可用的最新美国交易日 (≤ day-1)。默认今天。"""
    if day is None:
        day = pd.Timestamp.now().normalize()
    d = day - pd.Timedelta(days=1)
    while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in _US_HOLIDAYS:
        d -= pd.Timedelta(days=1)
    return d


def qqq_cny_freshness(day=None):
    """返回 (csv最新日期, 期望最新美国交易日, 是否新鲜)。
    不新鲜 = 纳指动量会回退纯 ETF 口径, 实盘必须告警 (2026-09-09 空仓信号丢失事故)。"""
    df = pd.read_csv(CSV, parse_dates=["date"])
    last = df["date"].max()
    expect = last_us_trade_day(day)
    return last, expect, bool(last >= expect)
