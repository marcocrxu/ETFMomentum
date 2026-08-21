# -*- coding: utf-8 -*-
"""国内黄金杠杆替代品种 代入动量策略回测 (2026-08-21)

方法: 把默认池(纳指/黄金/创业)中的「黄金(518880)」逐一替换为候选杠杆品种,
池大小保持 3 (扩池被 AGENT.md F4 纪律禁止), 其余参数全部用生产默认。

候选:
  股票(全窗口 2016-07-07~今): 山东黄金 600547 / 赤峰黄金 600988 /
                              紫金矿业 601899 / 中金黄金 600489
  黄金股ETF(2023-01-01~今):   159562 华夏 / 517520 永赢
  合成杠杆(518880日收益×L, 日重置, 近似"轻仓期货"): 1.5x / 2x

数据: 写入共享 FEED 缓存目录(与生产同 schema), 由同一引擎读取;
      股票用 pro.daily+pro.adj_factor 复权, 与 ETF 复权口径一致。
用法: PYTHONPATH=. python3 backtest/gold_leverage_backtest.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import tushare as ts

from web.config_local import TUSHARE_TOKEN
from web import bt_runner as bt

CACHE = "/root/backtest/cache/etf_momentum"
pro = ts.pro_api(TUSHARE_TOKEN)

def fetch_stock(code):
    """pro.daily + pro.adj_factor → 与 FEED 缓存同 schema 的 DataFrame"""
    d = pro.daily(ts_code=code, start_date="20160101", end_date="20260821")
    a = pro.adj_factor(ts_code=code, start_date="20160101", end_date="20260821")
    df = d.merge(a[["trade_date", "adj_factor"]], on="trade_date", how="left")
    df = df.sort_values("trade_date")
    df["adj_factor"] = df["adj_factor"].ffill().bfill().fillna(1.0)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    return df

def fetch_fund(code):
    d = pro.fund_daily(ts_code=code, start_date="20160101", end_date="20260821")
    try:
        a = pro.fund_adj(ts_code=code)
        d = d.merge(a[["trade_date", "adj_factor"]], on="trade_date", how="left")
        d["adj_factor"] = d["adj_factor"].ffill().bfill().fillna(1.0)
    except Exception:
        d["adj_factor"] = 1.0
    d = d.sort_values("trade_date")
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.strftime("%Y-%m-%d")
    return d

def write_cache(code, df):
    cols = ["ts_code", "trade_date", "pre_close", "open", "high", "low",
            "close", "change", "pct_chg", "vol", "amount", "adj_factor"]
    df = df.copy()
    df["ts_code"] = code
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    path = os.path.join(CACHE, code.replace(".", "_") + ".csv")
    df[cols].to_csv(path, index=False)
    print("  缓存写入 %s (%d 行, %s~%s)" % (path, len(df), df["trade_date"].iloc[0], df["trade_date"].iloc[-1]))

def synth_leverage(base_code, lev, name, code):
    """518880 复权收盘日收益 × lev (日重置) → 合成杠杆序列"""
    b = pd.read_csv(os.path.join(CACHE, base_code.replace(".", "_") + ".csv"))
    adj = b["close"] * b["adj_factor"] / b["adj_factor"].iloc[-1]
    r = adj.pct_change().fillna(0.0)
    nav = (1 + lev * r).cumprod()
    nav = nav * (adj.iloc[0] / nav.iloc[0])          # 对齐起始价位, 便于对照
    df = pd.DataFrame({"ts_code": code, "trade_date": b["trade_date"],
                       "close": nav, "adj_factor": 1.0})
    write_cache(code, df)
    return df

def run(universe, start, end, label):
    r = bt.run_backtest({"universe": ",".join(universe), "start_date": start, "end_date": end})
    if "error" in r:
        print("  [%s] 回测失败: %s" % (label, r["error"]))
        return None
    print("  [%s] %s~%s: 累计 %.1f%%  年化 %.1f%%  最大回撤 %.1f%%  Calmar %.2f  夏普 %.2f  换仓 %d 次  末持仓 %s" % (
        label, start, end, r["total_ret"], r["ann_ret"], r["max_dd"],
        r["calmar"], r["sharpe"], r["trade_count"], r["last_position"]))
    return r

def main():
    print("== 准备数据 ==")
    # 股票 (全历史)
    stocks = {"山东黄金": "600547.SH", "赤峰黄金": "600988.SH",
              "紫金矿业": "601899.SH", "中金黄金": "600489.SH"}
    for nm, code in stocks.items():
        try:
            write_cache(code, fetch_stock(code))
            bt.ALL_CODES[nm] = code
        except Exception as e:
            print("  %s 拉取失败: %s" % (nm, str(e)[:100]))
    # 黄金股 ETF
    funds = {"黄金股ETF(华夏)": "159562.SZ", "黄金股ETF(永赢)": "517520.SH"}
    for nm, code in funds.items():
        try:
            write_cache(code, fetch_fund(code))
            bt.ALL_CODES[nm] = code
        except Exception as e:
            print("  %s 拉取失败: %s" % (nm, str(e)[:100]))
    # 合成杠杆 (近似轻仓期货)
    synth_leverage("518880.SH", 1.5, "1.5倍黄金(合成)", "AU15X.SH"); bt.ALL_CODES["1.5倍黄金(合成)"] = "AU15X.SH"
    synth_leverage("518880.SH", 2.0, "2倍黄金(合成)", "AU20X.SH");  bt.ALL_CODES["2倍黄金(合成)"] = "AU20X.SH"

    FULL = ("2016-07-07", "2026-08-21")
    SHORT = ("2023-01-01", "2026-08-21")

    print("\n== 全窗口 (2016-07-07 ~ 2026-08-21, 替换池中「黄金」) ==")
    run(["纳指", "黄金", "创业"], *FULL, "基准: 黄金ETF")
    for nm in stocks:
        if nm in bt.ALL_CODES:
            run(["纳指", nm, "创业"], *FULL, "替换为: " + nm)
    run(["纳指", "1.5倍黄金(合成)", "创业"], *FULL, "替换为: 1.5x黄金(合成)")
    run(["纳指", "2倍黄金(合成)", "创业"], *FULL, "替换为: 2x黄金(合成)")

    print("\n== 短窗口 (2023-01-01 ~ 2026-08-21, 黄金股ETF 上市后) ==")
    run(["纳指", "黄金", "创业"], *SHORT, "基准: 黄金ETF")
    for nm in funds:
        if nm in bt.ALL_CODES:
            run(["纳指", nm, "创业"], *SHORT, "替换为: " + nm)

if __name__ == "__main__":
    main()
