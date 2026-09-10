# -*- coding: utf-8 -*-
"""用官方 v9 代码/数据, 跑我方口径窗口的对照"""
import sys
sys.path.insert(0, '/home/ecs-user/pdfgen/v9code')
from market_data import UNIVERSE, fetch_history
from backtest import backtest, report, yearly

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
print("数据: %s ~ %s" % (calendar[0], calendar[-1]))

for start, end, tag in (
    ("2016-07-07", "2026-08-20", "我方主样本窗口"),
    ("2016-01-01", "2026-07-31", "打榜窗口"),
    ("2021-08-23", "2026-08-20", "样本外验证窗"),
):
    res = backtest(histories, calendar, start=start, end=end)
    report(res, "v9 官方代码 %s (%s ~ %s)" % (tag, start, end))
    for y, r in yearly(res["daily"]):
        print("  %s: %+7.1f%%" % (y, r * 100))
