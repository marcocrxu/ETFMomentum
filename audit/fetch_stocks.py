"""抓取 A股个股日线 (baostock, 前复权) — 沪深300 + 中证500 成分股
存到 /home/ecs-user/backtest/cache/stocks/{code}.csv
可断点续传 (已存在的文件跳过)
"""
import os
import sys
import time

import baostock as bs
import pandas as pd

OUT = "/home/ecs-user/backtest/cache/stocks"
os.makedirs(OUT, exist_ok=True)

lg = bs.login()
print("login:", lg.error_code, lg.error_msg, flush=True)

codes = []
for fn, lab in [(bs.query_hs300_stocks, "HS300"), (bs.query_zz500_stocks, "ZZ500")]:
    rs = fn()
    n = 0
    while rs.next():
        row = rs.get_row_data()
        codes.append(row[1])          # sh.600519
        n += 1
    print("%s: %d 只" % (lab, n), flush=True)
codes = sorted(set(codes))
print("合计 %d 只" % len(codes), flush=True)

SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
codes = codes[SHARD::NSHARD]
print("分片 %d/%d: 本片 %d 只" % (SHARD, NSHARD, len(codes)), flush=True)

FIELDS = "date,close,volume,amount,turn,pctChg,tradestatus,isST"
done = fail = skip = 0
t0 = time.time()
for i, code in enumerate(codes):
    f = os.path.join(OUT, code.replace(".", "_") + ".csv")
    if os.path.exists(f) and os.path.getsize(f) > 500:
        skip += 1
        continue
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(code, FIELDS, start_date="2013-01-01",
                                              end_date="2026-10-09", frequency="d",
                                              adjustflag="2")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=FIELDS.split(","))
                df.to_csv(f, index=False)
                done += 1
            else:
                fail += 1
            break
        except Exception as e:
            if attempt == 2:
                fail += 1
                print("FAIL", code, e, flush=True)
            else:
                time.sleep(1.0)
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print("进度 %d/%d  新增 %d 跳过 %d 失败 %d  用时 %.0fs" %
              (i + 1, len(codes), done, skip, fail, el), flush=True)
    time.sleep(0.05)

bs.logout()
print("完成: 新增 %d, 跳过 %d, 失败 %d, 总用时 %.0fs" % (done, skip, fail, time.time() - t0), flush=True)
