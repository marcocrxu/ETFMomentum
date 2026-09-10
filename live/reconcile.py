"""每日对账 (cron 21:30) — 收盘口径回放 vs 14:50 实盘决策, 不一致即钉钉告警

为什么要这个模块 (2026-09-10 事故): 旧实盘链路是策略副本, 与主策略分叉 2 个月
无人察觉。现在实盘与回测同源, 但仍有残余风险: 实时价 vs 收盘价的尾盘漂移、
QQQ 数据新鲜度、人为未执行信号。每天收盘后对账一次, 把分歧推到明面上。

对账项:
1. 收盘口径回放(run_backtest, 生产默认参数)的当前仓位 vs 14:50 决策记录的策略仓位
2. 策略仓位 vs 实际持仓 (state.json)
3. QQQ-CNY 数据新鲜度 (为明天 14:50 决策做准备)
"""
import json
import os
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECISIONS_LOG = os.path.join(BASE, "live", "decisions.jsonl")
STATE_FILE = os.path.join(BASE, "state.json")


def _last_decision():
    if not os.path.exists(DECISIONS_LOG):
        return None
    with open(DECISIONS_LOG, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None


def _load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def run_reconcile(push=True):
    from live import notify
    today = datetime.now().strftime("%Y-%m-%d")
    problems = []
    notes = []

    # ── 1. 收盘口径回放 ──
    bt_pos = None
    bt_data_date = None   # 回放实际覆盖到的最后数据日 (≠ 最后决策日)
    try:
        from web.bt_runner import run_backtest
        import pandas as pd
        start = (pd.Timestamp(today) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        r = run_backtest({"universe": "纳指,黄金,创业",
                          "start_date": start, "end_date": today})
        if "error" in r:
            problems.append("收盘回放失败: %s" % r["error"])
        else:
            bt_pos = r.get("last_position")  # "空仓" 或标的名
            dates = r.get("dates") or []
            bt_data_date = dates[-1] if dates else None
    except Exception as e:
        problems.append("收盘回放异常: %s: %s" % (type(e).__name__, e))

    # ── 2. 14:50 实盘决策记录 ──
    dec = _last_decision()
    if dec is None:
        problems.append("无 14:50 决策记录 (daily_run 今天没跑?)")
    else:
        sim_pos = dec.get("sim_position") or "空仓"
        if dec.get("date") != today:
            notes.append("最近决策记录日期 %s ≠ 今天 %s (非交易日属正常)"
                         % (dec.get("date"), today))
        if bt_pos is not None and bt_data_date and bt_data_date >= dec.get("date", ""):
            # 回放数据已覆盖到 14:50 决策当天: 两口径仓位应一致
            if (bt_pos or "空仓") != sim_pos:
                problems.append(
                    "口径分歧: 14:50实时口径仓位[%s] vs 收盘回放仓位[%s]@%s (尾盘漂移或数据问题)"
                    % (sim_pos, bt_pos, bt_data_date))
        elif bt_pos is not None:
            notes.append("回放数据仅到 %s < 决策日 %s, 仓位对比跳过 (tushare 收盘数据未就位)"
                         % (bt_data_date, dec.get("date")))

    # ── 3. 策略仓位 vs 实际持仓 ──
    st = _load_state()
    actual = st.get("position") or "空仓"
    if dec is not None:
        sim_pos = dec.get("sim_position") or "空仓"
        if actual != sim_pos:
            problems.append("持仓偏离: 实际[%s] vs 策略[%s] (用户未跟上信号或状态未更新)"
                            % (actual, sim_pos))

    # ── 4. QQQ-CNY 新鲜度 ──
    try:
        from data.usfeed import qqq_cny_freshness
        last, expect, fresh = qqq_cny_freshness()
        if not fresh:
            problems.append("QQQ-CNY 数据陈旧: 最新%s, 期望≥%s (明日纳指动量将降级)"
                            % (last.date(), expect.date()))
    except Exception as e:
        notes.append("QQQ 新鲜度检查异常: %s" % e)

    # ── 汇总推送 ──
    if problems:
        msg = "🚨 ETF 策略每日对账 (%s)\n\n" % today + \
              "\n".join("❌ " + p for p in problems)
        if notes:
            msg += "\n\n" + "\n".join("ℹ️ " + n for n in notes)
        if push:
            notify.send_dingtalk(msg)
        notify.archive("reconcile", msg, extra={"problems": problems, "notes": notes})
        print(msg)
        return False

    msg = "✅ ETF 策略每日对账 (%s): 实盘=收盘回放=实际持仓, QQQ 数据新鲜" % today
    if notes:
        msg += "\n" + "\n".join("ℹ️ " + n for n in notes)
    notify.archive("reconcile", msg, extra={"problems": [], "notes": notes})
    print(msg)
    return True


if __name__ == "__main__":
    import sys
    run_reconcile(push=("--no-push" not in sys.argv))
