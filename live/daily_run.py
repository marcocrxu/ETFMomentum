"""生产实盘入口 (cron 14:50) — 唯一权威决策链路

策略逻辑 = strategy/signal.py (与回测共用, 唯一权威引擎; AGENT.md 铁律)。
本模块只做: 刷新数据 → 跑 LiveRunner(内部调 compute_signals) → 组装消息 →
推送钉钉/微信 → 完整落盘 → 更新实际持仓状态。

历史教训 (2026-09-10 复盘):
- stock-bot/cron_etf_momentum.py 是 8 月中旬手工抄的策略副本, 主策略后续迭代
  (加速止盈/恐慌抄底/绝对止损/空仓冷却/QQQ混合) 全部没有同步, 07 月起实盘决策
  与主策略分叉, 09-09 丢失"上证空仓"信号。实盘决策只允许来自本模块。
- 旧 cron 只落盘决策一句话, 完整消息无存档; 本模块全量落盘 notify_log.jsonl。
- QQQ-CNY 数据陈旧必须显式告警, 禁止静默降级 (nz_mix_active=False 时消息里红字标注)。
"""
import json
import os
from datetime import datetime, timedelta

import pandas as pd


def _load_env_file(path):
    """极简 .env 解析 (不依赖 python-dotenv)"""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# token 来源: 环境变量 > stock-bot/.env > 项目 .env > web/config_local.py
_load_env_file("/home/ecs-user/stock-bot/.env")
_load_env_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from strategy.signal import SignalConfig
from data.feed import TushareFeed
from live.runner import LiveRunner
from live import notify

# 生产标的池 (窄池 3 标的, 与回测基线一致)
UNIVERSE = {"纳指": "513100.SH", "创业": "159915.SZ", "黄金": "518880.SH"}

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(BASE, "state.json")          # 实际持仓 (已从 stock-bot 迁移)
DECISIONS_LOG = os.path.join(BASE, "live", "decisions.jsonl")
CACHE_DIR = "/home/ecs-user/backtest/cache/etf_momentum"

# 旧 cron (stock-bot) 持仓名 → 策略池名
NAME_ALIAS = {"黄金ETF": "黄金", "纳斯达克": "纳指", "创业板": "创业",
              "纳指ETF": "纳指", "创业板ETF": "创业"}


def production_config():
    """生产口径 = SignalConfig 全部默认值 (AGENT.md: 对账基准 = 当前默认口径)。
    单独成函数是为了让测试钉死这一点: 任何默认值变更都会反映到实盘。"""
    return SignalConfig()


def _tushare_token():
    tok = os.environ.get("TUSHARE_TOKEN", "")
    if tok:
        return tok
    try:
        from web.config_local import TUSHARE_TOKEN
        return TUSHARE_TOKEN
    except ImportError:
        return ""


def load_state():
    """实际持仓状态。优先 state.json; 不存在则从旧 cron 的 momentum_state.json 迁移一次。"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    legacy = "/home/ecs-user/stock-bot/momentum_state.json"
    if os.path.exists(legacy):
        with open(legacy) as f:
            st = json.load(f)
        st["position"] = NAME_ALIAS.get(st.get("position"), st.get("position"))
        st["migrated_from"] = legacy
        save_state(st)
        return st
    return {"position": None, "entry_date": None, "entry_price": 0}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def suggestion_for(res, state):
    """操作建议 — 一律以【实际持仓】为锚，返回 (文本, 级别)。

    2026-09-11 用户反馈: 此前按策略模拟仓位说"卖出纳指，买入黄金",
    但用户实际从未持有纳指 —— 建议必须回答"我该怎么办", 而不是"模拟账户该怎么办"。"""
    sig, target = res.get("signal"), res.get("target")
    actual = state.get("position") or None
    sim_pos = res.get("sim_position") or ""
    entry = res.get("sim_entry_date", "?")

    if sig == "BUY" and target:
        if actual == target:
            return ("已持有 %s，无需操作" % target, "hold")
        if actual:
            return ("卖出 %s，买入 %s" % (actual, target), "switch")
        return ("买入 %s" % target, "switch")
    if sig == "CASH":
        if actual:
            return ("卖出 %s，空仓" % actual, "cash")
        return ("保持空仓", "wait")
    # HOLD: 无新信号, 按 实际 vs 策略 对齐情况给建议
    if actual and actual == sim_pos:
        # 日期优先用实际持仓的 entry_date (策略模拟入场日可能比实盘早, 如 2026-10-14 脏数据漏推)
        return ("继续持有 %s（自 %s 起）" % (actual, state.get("entry_date") or entry), "hold")
    if actual and not sim_pos:
        return ("卖出 %s，空仓（对齐策略）" % actual, "cash")
    if actual and sim_pos and actual != sim_pos:
        # 实际与策略各持一只(如漏跟了历史信号) → 明确切换, 不得落回"空仓观望"
        return ("卖出 %s，买入 %s（对齐策略仓位）" % (actual, sim_pos), "switch")
    if not actual and sim_pos:
        return ("买入 %s（对齐策略仓位）" % sim_pos, "switch")
    return ("继续空仓观望", "wait")


def data_integrity_problems(dfs, open_days, start=None, end=None):
    """数据体检: 缓存交易日 vs 交易所日历, 返回问题描述列表。

    2026-10-15 事故: 000001.SH 缓存混入一条周六行(2026-09-12) → 对齐日历多出一天,
    其余标的当天全 NaN → 创业板指 MA200 变 NaN, 恐慌抄底被静默否决, 而 14:50 推送
    与 21:30 收盘回放读的是同一份脏缓存, 两层一致地错 → 无人发现。
    体检放在推送链路里, 异常直接进推送正文 (不再只写日志)。

    dfs: {code: DataFrame(含 trade_date)}; open_days: 交易日集合(pd.Timestamp)。
    """
    problems = []
    for code, df in sorted(dfs.items()):
        if df is None or len(df) == 0 or "trade_date" not in df.columns:
            continue
        ds = pd.to_datetime(df["trade_date"])
        if start is not None:
            ds = ds[ds >= start]
        if end is not None:
            ds = ds[ds <= end]
        if len(ds) == 0:
            continue
        bad = sorted(set(ds) - set(open_days))
        if bad:
            problems.append("%s 含 %d 个非交易日行: %s"
                            % (code, len(bad), ", ".join(str(d.date()) for d in bad[:5])))
        expect = {d for d in open_days if ds.min() <= d <= ds.max()}
        miss = sorted(expect - set(ds))
        if miss:
            problems.append("%s 缺 %d 个交易日: %s"
                            % (code, len(miss), ", ".join(str(d.date()) for d in miss[:5])))
    return problems


def _fetch_integrity_problems(feed, codes, days=120):
    """拉最近 days 天的缓存做体检; 取数失败不阻断推送(返回告警文本)。"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        dfs = feed.get_dataframe(codes, start, end)
        cal = feed.pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        open_days = set(pd.to_datetime(cal[cal["is_open"] == 1]["cal_date"], format="%Y%m%d"))
        return data_integrity_problems(dfs, open_days, start=pd.Timestamp(start), end=pd.Timestamp(end))
    except Exception as e:
        return ["数据体检未能执行: %s" % e]


def build_message(res, state, problems=None):
    """组装推送消息。res=LiveRunner.check() 结果, state=实际持仓。
    problems=数据体检问题列表(非空时置顶告警: 信号可能失真)。"""
    lines = ["📊 ETF 20日动量轮动 (共享引擎口径)", f"⏰ {res['date']}", ""]

    if problems:
        lines.append("🚨 **数据体检异常 —— 以下信号可能失真, 勿直接执行**")
        for p in problems[:5]:
            lines.append("  • %s" % p)
        lines.append("")

    # 恐慌抄底"无法判定"(缺数据被静默吞掉) —— 与"真的没触发"必须区分开
    if res.get("crash_unknown"):
        lines.append("🚨 **恐慌抄底今日无法判定**(缺数据, 不等于未触发):")
        for p in res["crash_unknown"][:3]:
            lines.append("  • %s" % p)
        lines.append("")

    live = res.get("live_prices", {})
    actual_pos = state.get("position") or None
    ranked = sorted(res["momentums"].items(), key=lambda x: -x[1])
    sim_pos = res.get("sim_position", "")
    prev_pos = res.get("prev_position", "")
    for i, (name, mom) in enumerate(ranked):
        arrow = "🟢" if mom >= 0 else "🔴"
        icon = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "")
        px = "  (¥%.4f)" % live[name] if name in live else ""
        hold = " ← 策略仓位" if name == sim_pos else ""
        lines.append(f"{icon} {arrow} **{name}**: {mom:+.2f}%{px}{hold}")
    lines.append(f"    📉 上证指数: {res.get('sh_momentum', 0):+.2f}%")

    if res.get("nz_mix_active"):
        lines.append(f"    🧮 纳指动量 = {res.get('nz_note', '')}")
    else:
        lines.append(f"    ⚠️ 纳指动量降级: {res.get('nz_note', 'QQQ混合未生效')}")

    # 今日信号
    sig, target, reason = res["signal"], res.get("target"), res.get("reason", "")
    lines.extend(["", f"📋 **今日信号**: {sig}" + (f" → {target}" if target else "")])
    if reason:
        lines.append(f"    依据: {reason}")

    # 实际持仓: 成本 / 现价 / 盈亏 / 距止损线 (以 state.json 的实际成本为锚)
    if actual_pos and actual_pos in live and state.get("entry_price"):
        _ep = float(state["entry_price"]); _px = live[actual_pos]
        _line = ["💼 持仓: %s @%.3f (%s)" % (actual_pos, _ep, state.get("entry_date", "?"))]
        if _px > 0:
            _line.append("现价 ¥%.3f 盈亏 %+.2f%%" % (_px, (_px / _ep - 1) * 100))
        # 止损线必须锚定"你手里这仓的成本": res["stop_price"] 是**策略模拟仓**的入场价 ×
        # (1+stop_loss), 实际持仓与模拟仓不同时拿它冒充会误导 (2026-10-21~22 事故:
        # 引擎按其模拟入场 2.891 触发 -5% 止损, 用户实际成本 2.786 当时只有 -2.62%,
        # 推送却把纳指的止损线 2.195 印在创业持仓那一行)
        _sim_sp = res.get("stop_price")
        if actual_pos == sim_pos:
            _sp, _tag = _sim_sp, "引擎止损线"
        else:
            _sp = round(_ep * (1 + production_config().stop_loss / 100.0), 4)
            _tag = "实盘止损线(按你的成本)"
        if _sp and _px > 0:
            _gap = (_sp / _px - 1) * 100
            _line.append("%s ¥%.3f (%s)" % (
                _tag, _sp,
                ("还差 %.2f%% 跌破" % -_gap) if _gap < 0 else ("已跌破 %.2f%%" % _gap)))
            if _sim_sp and actual_pos != sim_pos:
                _line.append("策略模拟仓[%s]止损线 ¥%.3f" % (sim_pos or "空仓", _sim_sp))
        lines.append(" | ".join(_line))

    # 操作建议 (以实际持仓为锚, 与 /live 页同一函数)
    sug_text, _sug_level = suggestion_for(res, state)
    sig_emoji = {"switch": "🔄", "cash": "⚠️", "hold": "✅", "wait": "⏸️"}[_sug_level]
    lines.append(f"{sig_emoji} 操作建议: **{sug_text}**")

    # 实际持仓 vs 策略仓位偏离告警
    actual = state.get("position") or None
    if actual != (sim_pos or None):
        lines.extend(["",
                      f"🚨 **持仓偏离**: 你的实际持仓 [{actual or '空仓'}] "
                      f"与策略仓位 [{sim_pos or '空仓'}] 不一致！"])

    lines.extend(["", "📐 引擎: strategy/signal.py 共享口径 (回测=实盘)"])
    return "\n".join(lines)


def update_state_after_signal(state, res):
    """今日有买卖信号时更新实际持仓状态 (沿用旧 cron 语义: 假设按信号执行)。
    返回 (state, changed)。"""
    sig = res["signal"]
    if sig == "BUY" and res.get("target"):
        state["position"] = res["target"]
        state["entry_date"] = datetime.now().strftime("%Y-%m-%d")
        px = res.get("live_prices", {}).get(res["target"], 0)
        state["entry_price"] = px
        return state, True
    if sig == "CASH":
        state["position"] = None
        state["entry_date"] = None
        state["entry_price"] = 0
        return state, True
    return state, False


def main(push=True, write=True):
    """push=推送钉钉/微信; write=更新实际持仓状态+写决策记录。
    干跑 (--no-push) 默认也不写任何状态, 防止测试污染生产状态
    (2026-09-11 教训: 干跑时降级口径的假信号曾覆盖真实持仓记录)。"""
    # 1. 刷新 QQQ-CNY (内含新浪/东财+中国货币网兜底)
    try:
        from data.usfeed import refresh_qqq_cny
        refresh_qqq_cny()
    except Exception as e:
        print("QQQ-CNY 刷新失败(将用旧缓存): %s" % e)

    # 2. 跑共享引擎
    feed = TushareFeed(_tushare_token(), cache_dir=CACHE_DIR)
    runner = LiveRunner(production_config(), feed, UNIVERSE, state_file=STATE_FILE)
    res = runner.check()
    if "error" in res:
        msg = "ETF 动量数据获取失败: %s" % res["error"]
        if push:
            notify.send_dingtalk(msg)
        notify.archive("alert", msg)
        return msg
    if res["signal"] == "非交易日":
        print("非交易日，跳过")
        return "非交易日"

    # 3. 数据体检 + 状态 + 消息
    # (2026-10-15: 脏行曾让 14:50 推送与 21:30 对账一致地错, 体检必须进推送正文)
    problems = _fetch_integrity_problems(feed, list(UNIVERSE.values()) + ["000001.SH", "399006.SZ"])
    if problems:
        print("数据体检异常: %s" % problems)
    state = load_state()
    msg = build_message(res, state, problems=problems)

    # 4. 推送 + 完整落盘 (干跑也归档但标 dryrun, 便于区分"用户实际收到的")
    push_result = notify.push_all(msg) if push else {}
    notify.archive("daily" if push else "daily-dryrun", msg, extra={
        "signal": res["signal"], "target": res.get("target"),
        "reason": res.get("reason", ""),
        "sim_position": res.get("sim_position", ""),
        "actual_position": state.get("position"),
        "nz_mix_active": res.get("nz_mix_active"),
        "momentums": res.get("momentums"),
        "sh_momentum": res.get("sh_momentum"),
        "data_problems": problems,
        "push": {k: v[0] for k, v in push_result.items()} if push_result else None,
    })

    # 5. 有信号时更新实际持仓状态
    if write:
        state, changed = update_state_after_signal(state, res)
        if changed:
            save_state(state)

    # 6. 机器可读决策记录 (reconcile 用)
    if write:
        with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": res["date"], "signal": res["signal"],
                "target": res.get("target"), "reason": res.get("reason", ""),
                "sim_position": res.get("sim_position", ""),
                "actual_position_after": state.get("position"),
                "nz_mix_active": res.get("nz_mix_active"),
            }, ensure_ascii=False) + "\n")

    print(msg)
    print("推送:", push_result)
    return msg


if __name__ == "__main__":
    import sys
    _push = "--no-push" not in sys.argv
    _write = _push and ("--no-write" not in sys.argv)
    main(push=_push, write=_write)
