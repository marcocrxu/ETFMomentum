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
from datetime import datetime


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
        return ("继续持有 %s（自 %s 起）" % (actual, entry), "hold")
    if actual and not sim_pos:
        return ("卖出 %s，空仓（对齐策略）" % actual, "cash")
    if not actual and sim_pos:
        return ("买入 %s（对齐策略仓位）" % sim_pos, "switch")
    return ("继续空仓观望", "wait")


def build_message(res, state):
    """组装推送消息。res=LiveRunner.check() 结果, state=实际持仓。"""
    lines = ["📊 ETF 20日动量轮动 (共享引擎口径)", f"⏰ {res['date']}", ""]

    live = res.get("live_prices", {})
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

    # 3. 状态 + 消息
    state = load_state()
    msg = build_message(res, state)

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
