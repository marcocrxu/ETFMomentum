"""实盘实况页数据装配 — /live 页面的唯一数据模块

数据源全部来自共享引擎链路 (live/query.py → LiveRunner → strategy/signal.py),
本模块只做展示层装配, 不含任何策略逻辑 (铁律见 AGENT.md §7b)。
"""
import json
import os
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(BASE, "state.json")
NOTIFY_LOG = os.path.join(BASE, "live", "notify_log.jsonl")
DECISIONS_LOG = os.path.join(BASE, "live", "decisions.jsonl")


def _load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _last_of(path, kind=None):
    if not os.path.exists(path):
        return None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or rec.get("kind") == kind:
                last = rec
    return last


def get_live_data():
    """装配 /live 页面与 /api/live/today 的全部数据"""
    from live.query import current_view

    view = current_view()
    state = _load_state()
    last_reconcile = _last_of(NOTIFY_LOG, "reconcile")
    last_decision = _last_of(DECISIONS_LOG)

    data = {
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ok": False,
    }
    if "error" in view:
        data["error"] = view["error"]
        return data
    if view.get("signal") == "非交易日" or "momentums" not in view:
        data["error"] = "非交易日"
        return data

    live = view.get("live_prices", {})
    sim_pos = view.get("sim_position", "")
    actual_pos = state.get("position") or ""

    # 标的卡片 (按动量排名)
    cards = []
    for rank, (name, mom) in enumerate(
            sorted(view["momentums"].items(), key=lambda x: -x[1])):
        cards.append({
            "rank": rank,
            "name": name,
            "momentum": mom,
            "price": live.get(name),
            "is_strategy_pos": name == sim_pos,
            "is_actual_pos": name == actual_pos,
        })

    # 今日信号与操作建议 (与 14:50 推送同一函数, 以实际持仓为锚)
    from live.daily_run import suggestion_for
    suggestion, suggestion_level = suggestion_for(view, state)
    sig = view.get("signal", "HOLD")
    target = view.get("target")

    # 实际持仓盈亏
    actual = None
    if actual_pos:
        entry_price = state.get("entry_price", 0)
        cur = live.get(actual_pos, 0)
        pnl = (cur / entry_price - 1) * 100 if entry_price > 0 and cur else None
        actual = {
            "position": actual_pos,
            "entry_date": state.get("entry_date"),
            "entry_price": entry_price,
            "shares": state.get("shares"),
            "current_price": cur,
            "pnl_pct": round(pnl, 2) if pnl is not None else None,
        }

    data.update({
        "ok": True,
        "date": view.get("date"),
        "cards": cards,
        "sh_momentum": view.get("sh_momentum"),
        "nz_mix_active": view.get("nz_mix_active"),
        "nz_note": view.get("nz_note", ""),
        "signal": {"signal": sig, "target": target, "reason": view.get("reason", "")},
        "suggestion": suggestion,
        "suggestion_level": suggestion_level,
        "strategy_position": {
            "position": sim_pos,
            "entry_date": view.get("sim_entry_date"),
            "entry_price": view.get("sim_entry_price"),
        },
        "actual_position": actual,
        "diverged": actual_pos != sim_pos,
        "recent_actions": view.get("recent_actions", []),
        "last_reconcile": ({
            "ts": last_reconcile["ts"],
            "ok": not (last_reconcile.get("extra") or {}).get("problems"),
            "message": last_reconcile.get("message", ""),
        } if last_reconcile else None),
        "last_decision": last_decision,
    })
    return data
