"""生产链路唯一性测试 — 钉死"实盘决策只能来自共享引擎"这条铁律

2026-09-10 事故: stock-bot/cron_etf_momentum.py 是 8 月的策略副本,
主策略迭代未同步, 实盘与回测分叉 2 个月无人发现, 09-09 丢失空仓信号。
这些测试保证:
1. 生产配置 = SignalConfig 默认值 (AGENT.md: 对账基准 = 当前默认口径)
2. 生产入口 daily_run 经由 LiveRunner → compute_signals (共享引擎)
3. stock-bot 里不再存在任何策略副本/简化判断
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ETF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_BOT = "/home/ecs-user/stock-bot"


def test_production_config_is_signal_defaults():
    from live.daily_run import production_config
    from strategy.signal import SignalConfig
    assert production_config() == SignalConfig(), \
        "生产配置偏离了 SignalConfig 默认值 — 实盘口径与回测口径将再次分叉"


def test_daily_run_uses_shared_engine():
    """daily_run.py 必须经过 LiveRunner.check → compute_signals"""
    src = open(os.path.join(ETF_DIR, "live", "daily_run.py"), encoding="utf-8").read()
    assert "LiveRunner" in src and "production_config" in src
    runner_src = open(os.path.join(ETF_DIR, "live", "runner.py"), encoding="utf-8").read()
    assert "compute_signals" in runner_src
    assert "from strategy.signal import" in runner_src


def test_runner_blocks_stale_qqq():
    """QQQ-CNY 陈旧时必须显式回退+标注, 禁止静默 ffilled 混合"""
    runner_src = open(os.path.join(ETF_DIR, "live", "runner.py"), encoding="utf-8").read()
    assert "qqq_cny_freshness" in runner_src
    assert "数据陈旧" in runner_src


def test_runner_matches_backtest_prep():
    """runner.check 必须与 bt_runner 同口径: 预热段 + 决策区切片 + pre_days + accel_th 预计算
    (2026-09-11 实测对齐: 同输入下两条链路 actions/nav 逐位一致)"""
    src = open(os.path.join(ETF_DIR, "live", "runner.py"), encoding="utf-8").read()
    assert "WARMUP_DAYS = 320" in src, "预热窗口须与 bt_runner._warmup 下限一致"
    assert "pre_days=pre_days" in src, "必须传 pre_days (恐慌阈值/MA200 用全量历史)"
    assert "calc_accel_thresholds" in src, "加速止盈阈值必须全量预计算后切片"
    assert "[:, pre_days:]" in src, "决策区矩阵必须切片"


def test_reconcile_replay_window_matches_production():
    """对账回放的窗口必须与实盘/基线同源 — 2026-10-12 事故:
    '今天-120天' 的短窗口把 expanding σ 样本从 2686 缩到 274 天, 恐慌抄底触发线
    从 -8.35% 抬到 -5.88% → 09-28 假触发抄底创业, 连报 3 天假"口径分歧"。"""
    from live.runner import LiveRunner
    from live import reconcile

    assert reconcile.REPLAY_START == LiveRunner.DECISION_START, \
        "对账回放起点偏离实盘决策区起点 — 阈值会再次漂移"

    # 事故窗口 (120 天 ≈ 274 个交易日) 必须被抓出来, 正常窗口放行
    assert reconcile.replay_window_problem(274, "2026-06-15") is not None
    assert reconcile.replay_window_problem(2500, LiveRunner.DECISION_START) is None

    src = open(os.path.join(ETF_DIR, "live", "reconcile.py"), encoding="utf-8").read()
    assert "days=120" not in src, "禁止再出现'今天-120天'式短窗口"


def test_reconcile_flags_short_window(monkeypatch):
    """窗口体检必须挂在对账主流程上: 短窗口 → 报"窗口过短", 且不拿错口径仓位做对比"""
    import web.bt_runner as bt
    from live import notify, reconcile

    calls = {}
    monkeypatch.setattr(bt, "run_backtest",
                        lambda p: {"last_position": "创业",
                                   "dates": ["2026-06-15"] * 274})
    monkeypatch.setattr(notify, "archive",
                        lambda kind, msg, extra=None: calls.update(msg=msg, extra=extra))
    monkeypatch.setattr(notify, "send_dingtalk", lambda m: None)
    monkeypatch.setattr(reconcile, "REPLAY_START", "2026-06-15")

    assert reconcile.run_reconcile(push=False) is False
    problems = calls["extra"]["problems"]
    assert any("回放窗口过短" in p for p in problems), problems
    assert not any("口径分歧" in p for p in problems), \
        "短窗口回放的仓位是假信号, 不得用于口径对比: %s" % problems


def test_cron_script_is_thin_wrapper():
    """旧 cron 脚本必须只是转发入口, 不得再含策略判断"""
    src = open(os.path.join(STOCK_BOT, "cron_etf_momentum.py"), encoding="utf-8").read()
    assert "daily_run" in src
    # 旧副本的策略关键词一个都不许留
    for kw in ["should_cash", "get_momentums", "MIN_HOLD", "stop_line_of"]:
        assert kw not in src, f"cron_etf_momentum.py 仍包含策略逻辑: {kw}"


def test_no_duplicate_strategy_impl_in_stockbot():
    """stock-bot 可执行代码里禁止再出现简化版空仓判断"""
    for path in [os.path.join(STOCK_BOT, "app.py"),
                 os.path.join(STOCK_BOT, "services", "bot_reply.py")]:
        src = open(path, encoding="utf-8").read()
        assert "cron_etf_momentum import get_momentums" not in src, \
            f"{path} 仍 import 旧副本"
        assert "sh_mom[0] > etf_max" not in src, \
            f"{path} 仍包含简化版上证空仓判断"
        assert "current_view" in src, f"{path} 未接入共享查询视图"


def test_live_page_uses_shared_view():
    """实盘实况页只做展示装配, 数据必须来自 live.query.current_view"""
    app_src = open(os.path.join(ETF_DIR, "web", "app.py"), encoding="utf-8").read()
    assert '"/live"' in app_src and '"/api/live/today"' in app_src
    page_src = open(os.path.join(ETF_DIR, "web", "live_page.py"), encoding="utf-8").read()
    assert "current_view" in page_src
    # 展示层不得出现策略判断
    for kw in ["compute_signals", "sh_mom >", "exception_gap", "min_hold"]:
        assert kw not in page_src, f"live_page.py 不得包含策略逻辑: {kw}"


def test_suggestion_anchors_actual_position():
    """操作建议必须以实际持仓为锚 (2026-09-11 事故: 按模拟仓位说'卖出纳指',
    用户实际从未持有纳指, 推送造成恐慌)"""
    from live.daily_run import suggestion_for

    # 核心事故场景: 模拟仓=纳指, 实际仓=黄金 → 绝不允许出现"卖出纳指"
    text, level = suggestion_for(
        {"signal": "BUY", "target": "黄金", "sim_position": "黄金",
         "prev_position": "纳指", "sim_entry_date": "2026-09-11"},
        {"position": "黄金"})
    assert "纳指" not in text, f"操作建议不得引用用户未持有的标的: {text}"
    assert text == "已持有 黄金，无需操作"

    # 策略空仓 + 实际持仓 → 让用户对齐卖出
    assert suggestion_for({"signal": "HOLD", "sim_position": "", "sim_entry_date": ""},
                          {"position": "黄金"}) == ("卖出 黄金，空仓（对齐策略）", "cash")
    # 空仓信号 + 实际持仓 → 卖出实际持仓
    assert suggestion_for({"signal": "CASH", "sim_position": ""},
                          {"position": "黄金"}) == ("卖出 黄金，空仓", "cash")
    # 空仓信号 + 实际已空仓
    assert suggestion_for({"signal": "CASH", "sim_position": ""},
                          {"position": None}) == ("保持空仓", "wait")
    # 策略持仓 + 实际空仓 → 提示买入对齐
    assert suggestion_for({"signal": "HOLD", "sim_position": "黄金", "sim_entry_date": "2026-08-11"},
                          {"position": None}) == ("买入 黄金（对齐策略仓位）", "switch")
    # 实际持仓 ≠ 策略持仓 (漏跟信号) → 明确切换, 不得落回"继续空仓观望"
    # (2026-10-15: 用户实际纳指 / 策略创业, 旧逻辑给出"继续空仓观望")
    assert suggestion_for({"signal": "HOLD", "sim_position": "创业", "sim_entry_date": "2026-10-14"},
                          {"position": "纳指"}) == ("卖出 纳指，买入 创业（对齐策略仓位）", "switch")


def test_stop_line_anchored_to_actual_cost():
    """止损线必须按【用户实际成本】算, 不得把策略模拟仓的止损价印在实际持仓行上

    2026-10-21 事故: 引擎按其模拟入场 2.891(10-14 恐慌抄底) 触发 -5% 止损,
    用户实际成本 2.786 当时只有 -2.62%; 推送却把纳指的止损线 2.195 印在创业持仓
    那一行, 用户据此以为"策略判我亏了 5%" (2026-10-22 用户质疑)。"""
    from live.daily_run import build_message

    res = {
        "date": "2026-10-22", "signal": "HOLD", "target": None, "reason": "",
        "momentums": {"纳指": 2.23, "黄金": -6.18, "创业": -21.29},
        "sh_momentum": -5.76, "sim_position": "纳指", "prev_position": "纳指",
        "stop_price": 2.1954, "nz_mix_active": True,
        "nz_note": "0.7×QQQ-CNY(+2.21%) + 0.3×513100",
        "live_prices": {"纳指": 2.267, "创业": 2.617, "黄金": 8.358},
    }
    # 实际持创业(成本 2.786) / 策略模拟仓纳指 → 两个止损线都要标清楚
    msg = build_message(res, {"position": "创业", "entry_date": "2026-10-15",
                              "entry_price": 2.786})
    assert "实盘止损线(按你的成本) ¥2.647" in msg, msg        # 2.786 × 0.95
    assert "策略模拟仓[纳指]止损线 ¥2.195" in msg, msg
    assert "卖出 创业，买入 纳指（对齐策略仓位）" in msg, msg

    # 实盘=模拟仓时只显示引擎止损线, 不引入第二套口径
    msg2 = build_message(res, {"position": "纳指", "entry_date": "2026-10-21",
                               "entry_price": 2.312})
    assert "引擎止损线 ¥2.195" in msg2, msg2
    assert "实盘止损线" not in msg2, msg2


def test_data_integrity_flags_bad_and_missing_days():
    """数据体检: 非交易日脏行 / 缺失交易日都必须报出来 (2026-10-15 事故)

    事故链: 000001.SH 一条周六行 → 对齐日历多一天 → 其余标的该日全 NaN →
    创业板指 MA200 全 NaN → 恐慌抄底被静默否决; 且 14:50 推送与 21:30 收盘回放
    读同一份脏缓存, 一致地错。体检必须进推送正文, 不能再只写日志。"""
    import pandas as pd
    from live.daily_run import data_integrity_problems

    open_days = set(pd.date_range("2026-09-01", "2026-10-15", freq="B"))

    clean = pd.DataFrame({"trade_date": sorted(open_days)})
    assert data_integrity_problems({"513100.SH": clean}, open_days) == []

    dirty = pd.DataFrame({"trade_date": sorted(open_days | {pd.Timestamp("2026-09-12")})})
    probs = data_integrity_problems({"000001.SH": dirty}, open_days)
    assert probs == ["000001.SH 含 1 个非交易日行: 2026-09-12"], probs

    missing = pd.DataFrame({"trade_date": sorted(open_days - {pd.Timestamp("2026-09-10")})})
    probs = data_integrity_problems({"159915.SZ": missing}, open_days)
    assert probs == ["159915.SZ 缺 1 个交易日: 2026-09-10"], probs


def test_crash_unknown_is_three_state_and_surfaced():
    """恐慌抄底三态: 缺数据必须报"无法判定", 不能与"未触发"混为一谈

    旧写法 `mom < th and px < ma` 遇 NaN 恒为 False, 无异常无 warning
    (IEEE 754: NaN 比较合法且静默), 于是脏数据失效看起来就是一次普通 HOLD。"""
    import numpy as np
    from conftest import build_inputs, gbm_prices
    from strategy.signal import SignalConfig, compute_signals, CRASH_DIAG

    n = 400
    prices = {"纳指": gbm_prices(n, 100, 0.0008, 0.008, seed=1),
              "黄金": gbm_prices(n, 100, 0.0004, 0.006, seed=2),
              "创业": gbm_prices(n, 100, 0.0006, 0.014, seed=3)}
    cyb = gbm_prices(n, 3000, 0.0003, 0.010, seed=4)

    # 干净数据: 无"无法判定"事件 (预热期 MA200 未就绪不算异常)
    inp = build_inputs(prices, {"cyb": cyb, "sh": gbm_prices(n, 3000, 0.0002, 0.008, seed=5)})
    compute_signals(SignalConfig(), **inp)
    assert CRASH_DIAG == [], f"干净数据不应报异常: {CRASH_DIAG[:3]}"

    # 指数序列尾部有数据空洞 → 必须记为"无法判定"
    holed = cyb.copy(); holed[390] = np.nan
    inp = build_inputs(prices, {"cyb": holed, "sh": gbm_prices(n, 3000, 0.0002, 0.008, seed=5)})
    compute_signals(SignalConfig(), **inp)
    labels = {e[1] for e in CRASH_DIAG}
    assert "创业板指" in labels, f"应报创业板指无法判定: {CRASH_DIAG}"
    assert any("动量" in e[2] or "指数价" in e[2] for e in CRASH_DIAG), CRASH_DIAG

    # 推送正文必须把"无法判定"写出来
    from live.daily_run import build_message
    msg = build_message({"date": "2026-10-15", "signal": "HOLD", "target": None,
                         "reason": "", "momentums": {"纳指": 1.0}, "sh_momentum": 0.0,
                         "sim_position": "创业", "prev_position": "创业",
                         "live_prices": {}, "nz_mix_active": True, "nz_note": "x",
                         "crash_unknown": ["创业板指 缺MA200(近200日窗口内有空洞)"]},
                        {"position": "纳指"})
    assert "恐慌抄底今日无法判定" in msg and "不等于未触发" in msg, msg
