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
