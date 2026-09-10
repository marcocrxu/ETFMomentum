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
