"""只读查询视图 — 供 stock-bot 的钉钉/QQ/微信 bot 应答和 HTTP API 使用

唯一数据源 = live.runner.LiveRunner.check() (内部调共享引擎 strategy/signal.py)。
stock-bot 里禁止再出现任何自行实现的动量/空仓判断逻辑
(2026-09-10 事故: bot_reply.py/app.py 各有一套简化版判断, 与主策略口径不一致,
 群里推"空仓"的同时 cron 推"持有", 用户收到互相矛盾的指令)。
"""
import sys
import time

_ETF_STRATEGY = "/home/ecs-user/etf_strategy"
if _ETF_STRATEGY not in sys.path:
    sys.path.insert(0, _ETF_STRATEGY)

_CACHE = {"ts": 0.0, "view": None}
_CACHE_TTL = 60.0  # 秒; bot 高频查询不重复跑全量回放


def current_view(refresh=False):
    """返回 LiveRunner.check() 的结果 dict (带 60s 缓存)。

    关键字段: momentums, sh_momentum, signal/target/reason (今日信号),
    sim_position (策略仓位), nz_mix_active (QQQ混合是否生效), live_prices。
    """
    now = time.time()
    if not refresh and _CACHE["view"] is not None and now - _CACHE["ts"] < _CACHE_TTL:
        return _CACHE["view"]

    from live.daily_run import production_config, UNIVERSE, _tushare_token, CACHE_DIR
    from data.feed import TushareFeed
    from live.runner import LiveRunner

    feed = TushareFeed(_tushare_token(), cache_dir=CACHE_DIR)
    runner = LiveRunner(production_config(), feed, UNIVERSE)
    # require_trade_day=False: bot 在周末/盘后查询时也给最近交易日口径的视图
    view = runner.check(require_trade_day=False)
    _CACHE["ts"] = now
    _CACHE["view"] = view
    return view
