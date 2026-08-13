"""核心引擎不变量测试 (合成数据, 无网络)"""
import numpy as np
import pytest

from strategy.signal import SignalConfig, compute_signals
from conftest import build_inputs, gbm_prices, replay_nav


# ── 1. 自洽性: 独立重放必须复现引擎 nav_log ──
class TestReplaySelfConsistency:
    def test_calm_market(self, base_cfg, calm_market):
        actions, final_nav, nav_log = compute_signals(base_cfg, **calm_market)
        rep = replay_nav(actions, calm_market["price_mat"],
                         calm_market["etf_names"], base_cfg)
        vs = calm_market["n_days"] - len(nav_log)
        np.testing.assert_allclose(rep[vs:], nav_log, atol=1e-10,
                                   err_msg="引擎 nav_log 与独立重放不一致")

    def test_partial_crash_composite(self, calm_market):
        """crash_alloc=50 的组合仓路径也必须自洽 (Bug E 回归)"""
        cfg = SignalConfig(crash_alloc=0.5)
        # 制造恐慌: 创业板指在 300 天后 5 天跌 10%
        n = calm_market["n_days"]
        cyb = calm_market["cyb_idx_p"].copy()
        cyb[300:] = cyb[300] * np.linspace(1, 0.90, n - 300)
        inp = dict(calm_market, cyb_idx_p=cyb)
        actions, final_nav, nav_log = compute_signals(cfg, **inp)
        assert any(a[3] == "恐慌抄底" for a in actions), "合成恐慌未触发抄底"
        rep = replay_nav(actions, inp["price_mat"], inp["etf_names"], cfg)
        vs = n - len(nav_log)
        np.testing.assert_allclose(rep[vs:], nav_log, atol=1e-10)


# ── 2. nav_log 逐日对齐 (Bug F 回归): 数据缺口日不得吞掉 nav_log ──
def test_nav_log_aligned_with_nan_gap(base_cfg, calm_market):
    inp = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in calm_market.items()}
    # 在决策区中部注入一天全 NaN 动量 (所有 ETF 当天价格缺失)
    gap_day = 250
    inp["price_mat"][:, gap_day] = np.nan
    inp["mom20_mat"][:, gap_day] = np.nan
    actions, _, nav_log = compute_signals(base_cfg, **inp)
    vs = _expected_vs(inp)
    assert len(nav_log) == inp["n_days"] - vs, \
        f"NaN 缺口日导致 nav_log 错位: {len(nav_log)} != {inp['n_days'] - vs}"


def _expected_vs(inp):
    valid_cnt = np.sum(~np.isnan(inp["mom20_mat"]), axis=0)
    cand = np.where(valid_cnt >= min(2, len(inp["etf_names"])))[0]
    vs = cand[0] if len(cand) else 0
    fv = np.where(~np.isnan(inp["sh_mom"]))[0]
    if len(fv):
        vs = max(vs, fv[0])
    return vs


# ── 3. 恐慌抄底锁仓语义 ──
def _crash_scenario(n=400, crash_start=300, lock_drops=False):
    """纳指慢牛(会被持有), 创业板指 crash_start 起暴跌触发恐慌抄底"""
    prices = {
        "纳指": gbm_prices(n, 100, 0.001, 0.004, seed=11),   # 稳定最强
        "黄金": gbm_prices(n, 100, 0.0001, 0.003, seed=12),
        "创业": gbm_prices(n, 100, 0.0002, 0.008, seed=13),
    }
    sh = np.full(n, 3000.0) * np.exp(np.cumsum(np.full(n, 0.0001)))
    cyb = np.full(n, 3000.0)
    drop = np.linspace(1, 0.78, n - crash_start)      # 大跌22%触发恐慌
    cyb[crash_start:] = cyb[crash_start] * drop
    if lock_drops:
        # 抄底后创业ETF本身先急跌后缓跌 (锁仓期内PnL快速跌破-5%)
        first = np.linspace(1, 0.90, 10)               # 前10天急跌10%
        second = np.linspace(0.90, 0.88, n - crash_start - 10)  # 后续缓跌
        prices["创业"][crash_start:] *= np.concatenate([first, second])
    return build_inputs(prices, {"cyb": cyb, "sh": sh})


class TestCrashLock:
    def test_lock_blocks_sh_cash_and_rotation(self):
        """锁仓期内: 上证空仓/轮动不得卖掉抄底仓 (2026-08 修复的语义)"""
        cfg = SignalConfig(crash_lock=7)
        inp = _crash_scenario()
        # 锁仓期内让上证动量飙升 (上证空仓信号), 同时纳指动量最强
        sh = inp["sh_idx_p"].copy()
        sh[295:] = sh[295] * np.exp(np.cumsum(np.full(len(sh) - 295, 0.004)))
        inp = dict(inp, sh_idx_p=sh,
                   sh_mom=np.where(np.arange(len(sh)) >= 20,
                                   (sh / np.roll(sh, 20) - 1) * 100, np.nan))
        actions, _, _ = compute_signals(cfg, **inp)
        crash_day = next(a[0] for a in actions if a[3] == "恐慌抄底")
        during = [a for a in actions if crash_day < a[0] < crash_day + cfg.crash_lock]
        bad = [a for a in during if a[3] in ("上证空仓", "正常切换", ">3%切换", "加速止盈")]
        assert not bad, f"锁仓期内出现禁止动作: {bad}"

    def test_lock_allows_stop_loss(self):
        """锁仓期内唯一允许的退出: 止损"""
        cfg = SignalConfig(crash_lock=7)
        inp = _crash_scenario(lock_drops=True)
        actions, _, _ = compute_signals(cfg, **inp)
        crash_day = next(a[0] for a in actions if a[3] == "恐慌抄底")
        during = [a for a in actions if crash_day < a[0] < crash_day + cfg.crash_lock]
        assert any(a[3] == "止损" for a in during), "锁仓期止损保护未触发"

    def test_already_holding_cyb_no_duplicate_buy(self):
        """已持创业时恐慌刷新锁仓但不重复交易"""
        cfg = SignalConfig()
        inp = _crash_scenario()
        # 让创业成为动量第一: 纳指走平
        prices = {"纳指": np.full(400, 100.0), "黄金": np.full(400, 100.0),
                  "创业": gbm_prices(400, 100, 0.001, 0.004, seed=21)}
        cyb_idx = np.full(400, 3000.0)
        cyb_idx[350:] *= np.linspace(1, 0.85, 50)
        inp = build_inputs(prices, {"cyb": cyb_idx, "sh": np.full(400, 3000.0)})
        actions, _, _ = compute_signals(cfg, **inp)
        crashes = [a for a in actions if a[3] == "恐慌抄底" and a[1] != a[2]]
        assert len(crashes) <= 1, f"重复抄底: {crashes}"

    def test_crash_lock_not_silently_extended(self):
        """Bug G 回归: 已持创业时二次恐慌不刷新锁仓期, 保证锁仓到期后可正常轮动"""
        n = 500
        # 纳指最强, 创业弱但 index 持续暴跌 → 先恐慌买入创业, index 继续跌二次恐慌,
        # 锁仓应始终为第一次的 expire, 到期后纳指动量差足够 → 正常切换
        prices = {"纳指": gbm_prices(n, 100, 0.0015, 0.006, seed=91),
                  "黄金": gbm_prices(n, 100, 0.0001, 0.003, seed=92),
                  "创业": np.full(n, 100.0)}
        sh = np.full(n, 3000.0)
        cyb = np.full(n, 3000.0)
        # 分两段: 先跌触发首次恐慌, 持续跌触发二次(同持), 然后企稳
        cyb[300:320] = cyb[300] * np.exp(np.cumsum(np.full(20, -0.02)))  # 急跌20天
        cyb[320:340] = cyb[320] * np.exp(np.cumsum(np.full(20, -0.015))) # 续跌(二次恐慌)
        cyb[340:] = cyb[340]                                            # 企稳
        inp = build_inputs(prices, {"cyb": cyb, "sh": sh})
        cfg = SignalConfig(crash_lock=7, crash_alloc=1.0,
                           sh_cash=False, accel_enabled=False, abs_stop_loss=False)
        actions, _, _ = compute_signals(cfg, **inp)

        crashes = [a for a in actions if a[3] == "恐慌抄底"]
        assert len(crashes) == 1, \
            f"已持创业时二次恐慌应不产生新动作, 实际: {[(a[0],a[3]) for a in crashes]}"

        # 首次恐慌后的第 crash_lock 天应可正常切换到纳指
        crash_t = crashes[0][0]
        lock_end = crash_t + cfg.crash_lock + 1
        post_lock = [a for a in actions if a[0] > lock_end]
        switches = [a for a in post_lock
                    if a[3] in ("正常切换", ">3%切换") and a[2] == inp["etf_names"].index("纳指")]
        assert len(switches) >= 1, \
            f"锁仓到期后应可切换到纳指, post_lock actions: {[(a[0], inp['etf_names'][a[2]] if a[2]>=0 else '-', a[3]) for a in post_lock]}"


# ── 4. 恐慌抄底费用口径 (Bug E 回归) ──
def test_crash_entry_double_sided_fee():
    """已持其他标的时抄底(alloc=1)入场应扣双边费 (1-c)², 与正常切换一致"""
    cfg = SignalConfig(commission=0.001, slippage=0.0, crash_alloc=1.0)
    inp = _crash_scenario()
    actions, _, nav_log = compute_signals(cfg, **inp)
    crash = next(a for a in actions if a[3] == "恐慌抄底" and a[1] >= 0)
    t = crash[0]
    # 当日净值比 = 旧仓当日收益 × (1-c)²
    old = crash[1]
    p0, p1 = inp["price_mat"][old, t - 1], inp["price_mat"][old, t]
    day_ret = p1 / p0
    c = cfg.commission
    expect = nav_log[t - 1 - (inp["n_days"] - len(nav_log))] * day_ret * (1 - c) ** 2
    vs = inp["n_days"] - len(nav_log)
    assert nav_log[t - vs] == pytest.approx(expect, rel=1e-10), \
        "抄底入场费不是双边 (1-c)²"


# ── 5. 绝对止损盲区补丁 ──
def test_abs_stop_loss_exits_when_still_top():
    """持仓仍是动量第一但 pnl≤stop 时, abs_stop_loss 清到空仓 (不再硬扛)"""
    n = 400
    # 全部资产前200天慢涨, 后200天齐跌。纳指跌最慢 → 始终 ti，但 PnL 仍破 -5%
    base = gbm_prices(200, 100, 0.002, 0.008, seed=31)
    tail_base = base[-1]
    tail_t = np.arange(200)
    prices = {
        "纳指": np.concatenate([base,
            tail_base * np.exp(np.cumsum(np.full(200, -0.004)))]),   # -0.4%/day
        "黄金": np.concatenate([base,
            tail_base * np.exp(np.cumsum(np.full(200, -0.006)))]),   # 跌更快
        "创业": np.concatenate([base,
            tail_base * np.exp(np.cumsum(np.full(200, -0.007)))]),
    }
    inp = build_inputs(prices, {
        "cyb": np.full(n, 3000.0) * np.exp(np.cumsum(np.full(n, -0.0001))),
        "sh": np.full(n, 3000.0) * np.exp(np.cumsum(np.full(n, -0.0001))),
    })
    cfg = SignalConfig(abs_stop_loss=True, stop_loss=-5.0,
                       crash_enabled=False, sh_cash=False)
    actions, _, _ = compute_signals(cfg, **inp)
    abs_stops = [a for a in actions if a[3] == "绝对止损" and a[2] == -1]
    assert len(abs_stops) >= 1, f"绝对止损未触发: {[a[3] for a in actions]}"

    cfg_off = SignalConfig(abs_stop_loss=False, stop_loss=-5.0,
                           crash_enabled=False, sh_cash=False)
    actions2, _, _ = compute_signals(cfg_off, **inp)
    assert not any(a[3] == "绝对止损" for a in actions2), "开关关闭仍触发"


# ── 6. pre_mats 消除冷启动 (Bug D 回归) ──
def test_premats_accel_no_coldstart():
    """传入全量预算的 accel_th 时, 窗口第0天阈值即可用; 内部冷启动则需60天"""
    from strategy.signal import calc_accel_thresholds
    n_full, pre = 400, 300   # 决策区从 t=300 开始
    prices = {"纳指": gbm_prices(n_full, 100, 0.001, 0.010, seed=41),
              "黄金": gbm_prices(n_full, 100, 0.0005, 0.008, seed=42)}
    full = build_inputs(prices, {"cyb": gbm_prices(n_full, 3000, 0.0003, 0.010, seed=43),
                                 "sh": gbm_prices(n_full, 3000, 0.0002, 0.008, seed=44)})
    cfg = SignalConfig()
    accel_full = calc_accel_thresholds(full["mom10_mat"], cfg)
    # 切片传入 (模拟 bt_runner 行为)
    pre_mats = {"accel_th": accel_full[:, pre:]}
    assert not np.isnan(pre_mats["accel_th"][:, 0]).all(), \
        "pre_mats 切片后窗口首日阈值仍 NaN (冷启动未消除)"


# ── 7. Top-K 引擎自洽 ──
def test_topk_runs_and_navlog_aligned(calm_market):
    cfg = SignalConfig(top_k=2)
    actions, final_nav, nav_log = compute_signals(cfg, **calm_market)
    vs = _expected_vs(calm_market)
    assert len(nav_log) == calm_market["n_days"] - vs
    assert final_nav > 0


# ── 8. 止损后同标的冷却 (stop_cooldown) ──
def _stop_scenario(n=400):
    """全部资产前200天慢涨后200天齐跌; 纳指跌最慢始终动量第一 → 反复触发绝对止损"""
    base = gbm_prices(200, 100, 0.002, 0.008, seed=31)
    tail_base = base[-1]
    prices = {
        "纳指": np.concatenate([base,
            tail_base * np.exp(np.cumsum(np.full(200, -0.004)))]),
        "黄金": np.concatenate([base,
            tail_base * np.exp(np.cumsum(np.full(200, -0.006)))]),
        "创业": np.concatenate([base,
            tail_base * np.exp(np.cumsum(np.full(200, -0.007)))]),
    }
    return build_inputs(prices, {
        "cyb": np.full(n, 3000.0) * np.exp(np.cumsum(np.full(n, -0.0001))),
        "sh": np.full(n, 3000.0) * np.exp(np.cumsum(np.full(n, -0.0001))),
    })


def test_stop_cooldown_blocks_same_symbol_reentry():
    """冷却=10: 绝对止损后10天内不得买回同一标的; 冷却=0: 2天确认后即买回"""
    inp = _stop_scenario()
    base = dict(abs_stop_loss=True, stop_loss=-5.0,
                crash_enabled=False, sh_cash=False, accel_enabled=False)

    # 冷却关闭: 纳指被绝对止损后很快又被买回 (确认买入, 2天)
    cfg0 = SignalConfig(stop_cooldown=0, **base)
    actions0, _, _ = compute_signals(cfg0, **inp)
    stop0 = next(a for a in actions0 if a[3] == "绝对止损" and a[1] == 0)
    rebuy0 = next((a for a in actions0 if a[2] == 0 and a[0] > stop0[0]), None)
    assert rebuy0 is not None and rebuy0[0] - stop0[0] <= 4, \
        f"冷却=0 时止损后未快速买回: stop={stop0}, rebuy={rebuy0}"

    # 冷却=10: 同一标的 10 天内不得买回, 但引擎应正常买入未冷却候选
    cfg10 = SignalConfig(stop_cooldown=10, **base)
    actions10, _, _ = compute_signals(cfg10, **inp)
    stop10 = next(a for a in actions10 if a[3] == "绝对止损" and a[1] == 0)
    bad = [a for a in actions10
           if a[2] == 0 and stop10[0] < a[0] < stop10[0] + 10]
    assert not bad, f"止损冷却期内买回同一标的: {bad}"
    others = [a for a in actions10
              if a[2] in (1, 2) and stop10[0] < a[0] <= stop10[0] + 4]
    assert others, "冷却期内引擎未切向未冷却候选 (错误地冻结了)"


def test_stop_cooldown_replay_consistent():
    """开启冷却不改变记账口径: 独立重放必须复现 nav_log"""
    cfg = SignalConfig(stop_cooldown=10, abs_stop_loss=True, stop_loss=-5.0,
                       crash_enabled=False, sh_cash=False, accel_enabled=False)
    inp = _stop_scenario()
    actions, _, nav_log = compute_signals(cfg, **inp)
    rep = replay_nav(actions, inp["price_mat"], inp["etf_names"], cfg)
    vs = inp["n_days"] - len(nav_log)
    np.testing.assert_allclose(rep[vs:], nav_log, atol=1e-10)


def test_stop_cooldown_topk():
    """Top-K: 止损卖出的标的冷却期内不得作为替换候选买回
    场景时间线(确定性): 黄金 t=59 止损→创业补位; 创业 t≈69 止损时黄金已反弹回 top2,
    冷却=0 当天买回黄金; 冷却=12 则冷却期内不得买回"""
    n = 140
    nz = np.exp(np.cumsum(np.full(n, 0.002)))
    hj = np.exp(np.cumsum(np.r_[np.full(50, 0.002),
                                np.full(12, -0.012),      # 急跌逼出止损
                                np.full(n - 62, 0.012)]))  # 强反弹回到 top2
    cy = np.exp(np.cumsum(np.r_[np.full(64, 0.0005),
                                np.full(15, -0.012),      # 补位后也止损
                                np.full(n - 79, 0.0005)]))
    inp = build_inputs({"纳指": 100 * nz, "黄金": 100 * hj, "创业": 100 * cy},
                       {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})
    base = dict(top_k=2, stop_loss=-5.0,
                crash_enabled=False, sh_cash=False, accel_enabled=False)

    cd = 12
    cfg = SignalConfig(stop_cooldown=cd, **base)
    actions, _, _ = compute_signals(cfg, **inp)
    stops = [a for a in actions if a[3] == "止损" and a[1] >= 0]
    assert stops, f"Top-K 场景未触发止损: {[a[3] for a in actions]}"
    for t, sold, _, _ in stops:
        bad = [a for a in actions if a[2] == sold and t < a[0] < t + cd]
        assert not bad, f"冷却期内买回被止损标的 {sold}: {bad}"

    # 对照组: 冷却=0 时黄金在止损后 12 天内即被买回 (证明场景有区分度)
    cfg0 = SignalConfig(stop_cooldown=0, **base)
    actions0, _, _ = compute_signals(cfg0, **inp)
    t0 = next(a[0] for a in actions0 if a[3] == "止损" and a[1] == 1)
    assert any(a[2] == 1 and t0 < a[0] < t0 + cd for a in actions0), \
        "冷却=0 时黄金未被快速买回, 场景无区分度"
