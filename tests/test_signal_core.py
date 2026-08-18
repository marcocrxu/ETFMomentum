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

# ── 9. 快速通道分档切换 (fast_lane_staged) ──
def _staged_inputs(jump_day=204, jump=0.017, jump_len=2, drift=0.0008, drift_len=40):
    """纳指: 前180天强(被持有)→80天走平→140天缓复苏; 黄金: 与纳指并列0动量,
    jump_day 起跳涨 jump_len 天制造 3~5% gap (正常切换1%门槛来不及抢跑), 其后 drift;
    创业: 常平。指数常平 (sh_cash/crash 不干扰)"""
    n = 400
    nz = np.r_[np.full(180, 0.004), np.full(80, 0.0), np.full(n - 260, 0.001)]
    gold = np.zeros(n)
    gold[jump_day:jump_day + jump_len] = jump
    gold[jump_day + jump_len:jump_day + jump_len + drift_len] = drift
    prices = {
        "纳指": 100 * np.exp(np.cumsum(nz)),
        "黄金": 100 * np.exp(np.cumsum(gold)),
        "创业": np.full(n, 100.0),
    }
    return build_inputs(prices, {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})


def _staged_cfg(**over):
    base = dict(fast_lane_staged=True, fast_full_gap=5.0,
                accel_enabled=False, sh_cash=False, crash_enabled=False)
    base.update(over)
    return SignalConfig(**base)


def test_staged_off_unchanged():
    """默认关闭: 快速通道仍为全切, 无组合仓动作"""
    inp = _staged_inputs()
    cfg_off = SignalConfig(accel_enabled=False, sh_cash=False, crash_enabled=False)
    actions, _, _ = compute_signals(cfg_off, **inp)
    assert any(a[3] == ">3%切换" for a in actions), \
        f"基线快速通道未触发: {[a[3] for a in actions]}"
    assert not any(a[3] == ">3%切换(50%)" for a in actions)


def test_staged_half_then_revert():
    """gap∈[3,5)% → 切50%成组合; 核心(纳指)重新领先≥min_gap → 卫星回撤全回核心"""
    inp = _staged_inputs()
    cfg = _staged_cfg()
    actions, _, nav_log = compute_signals(cfg, **inp)
    nz_i, gold_i = inp["etf_names"].index("纳指"), inp["etf_names"].index("黄金")
    half = [a for a in actions if a[3] == ">3%切换(50%)"]
    assert half and half[0][1] == nz_i and half[0][2] == gold_i, \
        f"未切50%到黄金: {[a[3] for a in actions]}"
    assert not any(a[3] == "分档升级全切" for a in actions), "gap<5% 不应升级全切"
    rev = [a for a in actions if a[3] == "卫星回撤"]
    assert rev and rev[0][2] == nz_i, f"核心重新领先后未回撤: {[a[3] for a in actions]}"
    rep = replay_nav(actions, inp["price_mat"], inp["etf_names"], cfg)
    vs = _expected_vs(inp)
    np.testing.assert_allclose(rep[vs:], nav_log, atol=1e-10)


def test_staged_escalate():
    """gap 冲到≥5% → 卫星升级全切"""
    inp = _staged_inputs(drift=0.003)
    cfg = _staged_cfg()
    actions, _, nav_log = compute_signals(cfg, **inp)
    gold_i = inp["etf_names"].index("黄金")
    half = [a for a in actions if a[3] == ">3%切换(50%)"]
    assert half, f"未先切50%: {[a[3] for a in actions]}"
    esc = [a for a in actions if a[3] == "分档升级全切"]
    assert esc and esc[0][2] == gold_i, f"未升级全切: {[a[3] for a in actions]}"
    assert not any(a[3] == "卫星回撤" for a in actions)
    rep = replay_nav(actions, inp["price_mat"], inp["etf_names"], cfg)
    vs = _expected_vs(inp)
    np.testing.assert_allclose(rep[vs:], nav_log, atol=1e-10)


def test_staged_gap_over_full_keeps_full_switch():
    """首日 gap 即≥5%: 分档开启也不切50%, 直接全切 (旧行为)"""
    inp = _staged_inputs(jump=0.03, drift=0.0, drift_len=0)
    cfg = _staged_cfg()
    actions, _, _ = compute_signals(cfg, **inp)
    assert any(a[3] == ">3%切换" for a in actions), \
        f"gap≥5% 应直接全切: {[a[3] for a in actions]}"
    assert not any(a[3] == ">3%切换(50%)" for a in actions)


def test_staged_half_fee_is_half_turnover():
    """50%切换日费用 = 当日收益 × (1-c×0.5)²"""
    inp = _staged_inputs()
    cfg = _staged_cfg(commission=0.001, slippage=0.0)
    actions, _, nav_log = compute_signals(cfg, **inp)
    half = next(a for a in actions if a[3] == ">3%切换(50%)")
    t, base = half[0], half[1]
    vs = _expected_vs(inp)
    p0, p1 = inp["price_mat"][base, t - 1], inp["price_mat"][base, t]
    expect = nav_log[t - 1 - vs] * (p1 / p0) * (1 - 0.001 * 0.5) ** 2
    assert nav_log[t - vs] == pytest.approx(expect, rel=1e-10), \
        "50%切换费用不是半换手双边"


def test_staged_combo_crash_takeover():
    """组合仓期间恐慌爆发: 清双腿(恐慌换仓) → 标准恐慌抄底, 组合仓不再存在"""
    inp = _staged_inputs()
    cyb = np.full(inp["n_days"], 3000.0)
    cyb[230:] = 3000 * np.linspace(1, 0.80, inp["n_days"] - 230)
    inp = dict(inp, cyb_idx_p=cyb)
    cfg = SignalConfig(fast_lane_staged=True, fast_full_gap=5.0,
                       accel_enabled=False, sh_cash=False)
    actions, _, nav_log = compute_signals(cfg, **inp)
    assert any(a[3] == "恐慌换仓" for a in actions), \
        f"组合仓未在恐慌日清仓: {[a[3] for a in actions]}"
    crash = [a for a in actions if a[3] == "恐慌抄底"]
    assert crash, "恐慌换仓后未抄底"
    rep = replay_nav(actions, inp["price_mat"], inp["etf_names"], cfg)
    vs = _expected_vs(inp)
    np.testing.assert_allclose(rep[vs:], nav_log, atol=1e-10)


# ── 10. 快速通道追涨闸门 (fast_lane_ma) ──
def _bounce_below_ma_inputs(n=200):
    """纳指: 常平(被持有, mom20≈0); 黄金: 100天匀速阴跌(现价远低于MA60),
    t=120 起 2 天急反弹制造 ≥3pp gap; 创业: 常平"""
    nz = np.zeros(n)
    gold = np.zeros(n)
    for i in range(20, 120):
        gold[i] = -0.012 if i % 2 == 0 else 0.002     # 锯齿阴跌, 均价-0.005/天
    gold[120:122] = 0.065                              # 2天急反弹
    gold[122:] = 0.0                                   # 其后走平
    prices = {
        "纳指": 100 * np.exp(np.cumsum(nz)),
        "黄金": 100 * np.exp(np.cumsum(gold)),
        "创业": np.full(n, 100.0),
    }
    inp = build_inputs(prices, {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})
    # 场景前提: 反弹次日 gap≥3 且现价 < MA60 (闸门应拦); 其后价格随MA60下移而上穿(闸门应放行)
    t = 121
    g_mom = inp["mom20_mat"][1, t] - inp["mom20_mat"][0, t]
    ma60 = np.mean(inp["price_mat"][1, t - 59: t + 1])
    assert g_mom >= 3, f"场景未制造≥3pp gap: {g_mom:.2f}"
    assert inp["price_mat"][1, t] < ma60, f"反弹次日黄金应在MA60之下: {inp['price_mat'][1, t]:.3f} vs {ma60:.3f}"
    return inp


def _bounce_cfg(**over):
    base = dict(accel_enabled=False, sh_cash=False, crash_enabled=False)
    base.update(over)
    return SignalConfig(**base)


def test_fast_ma_gate_blocks_while_below_ma60():
    """默认(60): 挑战者低于自身MA60期间不追涨; 高min_hold下无慢通道兜底, 快速切换被推迟"""
    inp = _bounce_below_ma_inputs()
    cfg_on = _bounce_cfg(min_hold=200, fast_lane_ma=60)
    cfg_off = _bounce_cfg(min_hold=200, fast_lane_ma=0)
    a_on, _, _ = compute_signals(cfg_on, **inp)
    a_off, _, _ = compute_signals(cfg_off, **inp)
    t_off = next(a[0] for a in a_off if a[3] == ">3%切换")
    fast_on = [a[0] for a in a_on if a[3] == ">3%切换"]
    assert fast_on, "闸门开启后价格上穿MA60应恢复快速通道"
    assert fast_on[0] > t_off, \
        f"闸门应推迟追涨: 关闭时t={t_off}切, 开启时t={fast_on[0]}切 (低于MA60期间不得追)"


def test_fast_ma_gate_falls_back_to_normal_lane():
    """默认min_hold(10)下: 被闸门拦住的快速切换落回慢通道 (持有≥10天正常切换)"""
    inp = _bounce_below_ma_inputs()
    cfg = _bounce_cfg(fast_lane_ma=60)              # min_hold=10
    actions, _, _ = compute_signals(cfg, **inp)
    first_sw = next(a for a in actions if a[3] in (">3%切换", "正常切换"))
    assert first_sw[3] == "正常切换", \
        f"低于MA60时不应走快速通道, 应落回慢通道: {first_sw[3]}"


def test_fast_ma_gate_passes_above_ma60():
    """挑战者高于自身MA60: 闸门放行, 快速通道正常触发"""
    n = 200
    nz = np.zeros(n)
    gold = np.zeros(n)
    gold[100:] = 0.003                              # 黄金 t=100 起持续上涨 (价格高于MA60)
    prices = {"纳指": 100 * np.exp(np.cumsum(nz)), "黄金": 100 * np.exp(np.cumsum(gold)),
              "创业": np.full(n, 100.0)}
    inp = build_inputs(prices, {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})
    cfg = _bounce_cfg(fast_lane_ma=60, min_gap=2.9)  # 抬高慢通道门槛, 让快速通道先触发
    actions, _, _ = compute_signals(cfg, **inp)
    assert any(a[3] == ">3%切换" for a in actions), \
        f"高于MA60的挑战者应被放行: {[a[3] for a in actions]}"


# ── 11. VIX 恐慌门 (vix_gate / vix_spike_exit) ──
def test_vix_gate_blocks_nz_entry():
    """VIX>门槛时禁止新买纳指; VIX 回落后恢复"""
    from conftest import build_inputs
    n = 400
    nz = 100 * np.exp(np.cumsum(np.full(n, 0.004)))          # 纳指最强
    prices = {"纳指": nz, "黄金": 100 * np.exp(np.cumsum(np.full(n, 0.001))),
              "创业": 100 * np.exp(np.cumsum(np.full(n, 0.0005)))}
    inp = build_inputs(prices, {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})
    vix = np.where(np.arange(n) < 100, 30.0, 18.0)           # 前100天 VIX 高企
    cfg = SignalConfig(vix_gate=25.0, accel_enabled=False, sh_cash=False, crash_enabled=False)
    pre = {"vix": vix}
    actions, _, _ = compute_signals(cfg, inp["etf_names"], inp["price_mat"],
                                    inp["mom20_mat"], inp["mom10_mat"], inp["sh_mom"],
                                    inp["cyb_idx_p"], inp["sh_idx_p"], n, pre_mats=pre)
    entries = [a for a in actions if a[3] == ">3%即买" or a[3] == "确认买入"]
    assert entries and entries[0][0] >= 100, \
        f"VIX 高企期间不应买纳指, 首次买入 t={entries[0][0] if entries else None}"


def test_vix_spike_exit_sells_nz():
    """VIX 单日飙升≥阈值时清纳指"""
    from conftest import build_inputs
    n = 400
    prices = {"纳指": 100 * np.exp(np.cumsum(np.full(n, 0.004))),
              "黄金": 100 * np.exp(np.cumsum(np.full(n, 0.001))),
              "创业": 100 * np.exp(np.cumsum(np.full(n, 0.0005)))}
    inp = build_inputs(prices, {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})
    vix = np.full(n, 15.0)
    vix[250] = 20.0                                            # 单日 +33%
    cfg = SignalConfig(vix_spike_exit=20.0, accel_enabled=False, sh_cash=False, crash_enabled=False)
    actions, _, _ = compute_signals(cfg, inp["etf_names"], inp["price_mat"],
                                    inp["mom20_mat"], inp["mom10_mat"], inp["sh_mom"],
                                    inp["cyb_idx_p"], inp["sh_idx_p"], n, pre_mats={"vix": vix})
    assert any(a[3] == "VIX异动" and a[1] == 0 for a in actions), \
        f"VIX 异动未清纳指: {[a[3] for a in actions]}"
    # 默认关闭时不触发
    actions0, _, _ = compute_signals(SignalConfig(accel_enabled=False, sh_cash=False, crash_enabled=False),
                                     inp["etf_names"], inp["price_mat"], inp["mom20_mat"],
                                     inp["mom10_mat"], inp["sh_mom"], inp["cyb_idx_p"],
                                     inp["sh_idx_p"], n, pre_mats={"vix": vix})
    assert not any(a[3] == "VIX异动" for a in actions0)


# ── 12. 止盈再入场闸门 (accel_reentry_gate) ──
def test_accel_reentry_gate_delays_rebuy():
    """加速止盈后 mom10 仍高于阈值期间禁止买回; 回落下方后允许"""
    from conftest import build_inputs
    from strategy.signal import calc_accel_thresholds
    n = 300
    nz = 100 * np.exp(np.cumsum(np.full(n, 0.003)))              # 纳指稳涨 (常驻候选)
    gold = np.zeros(n)
    gold[80:100] = 0.05                                         # 黄金暴涨 → mom10 冲高触发止盈
    gold[100:110] = -0.005                                      # 高位横盘 (mom10 仍高)
    gold[110:] = 0.002                                          # 其后企稳 (mom10 逐步回落)
    prices = {"纳指": nz, "黄金": 100 * np.exp(np.cumsum(gold)),
              "创业": 100 * np.exp(np.cumsum(np.full(n, 0.0005)))}
    inp = build_inputs(prices, {"cyb": np.full(n, 3000.0), "sh": np.full(n, 3000.0)})
    base = dict(sh_cash=False, crash_enabled=False)
    cfg0 = SignalConfig(**base)
    cfg1 = SignalConfig(accel_reentry_gate=True, **base)
    accel_full = calc_accel_thresholds(inp["mom10_mat"], cfg0)
    a0, _, _ = compute_signals(cfg0, inp["etf_names"], inp["price_mat"], inp["mom20_mat"],
                               inp["mom10_mat"], inp["sh_mom"], inp["cyb_idx_p"],
                               inp["sh_idx_p"], n, pre_mats={"accel_th": accel_full})
    a1, _, _ = compute_signals(cfg1, inp["etf_names"], inp["price_mat"], inp["mom20_mat"],
                               inp["mom10_mat"], inp["sh_mom"], inp["cyb_idx_p"],
                               inp["sh_idx_p"], n, pre_mats={"accel_th": accel_full})
    exit0 = next((a for a in a0 if a[3] == "加速止盈" and a[1] == 1), None)
    assert exit0 is not None, f"场景未触发黄金止盈: {[a[3] for a in a0]}"
    t_exit = exit0[0]
    rebuy0 = next((a[0] for a in a0 if a[2] == 1 and a[0] > t_exit), None)
    rebuy1 = next((a[0] for a in a1 if a[2] == 1 and a[0] > t_exit), None)
    assert rebuy0 is not None, "无闸门时黄金应被快速买回"
    assert rebuy1 is None or rebuy1 > rebuy0, \
        f"闸门应推迟/阻止黄金买回: 无闸门 t={rebuy0}, 有闸门 t={rebuy1}"
