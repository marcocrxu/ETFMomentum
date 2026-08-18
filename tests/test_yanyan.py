"""yanyan 引擎单测 — 全合成数据, 无网络"""
import numpy as np
import pytest

from strategy.yanyan import YanyanConfig, compute_yanyan, rolling_vol20


def _prices(drifts, n=300, noise=0.002, seed=0):
    """drifts: 各标的 (n,) 日收益序列; 加正弦噪声保证 vol20 非零且相近"""
    rng = np.random.default_rng(seed)
    mat = []
    for d in drifts:
        rets = np.asarray(d, dtype=float) + noise * np.sin(np.arange(n))
        mat.append(100 * np.exp(np.cumsum(rets)))
    return np.array(mat)


def _base_universe():
    names = ["A股1", "A股2", "跨境", "黄金", "货币"]
    groups = ["A", "A", "X", "G", "C"]
    return names, groups


def _flat(n):
    return np.zeros(n)


def test_nav_log_aligned_and_positive():
    names, groups = _base_universe()
    n = 300
    drifts = [np.full(n, 0.002), _flat(n), _flat(n), _flat(n), np.full(n, 0.0001)]
    pm = _prices(drifts, n)
    hs = 3000 * np.exp(np.cumsum(np.full(n, 0.0002)))
    cfg = YanyanConfig()
    actions, nav, log = compute_yanyan(cfg, names, groups, pm, hs, n)
    # vs 起点: hs MA200 就绪 (t=199)
    assert len(log) == n - 199, f"nav_log 长度 {len(log)} != {n - 199}"
    assert nav > 0 and all(v > 0 for v in log)


def test_bear_regime_blocks_a_pool():
    """牛转熊: A股持仓被牛熊开关清仓, 转持跨境/黄金"""
    names, groups = _base_universe()
    n = 300
    a1 = np.full(n, 0.004)              # A股1 强势 (牛市持有)
    x = np.full(n, 0.002)               # 跨境次强
    g = np.full(n, 0.001)               # 黄金
    a2 = _flat(n); c = np.full(n, 0.0001)
    pm = _prices([a1, a2, x, g, c], n)
    hs = 3000 * np.exp(np.cumsum(np.full(n, 0.0005)))
    hs[230:] = hs[230] * np.linspace(1, 0.75, n - 230)   # t=230 起崩 25% (引擎先入场后转熊)
    cfg = YanyanConfig()
    actions, _, _ = compute_yanyan(cfg, names, groups, pm, hs, n)
    bear_exit = [a for a in actions if a[3] == "牛熊开关"]
    assert bear_exit, f"熊市未触发牛熊开关: {[a[3] for a in actions]}"
    t_exit = bear_exit[0][0]
    assert bear_exit[0][1] == 0, "被清仓的应是 A 股持仓"
    after = [a for a in actions if a[0] >= t_exit]
    assert all(groups[a[2]] != "A" for a in after if a[2] >= 0), \
        f"熊市后不应再持 A 股池: {[(x[0], x[2], x[3]) for x in after]}"
    assert any(a[2] in (2, 3) for a in after), "熊市应转持跨境池/黄金"


def test_exit_on_mom20_negative():
    """持仓 MOM20 转负 → 动量转负离场"""
    names, groups = _base_universe()
    n = 300
    n = 450
    a1 = np.zeros(n)
    a1[220:300] = 0.004                    # 先涨 → 引擎入场
    a1[300:330] = -0.03                    # 后暴跌 → mom20 转负
    a2 = np.full(n, 0.001)
    x = np.full(n, 0.001)
    g = np.full(n, 0.001)
    c = np.full(n, 0.0001)
    pm = _prices([a1, a2, x, g, c], n)
    hs = 3000 * np.exp(np.cumsum(np.full(n, 0.0003)))
    cfg = YanyanConfig()
    actions, _, _ = compute_yanyan(cfg, names, groups, pm, hs, n)
    exits = [a for a in actions if a[3] == "动量转负" and a[1] == 0]
    assert exits, f"mom20 转负未离场: {[a[3] for a in actions]}"


def test_switch_buffer():
    """缓冲: 挑战者 MOM20 须超持仓 2% 才切换"""
    names, groups = _base_universe()
    n = 300
    a1 = np.full(n, 0.0025)                # 持仓, mom20 ≈ 5%
    a2 = np.zeros(n)
    a2[200:] = 0.0055                      # t=200 起加速 → 逐步逼近并超过 2% 缓冲
    x = _flat(n)
    g = _flat(n)
    c = np.full(n, 0.0001)
    pm = _prices([a1, a2, x, g, c], n)
    hs = 3000 * np.exp(np.cumsum(np.full(n, 0.0002)))
    cfg = YanyanConfig(switch_buffer=2.0)
    actions, _, _ = compute_yanyan(cfg, names, groups, pm, hs, n)
    sw = [a for a in actions if a[3] == "轮动切换" and a[2] == 1]
    assert sw, f"缓冲满足后未切换: {[a[3] for a in actions]}"
    # 切换日挑战者 mom20 - 持仓 mom20 >= 2
    t = sw[0][0]
    m = np.array([np.full(n, np.nan)] * 5)
    from strategy.yanyan import mom
    for i in range(5):
        m[i] = mom(pm[i], 20)
    gap = m[1, t] - m[0, t]
    assert gap >= 2.0 - 1e-9, f"切换日 gap {gap:.2f} < 2"


def test_overheat_exit_and_reentry_gate():
    """过热: MOM20>40% 且 MOM5<0 → 过热离场; MOM5>Q 前禁止再进场"""
    names, groups = _base_universe()
    n = 400
    n = 500
    a1 = np.zeros(n)
    a1[250:350] = 0.035                    # 暴涨 20 天窗口 => mom20 > 40%
    a1[350:355] = -0.01                    # 5 天回调 => mom5 < 0, mom20 仍 > 40
    a1[355:] = 0.008                       # 其后快速回升 (mom5 逐步 > Q)
    a2 = _flat(n)
    x = _flat(n)
    g = _flat(n)
    c = np.full(n, 0.0001)
    pm = _prices([a1, a2, x, g, c], n)
    hs = 3000 * np.exp(np.cumsum(np.full(n, 0.0002)))
    cfg = YanyanConfig(overheat_mom=40.0, overheat_q=3.0, cash_skip_buffer=False)
    actions, _, _ = compute_yanyan(cfg, names, groups, pm, hs, n)
    oh = [a for a in actions if a[3] == "过热离场" and a[1] == 0]
    assert oh, f"过热离场未触发: {[a[3] for a in actions]}"
    t_exit = oh[0][0]
    # 离场后到 MOM5>Q 之前的再进场应被闸门拦下: 观察是否有过早的"进场 A股1"
    from strategy.yanyan import mom
    m5 = mom(pm[0], 5)
    q_day = next((t for t in range(t_exit, n) if not np.isnan(m5[t]) and m5[t] > cfg.overheat_q), n)
    early = [a for a in actions if a[3] == "进场" and a[2] == 0 and t_exit < a[0] < q_day]
    assert not early, f"过热闸门未拦再进场: {[(a[0], a[2], a[3]) for a in early]}"
    # 若闸门解除后该标的仍 score 第一且 MOM20>0, 应允许再进场 (不强制断言, 只验证闸门期)


def test_use_vol_off_changes_score():
    """use_vol=False: score=纯MOM20; 场景构造让 vol 归一翻转排名:
    A股1 低波动中动量 (vol归一后 score 高), A股2 高波动高动量 (纯动量排名高)"""
    names, groups = _base_universe()
    n = 300
    i = np.arange(n)
    rets1 = np.full(n, 0.002) + 0.001 * np.sin(i)      # 低波动, mom20≈4%
    rets2 = np.full(n, 0.004) + 0.004 * np.sin(i)      # 高波动, mom20≈8%
    noise = 0.0002 * np.sin(i)                            # 微量噪声防 vol≈0 除零爆炸
    pm = np.array([100 * np.exp(np.cumsum(rets1)),
                   100 * np.exp(np.cumsum(rets2)),
                   100 * np.exp(np.cumsum(np.full(n, 0.0001) + noise)),
                   100 * np.exp(np.cumsum(np.full(n, 0.0001) + noise)),
                   100 * np.exp(np.cumsum(np.full(n, 0.0001) + noise))])
    hs = 3000 * np.exp(np.cumsum(np.full(n, 0.0002)))
    cfg_v = YanyanConfig(cash_skip_buffer=False)
    cfg_m = YanyanConfig(cash_skip_buffer=False, use_vol=False)
    a1, _, _ = compute_yanyan(cfg_v, names, groups, pm, hs, n)
    a2, _, _ = compute_yanyan(cfg_m, names, groups, pm, hs, n)
    e1 = next((x[2] for x in a1 if x[3] == "进场"), None)
    e2 = next((x[2] for x in a2 if x[3] == "进场"), None)
    assert e1 == 0, f"vol 归一应首选低波动 A股1, 实际 {e1}"
    assert e2 == 1, f"纯动量应首选高动量 A股2, 实际 {e2}"
