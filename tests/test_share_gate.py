"""份额风险门 (share_gate) 测试 — 合成数据, 无网络

覆盖:
1. build_flow_from 的 T+1 无前视纪律 (决策日 t 只能用 t-1 份额)
2. 篮子多 code 等权平均 / 单 code 缺失降级
3. signal 门: 净赎回期拦 A 股入场、放行纳指/黄金、数据缺失(NaN)不拦
"""
import numpy as np
import pandas as pd
import pytest

from conftest import build_inputs
from data.sharefeed import build_flow_from
from strategy.signal import SignalConfig, compute_signals


# ── build_flow_from ─────────────────────────────────────────
def _dates(n, start="2024-01-01"):
    return list(pd.date_range(start, periods=n, freq="B"))


def test_flow_t_plus_1_no_lookahead():
    """决策日 t 的 flow 只能用 t-1 及更早份额: 篡改 t 日份额不得影响 t 日 flow"""
    n = 60
    dates = _dates(n)
    shares = np.full(n, 100.0)
    shares[30:] = 130.0                      # t=30 起份额 +30%
    df = pd.DataFrame({"date": dates, "510300": shares})
    flow = build_flow_from(df, dates, ["510300"], window=10)
    # t=29: 看的是 29 日(shift后=28日)vs 18日, 都在跳变前 -> ~0
    assert abs(flow[29]) < 1e-9
    # t=40: shift后=39日(130) vs 29日(100) -> +30%
    assert flow[40] == pytest.approx(30.0, abs=1e-6)
    # 窗口未满 -> NaN
    assert np.isnan(flow[10])


def test_flow_basket_mean_and_missing_code():
    """篮子=逐code等权平均; 缺 code 的列自动跳过; 全缺返回全 NaN"""
    n = 40
    dates = _dates(n)
    df = pd.DataFrame({
        "date": dates,
        "510300": np.linspace(100, 110, n),   # 缓涨
        "510500": np.linspace(100, 130, n),   # 快涨
    })
    flow2 = build_flow_from(df, dates, ["510300", "510500", "510050"], window=10)
    flow1 = build_flow_from(df, dates, ["510300"], window=10)
    # 510050 不在表中 -> 被跳过; 两 code 均值介于各自之间
    i = 30
    assert not np.isnan(flow2[i])
    assert min(flow1[i], 0) <= flow2[i]       # 与单 code 同号
    empty = build_flow_from(pd.DataFrame(columns=["date"]), dates, ["510300"], 10)
    assert np.isnan(empty).all()


# ── signal 门 ───────────────────────────────────────────────
def _leader_market(n=120, leader="创业"):
    """创业(或纳指)持续领涨的合成市场; 上证平稳不触发 sh_cash"""
    up = np.linspace(100, 100 * 1.30, n)          # +30% 直线, mom20 ~ +5%
    flat = np.linspace(100, 100 * 1.02, n)
    prices = {"纳指": flat, "黄金": flat, "创业": flat}
    prices[leader] = up
    index = {"cyb": np.linspace(3000, 3090, n), "sh": np.linspace(3000, 3060, n)}
    return build_inputs(prices, index)


def _flow(n, value):
    arr = np.full(n, value)
    return {"share_flow": arr}


def test_gate_blocks_ashare_entry():
    """创业领涨 + 篮子深度净赎回 -> 不买入创业 (保持空仓)"""
    inp = _leader_market(leader="创业")
    n = inp["n_days"]
    cfg = SignalConfig(share_gate=3.0)
    actions, _, _ = compute_signals(cfg, **inp, pre_mats=_flow(n, -10.0))
    cyb_i = inp["etf_names"].index("创业")
    assert all(a[2] != cyb_i for a in actions), "净赎回期不应买入创业"


def test_gate_off_buys_normally():
    """门关闭(默认) -> 同样市场应买入领涨的创业"""
    inp = _leader_market(leader="创业")
    actions, _, _ = compute_signals(SignalConfig(), **inp,
                                    pre_mats=_flow(inp["n_days"], -10.0))
    cyb_i = inp["etf_names"].index("创业")
    assert any(a[2] == cyb_i for a in actions), "门关闭时应正常买入创业"


def test_gate_open_flow_ok_buys():
    """门开启但篮子净申购 -> 不拦, 正常买入创业"""
    inp = _leader_market(leader="创业")
    cfg = SignalConfig(share_gate=3.0)
    actions, _, _ = compute_signals(cfg, **inp, pre_mats=_flow(inp["n_days"], 5.0))
    cyb_i = inp["etf_names"].index("创业")
    assert any(a[2] == cyb_i for a in actions), "净申购期不应拦截"


def test_gate_exempts_nasdaq():
    """纳指领涨 + 净赎回 -> 纳指不受门影响 (QDII份额无国家队信息)"""
    inp = _leader_market(leader="纳指")
    cfg = SignalConfig(share_gate=3.0)
    actions, _, _ = compute_signals(cfg, **inp, pre_mats=_flow(inp["n_days"], -10.0))
    nz_i = inp["etf_names"].index("纳指")
    assert any(a[2] == nz_i for a in actions), "纳指应被豁免"


def test_gate_nan_degrades_gracefully():
    """份额数据缺失(NaN, 如2016年前) -> 不拦, 优雅降级"""
    inp = _leader_market(leader="创业")
    cfg = SignalConfig(share_gate=3.0)
    actions, _, _ = compute_signals(cfg, **inp,
                                    pre_mats={"share_flow": np.full(inp["n_days"], np.nan)})
    cyb_i = inp["etf_names"].index("创业")
    assert any(a[2] == cyb_i for a in actions), "数据缺失日不应拦截"


def test_gate_names_scope():
    """share_gate_names 显式限定受限标的"""
    inp = _leader_market(leader="创业")
    cfg = SignalConfig(share_gate=3.0, share_gate_names="红利低波")  # 创业不在名单
    actions, _, _ = compute_signals(cfg, **inp, pre_mats=_flow(inp["n_days"], -10.0))
    cyb_i = inp["etf_names"].index("创业")
    assert any(a[2] == cyb_i for a in actions), "名单外标的不应被拦"
